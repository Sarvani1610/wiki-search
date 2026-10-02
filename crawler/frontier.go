package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/redis/go-redis/v9"
)

// Frontier is the crawl queue shared by every worker through Redis.
//
// Keys (with prefix "wiki"):
//
//	wiki:queue     LIST  titles waiting to be fetched
//	wiki:seen      SET   every title ever enqueued (dedupe)
//	wiki:attempts  HASH  title -> failed attempts
//	wiki:failed    LIST  titles that ran out of attempts
//	wiki:stored    STRING count of pages written to disk
type Frontier struct {
	rdb         *redis.Client
	queue       string
	seen        string
	attempts    string
	failed      string
	stored      string
	maxFrontier int64
}

func NewFrontier(rdb *redis.Client, prefix string, maxFrontier int64) *Frontier {
	k := func(s string) string { return prefix + ":" + s }
	return &Frontier{rdb, k("queue"), k("seen"), k("attempts"), k("failed"), k("stored"), maxFrontier}
}

// enqueueScript marks titles as seen and pushes only the new ones, in one
// atomic round trip, so two workers that discover the same link never both
// queue it. It stops adding once the queue reaches the frontier cap; titles
// skipped that way are left unseen so they can be discovered again later.
var enqueueScript = redis.NewScript(`
local cap = tonumber(ARGV[1])
local len = redis.call('LLEN', KEYS[2])
local added = 0
for i = 2, #ARGV do
  if len >= cap then break end
  if redis.call('SADD', KEYS[1], ARGV[i]) == 1 then
    redis.call('RPUSH', KEYS[2], ARGV[i])
    len = len + 1
    added = added + 1
  end
end
return added
`)

func (f *Frontier) Enqueue(ctx context.Context, titles []string) (int64, error) {
	if len(titles) == 0 {
		return 0, nil
	}
	args := make([]interface{}, 0, len(titles)+1)
	args = append(args, f.maxFrontier)
	for _, t := range titles {
		args = append(args, t)
	}
	return enqueueScript.Run(ctx, f.rdb, []string{f.seen, f.queue}, args...).Int64()
}

// MarkSeen records titles (e.g. redirect targets) without queueing them.
func (f *Frontier) MarkSeen(ctx context.Context, titles ...string) error {
	if len(titles) == 0 {
		return nil
	}
	args := make([]interface{}, len(titles))
	for i, t := range titles {
		args[i] = t
	}
	return f.rdb.SAdd(ctx, f.seen, args...).Err()
}

// Next blocks up to wait for a title. ok is false on timeout.
func (f *Frontier) Next(ctx context.Context, wait time.Duration) (title string, ok bool, err error) {
	res, err := f.rdb.BLPop(ctx, wait, f.queue).Result()
	if errors.Is(err, redis.Nil) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return res[1], true, nil
}

// Retry puts a title back on the queue, or on the failed list once it has
// used up its attempts.
func (f *Frontier) Retry(ctx context.Context, title string, maxAttempts int) (gaveUp bool, err error) {
	n, err := f.rdb.HIncrBy(ctx, f.attempts, title, 1).Result()
	if err != nil {
		return false, err
	}
	if int(n) >= maxAttempts {
		pipe := f.rdb.TxPipeline()
		pipe.RPush(ctx, f.failed, title)
		pipe.HDel(ctx, f.attempts, title)
		_, err = pipe.Exec(ctx)
		return true, err
	}
	return false, f.rdb.RPush(ctx, f.queue, title).Err()
}

func (f *Frontier) IncStored(ctx context.Context) (int64, error) {
	return f.rdb.Incr(ctx, f.stored).Result()
}

func (f *Frontier) Stored(ctx context.Context) (int64, error) {
	n, err := f.rdb.Get(ctx, f.stored).Int64()
	if errors.Is(err, redis.Nil) {
		return 0, nil
	}
	return n, err
}

func (f *Frontier) Reset(ctx context.Context) error {
	return f.rdb.Del(ctx, f.queue, f.seen, f.attempts, f.failed, f.stored).Err()
}

type Stats struct{ Stored, Queued, Seen, Failed int64 }

func (s Stats) String() string {
	return fmt.Sprintf("stored=%d queued=%d seen=%d failed=%d", s.Stored, s.Queued, s.Seen, s.Failed)
}

func (f *Frontier) Stats(ctx context.Context) (Stats, error) {
	pipe := f.rdb.Pipeline()
	stored := pipe.Get(ctx, f.stored)
	q := pipe.LLen(ctx, f.queue)
	s := pipe.SCard(ctx, f.seen)
	fl := pipe.LLen(ctx, f.failed)
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return Stats{}, err
	}
	n, _ := stored.Int64()
	return Stats{n, q.Val(), s.Val(), fl.Val()}, nil
}

func normalizeTitle(t string) string {
	t = strings.TrimSpace(strings.ReplaceAll(t, "_", " "))
	if t == "" {
		return ""
	}
	// MediaWiki titles are case-sensitive except for the first letter.
	r, size := utf8.DecodeRuneInString(t)
	return string(unicode.ToUpper(r)) + t[size:]
}
