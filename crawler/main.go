// Command crawler fetches Wikipedia articles through the MediaWiki API.
// Workers (goroutines in one process, or several processes on different
// machines) share a single frontier in Redis, so the crawl can be scaled out
// by starting more copies against the same Redis instance.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
)

func main() {
	cfg := Config{}
	var seeds string
	flag.StringVar(&cfg.RedisAddr, "redis", "localhost:6379", "Redis address")
	flag.StringVar(&cfg.Prefix, "prefix", "wiki", "Redis key prefix (lets several crawls share one Redis)")
	flag.StringVar(&cfg.APIURL, "api", "https://en.wikipedia.org/w/api.php", "MediaWiki API endpoint")
	flag.StringVar(&cfg.OutDir, "out", "data/pages", "directory for gzipped JSONL output")
	flag.StringVar(&cfg.UserAgent, "ua", "wiki-search-course-project/1.0 (contact: you@example.com)", "User-Agent sent to the API (Wikimedia requires a contact)")
	flag.IntVar(&cfg.Workers, "workers", 8, "concurrent workers in this process")
	flag.Int64Var(&cfg.MaxPages, "max", 1_200_000, "stop after this many pages are stored (across all processes)")
	flag.Int64Var(&cfg.MaxFrontier, "max-frontier", 5_000_000, "do not grow the queue beyond this many titles")
	flag.IntVar(&cfg.MaxLinks, "max-links", 500, "max outgoing links recorded/enqueued per page")
	flag.IntVar(&cfg.MaxAttempts, "attempts", 3, "attempts per title before it is moved to the failed list")
	flag.IntVar(&cfg.RotateEvery, "rotate", 50_000, "start a new output file after this many pages per worker")
	flag.Float64Var(&cfg.RPS, "rps", 20, "request rate limit for this process (requests/second)")
	flag.StringVar(&seeds, "seeds", "Computer science,Mathematics,History,Biology,Physics,Music,Geography,Philosophy", "comma-separated seed titles")
	reset := flag.Bool("reset", false, "delete this prefix's Redis keys before starting")
	flag.Parse()

	ctx, cancel := context.WithCancel(context.Background())
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		log.Println("shutting down: finishing in-flight pages")
		cancel()
	}()

	rdb := redis.NewClient(&redis.Options{Addr: cfg.RedisAddr})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("redis: %v", err)
	}
	f := NewFrontier(rdb, cfg.Prefix, cfg.MaxFrontier)
	if *reset {
		if err := f.Reset(ctx); err != nil {
			log.Fatalf("reset: %v", err)
		}
	}
	var titles []string
	for _, s := range strings.Split(seeds, ",") {
		if s = normalizeTitle(s); s != "" {
			titles = append(titles, s)
		}
	}
	if n, err := f.Enqueue(ctx, titles); err != nil {
		log.Fatalf("seed: %v", err)
	} else if n > 0 {
		log.Printf("seeded %d titles", n)
	}
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		log.Fatal(err)
	}

	client := NewAPIClient(cfg.APIURL, cfg.UserAgent, cfg.RPS)
	var wg sync.WaitGroup
	for i := 0; i < cfg.Workers; i++ {
		w := &Worker{ID: i, Cfg: cfg, Frontier: f, API: client}
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := w.Run(ctx); err != nil {
				log.Printf("worker %d: %v", w.ID, err)
			}
		}()
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	tick := time.NewTicker(10 * time.Second)
	defer tick.Stop()
	start := time.Now()
	for {
		select {
		case <-done:
			st, _ := f.Stats(context.Background())
			log.Printf("finished: %s in %s", st, time.Since(start).Round(time.Second))
			return
		case <-tick.C:
			st, err := f.Stats(ctx)
			if err == nil {
				rate := float64(st.Stored) / time.Since(start).Seconds()
				log.Printf("%s  (%.1f pages/s)", st, rate)
			}
		}
	}
}
