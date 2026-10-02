package main

import (
	"context"
	"os"
	"testing"

	"github.com/redis/go-redis/v9"
)

func TestNormalizeTitle(t *testing.T) {
	cases := map[string]string{
		"computer_science": "Computer science",
		"  éclair ":        "Éclair",
		"":                 "",
	}
	for in, want := range cases {
		if got := normalizeTitle(in); got != want {
			t.Errorf("normalizeTitle(%q) = %q, want %q", in, got, want)
		}
	}
}

// Needs a running Redis: REDIS_ADDR=localhost:6379 go test ./...
func TestFrontierDedupeAndCap(t *testing.T) {
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		t.Skip("REDIS_ADDR not set")
	}
	ctx := context.Background()
	f := NewFrontier(redis.NewClient(&redis.Options{Addr: addr}), "wikitest", 3)
	f.Reset(ctx)
	defer f.Reset(ctx)

	if n, _ := f.Enqueue(ctx, []string{"A", "B", "A"}); n != 2 {
		t.Fatalf("first enqueue added %d, want 2", n)
	}
	if n, _ := f.Enqueue(ctx, []string{"B", "C", "D", "E"}); n != 1 {
		t.Fatalf("capped enqueue added %d, want 1 (cap 3)", n)
	}
	// D was skipped by the cap, so it must still be discoverable later.
	if title, ok, _ := f.Next(ctx, 0); !ok || title != "A" {
		t.Fatalf("Next = %q, want A", title)
	}
	if n, _ := f.Enqueue(ctx, []string{"D"}); n != 1 {
		t.Fatalf("D should be enqueueable after the cap was hit")
	}
}
