package main

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type Config struct {
	RedisAddr, Prefix, APIURL, OutDir, UserAgent string
	Workers, MaxLinks, MaxAttempts, RotateEvery  int
	MaxPages, MaxFrontier                        int64
	RPS                                          float64
}

type Worker struct {
	ID       int
	Cfg      Config
	Frontier *Frontier
	API      *APIClient
	out      *pageWriter
}

// Run pulls titles until the page budget is reached, the queue stays empty,
// or ctx is cancelled.
func (w *Worker) Run(ctx context.Context) error {
	defer func() {
		if w.out != nil {
			w.out.Close()
		}
	}()
	idle := 0
	for ctx.Err() == nil {
		if n, err := w.Frontier.Stored(ctx); err == nil && n >= w.Cfg.MaxPages {
			return nil
		}
		title, ok, err := w.Frontier.Next(ctx, 5*time.Second)
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		if !ok {
			// The queue can be briefly empty while other workers are still
			// fetching pages whose links will refill it.
			if idle++; idle >= 6 {
				return nil
			}
			continue
		}
		idle = 0
		w.process(ctx, title)
	}
	return nil
}

func (w *Worker) process(ctx context.Context, title string) {
	page, err := w.API.Fetch(ctx, title, w.Cfg.MaxLinks)
	switch {
	case errors.Is(err, ErrMissing):
		return
	case err != nil:
		if ctx.Err() != nil {
			// Shutting down: hand the title back untouched.
			w.Frontier.rdb.LPush(context.Background(), w.Frontier.queue, title)
			return
		}
		var re *RetryableError
		if errors.As(err, &re) {
			time.Sleep(re.After)
		}
		gaveUp, rerr := w.Frontier.Retry(ctx, title, w.Cfg.MaxAttempts)
		if rerr == nil && gaveUp {
			log.Printf("worker %d: giving up on %q: %v", w.ID, title, err)
		}
		return
	}
	if page.Text == "" {
		return
	}
	// A redirect resolves to a canonical title that may be fetched under its
	// own name later; marking it seen avoids storing the article twice.
	w.Frontier.MarkSeen(ctx, append(page.Redirects, page.Title)...)
	if err := w.write(page); err != nil {
		log.Printf("worker %d: write: %v", w.ID, err)
		return
	}
	w.Frontier.IncStored(ctx)
	if _, err := w.Frontier.Enqueue(ctx, page.Links); err != nil {
		log.Printf("worker %d: enqueue: %v", w.ID, err)
	}
}

func (w *Worker) write(p *Page) error {
	if w.out == nil || w.out.n >= w.Cfg.RotateEvery {
		if w.out != nil {
			if err := w.out.Close(); err != nil {
				return err
			}
		}
		host, _ := os.Hostname()
		name := fmt.Sprintf("%s-%d-w%02d-%d.jsonl.gz", host, os.Getpid(), w.ID, time.Now().UnixNano())
		pw, err := newPageWriter(filepath.Join(w.Cfg.OutDir, name))
		if err != nil {
			return err
		}
		w.out = pw
	}
	return w.out.Write(p)
}

// pageWriter appends JSON lines to a gzip file. Each worker owns its own
// file, so writes need no locking. Files are written under a .part name and
// renamed on close, so readers only ever see complete files.
type pageWriter struct {
	path string
	f    *os.File
	gz   *gzip.Writer
	buf  *bufio.Writer
	enc  *json.Encoder
	n    int
}

func newPageWriter(path string) (*pageWriter, error) {
	f, err := os.Create(path + ".part")
	if err != nil {
		return nil, err
	}
	gz := gzip.NewWriter(f)
	buf := bufio.NewWriterSize(gz, 1<<20)
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	return &pageWriter{path: path, f: f, gz: gz, buf: buf, enc: enc}, nil
}

func (pw *pageWriter) Write(p *Page) error {
	pw.n++
	return pw.enc.Encode(p)
}

func (pw *pageWriter) Close() error {
	if err := pw.buf.Flush(); err != nil {
		return err
	}
	if err := pw.gz.Close(); err != nil {
		return err
	}
	if err := pw.f.Close(); err != nil {
		return err
	}
	return os.Rename(pw.path+".part", pw.path)
}

func decodeBody(resp *http.Response) (io.ReadCloser, error) {
	if resp.Header.Get("Content-Encoding") == "gzip" {
		return gzip.NewReader(resp.Body)
	}
	return io.NopCloser(resp.Body), nil
}
