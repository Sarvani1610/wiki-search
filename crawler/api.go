package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// APIClient talks to the MediaWiki action API with a shared rate limit.
type APIClient struct {
	base   string
	ua     string
	http   *http.Client
	tokens chan struct{}
}

func NewAPIClient(base, ua string, rps float64) *APIClient {
	c := &APIClient{
		base:   base,
		ua:     ua,
		http:   &http.Client{Timeout: 30 * time.Second},
		tokens: make(chan struct{}, 1),
	}
	if rps <= 0 {
		rps = 1
	}
	go func() {
		t := time.NewTicker(time.Duration(float64(time.Second) / rps))
		for range t.C {
			select {
			case c.tokens <- struct{}{}:
			default:
			}
		}
	}()
	return c
}

// Page is one fetched article.
type Page struct {
	ID        int64    `json:"id"`
	Title     string   `json:"title"`
	Text      string   `json:"text"`
	Links     []string `json:"links"`
	Redirects []string `json:"redirects,omitempty"`
	FetchedAt string   `json:"fetched_at"`
}

// ErrMissing means the title does not exist (or is not an article).
var ErrMissing = fmt.Errorf("page missing")

// RetryableError marks failures worth retrying (timeouts, 5xx, 429, maxlag).
type RetryableError struct {
	Err   error
	After time.Duration
}

func (e *RetryableError) Error() string { return e.Err.Error() }

type apiResponse struct {
	Continue map[string]string `json:"continue"`
	Error    *struct {
		Code string `json:"code"`
		Info string `json:"info"`
	} `json:"error"`
	Query struct {
		Redirects []struct {
			From string `json:"from"`
			To   string `json:"to"`
		} `json:"redirects"`
		Pages []struct {
			PageID  int64  `json:"pageid"`
			NS      int    `json:"ns"`
			Title   string `json:"title"`
			Missing bool   `json:"missing"`
			Extract string `json:"extract"`
			Links   []struct {
				NS    int    `json:"ns"`
				Title string `json:"title"`
			} `json:"links"`
		} `json:"pages"`
	} `json:"query"`
}

// Fetch returns the plain-text body and article-namespace links for a title,
// following redirects and link pagination up to maxLinks.
func (c *APIClient) Fetch(ctx context.Context, title string, maxLinks int) (*Page, error) {
	page := &Page{}
	params := url.Values{
		"action":        {"query"},
		"format":        {"json"},
		"formatversion": {"2"},
		"redirects":     {"1"},
		"titles":        {title},
		"prop":          {"extracts|links"},
		"explaintext":   {"1"},
		"plnamespace":   {"0"},
		"pllimit":       {"max"},
		"maxlag":        {"5"},
	}
	for {
		var resp apiResponse
		if err := c.get(ctx, params, &resp); err != nil {
			return nil, err
		}
		if resp.Error != nil {
			if resp.Error.Code == "maxlag" {
				return nil, &RetryableError{Err: fmt.Errorf("maxlag"), After: 5 * time.Second}
			}
			return nil, fmt.Errorf("api error %s: %s", resp.Error.Code, resp.Error.Info)
		}
		for _, r := range resp.Query.Redirects {
			page.Redirects = append(page.Redirects, r.From)
		}
		if len(resp.Query.Pages) == 0 {
			return nil, ErrMissing
		}
		p := resp.Query.Pages[0]
		if p.Missing || p.NS != 0 {
			return nil, ErrMissing
		}
		page.ID, page.Title = p.PageID, p.Title
		if p.Extract != "" {
			page.Text = p.Extract
		}
		for _, l := range p.Links {
			if l.NS == 0 && len(page.Links) < maxLinks {
				page.Links = append(page.Links, l.Title)
			}
		}
		if resp.Continue == nil || len(page.Links) >= maxLinks {
			break
		}
		// Only keep paging links; the extract is already in hand.
		params.Set("prop", "links")
		for k, v := range resp.Continue {
			params.Set(k, v)
		}
	}
	page.FetchedAt = time.Now().UTC().Format(time.RFC3339)
	return page, nil
}

func (c *APIClient) get(ctx context.Context, params url.Values, out interface{}) error {
	select {
	case <-c.tokens:
	case <-ctx.Done():
		return ctx.Err()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"?"+params.Encode(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", c.ua)
	req.Header.Set("Accept-Encoding", "gzip")
	resp, err := c.http.Do(req)
	if err != nil {
		return &RetryableError{Err: err, After: 2 * time.Second}
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
		after := 5 * time.Second
		if s, err := strconv.Atoi(resp.Header.Get("Retry-After")); err == nil {
			after = time.Duration(s) * time.Second
		}
		io.Copy(io.Discard, resp.Body)
		return &RetryableError{Err: fmt.Errorf("http %d", resp.StatusCode), After: after}
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("http %d", resp.StatusCode)
	}
	body, err := decodeBody(resp)
	if err != nil {
		return &RetryableError{Err: err, After: time.Second}
	}
	defer body.Close()
	return json.NewDecoder(body).Decode(out)
}
