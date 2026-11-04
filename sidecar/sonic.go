// The sidecar for the navidrome-sonic plugin: keeps an embedding for every
// audio file under /music and answers similarity queries over HTTP.
//
// Once an hour it walks the library and analyses new or changed files
// (analysis.go). Results go to SQLite, keyed by path and with a hash of the
// audio stream, so a moved or retagged file reuses its embedding instead of
// being analysed again. After a scan that changed something it rebuilds the
// index (index.go) and swaps it in; queries use the old one until then.
//
//	GET /similar?size=&duration=&title=&count=N    tracks closest to one track
//	GET /path?start_size=&...&end_size=&...&count=N a route from one track to another
//	GET /mood?name=&count=N                        tracks with the highest score for a mood
//	GET /health                                    index size, for the healthcheck
//
// Navidrome and the sidecar share no track IDs. A track is looked up by its
// file size in bytes, the one key Navidrome hands a plugin without filesystem
// access; files of equal size are told apart by the duration closest to
// `duration` (seconds), then by an exact `title`. Answers go back as tags
// (title, artists, album, MBIDs, duration), which Navidrome matches to its own
// tracks. `go test` runs the self-check.
package main

import (
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

var music = "/music"

const (
	dbPath = "/data/sonic.db"
	// Bumped when a model or the analysis changes: the stored rows are dropped and rebuilt.
	analysisVersion = 2
	port            = ":8000"
	scanInterval    = time.Hour
	reindexEvery    = 200 // rebuild the index during a long first pass, not only at its end
)

var audioExt = map[string]bool{".mp3": true, ".flac": true, ".ogg": true, ".oga": true, ".opus": true,
	".m4a": true, ".mp4": true, ".aac": true, ".wav": true, ".aif": true, ".aiff": true, ".wma": true,
	".ape": true, ".wv": true, ".mpc": true, ".dsf": true, ".dff": true}

var index atomic.Pointer[Index]

func openDB(path string) (*sql.DB, error) {
	return sql.Open("sqlite3", "file:"+path+"?_journal_mode=WAL&_busy_timeout=5000")
}

func initDB(db *sql.DB) error {
	var v int
	if err := db.QueryRow("PRAGMA user_version").Scan(&v); err != nil {
		return err
	}
	stmts := []string{
		"CREATE TABLE IF NOT EXISTS track " +
			"(path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, meta TEXT, emb BLOB, audio TEXT, mood TEXT)",
		// Queries look tracks up by size, and the scan reuses embeddings by audio hash.
		"CREATE INDEX IF NOT EXISTS track_size ON track (size)",
		"CREATE INDEX IF NOT EXISTS track_audio ON track (audio)",
	}
	if v != analysisVersion {
		stmts = append([]string{"DROP TABLE IF EXISTS track",
			"PRAGMA user_version = " + strconv.Itoa(analysisVersion)}, stmts...)
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	return nil
}

func loadIndex(db *sql.DB) error {
	idx, err := newIndex(db)
	if err != nil {
		return err
	}
	index.Store(idx)
	log.Printf("index: %d tracks, threshold %.4f", len(idx.paths), idx.threshold)
	return nil
}

// analyse returns a file's stored columns; a failed file keeps nil emb and mood.
func analyse(db *sql.DB, models map[string]model, path string) (meta, emb, mood any, audio string, analysed bool, err error) {
	duration, s, albumArtist, err := probe(path)
	if err != nil {
		return
	}
	m, _ := json.Marshal(map[string]any{"song": s, "albumArtist": albumArtist})
	meta = string(m)
	if audio, err = audioHash(path); err != nil {
		return
	}
	// The same audio elsewhere, such as the file before it moved, or a copy
	// analysed earlier in this pass.
	var e []byte
	var md string
	err = db.QueryRow("SELECT emb, mood FROM track WHERE audio = ? AND emb IS NOT NULL LIMIT 1", audio).Scan(&e, &md)
	if err == nil {
		return meta, e, md, audio, false, nil
	}
	if err != sql.ErrNoRows {
		return
	}
	vec, scores, err := embed(models, path, duration)
	if err != nil || vec == nil {
		return meta, nil, nil, audio, true, err
	}
	b := make([]byte, 4*len(vec))
	for i, x := range vec {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	j, _ := json.Marshal(scores)
	return meta, b, string(j), audio, true, nil
}

// scan analyses new and changed files under music and drops removed ones. It
// reports whether anything changed.
func scan(db *sql.DB, models map[string]model) (bool, error) {
	type stamp struct{ mtime, size int64 }
	known := map[string]stamp{}
	res, err := db.Query("SELECT path, mtime, size FROM track")
	if err != nil {
		return false, err
	}
	for res.Next() {
		var p string
		var s stamp
		if err := res.Scan(&p, &s.mtime, &s.size); err != nil {
			res.Close()
			return false, err
		}
		known[p] = s
	}
	if err := res.Err(); err != nil {
		return false, err
	}
	seen, changed, analysed := map[string]bool{}, 0, 0
	// An error aborts the pass instead of reading as an empty library and dropping
	// every embedding.
	err = filepath.WalkDir(music, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != music && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		// "._*" are macOS AppleDouble files that copy the audio extension.
		if strings.HasPrefix(d.Name(), ".") || !audioExt[strings.ToLower(filepath.Ext(d.Name()))] {
			return nil
		}
		st, err := os.Stat(path)
		if err != nil {
			return err
		}
		seen[path] = true
		now := stamp{st.ModTime().Unix(), st.Size()}
		if s, ok := known[path]; ok && s == now {
			return nil
		}
		// One bad file must not stop the pass.
		meta, emb, mood, audio, a, err := analyse(db, models, path)
		if a {
			analysed++
		}
		if err != nil {
			log.Printf("skip %s: %v", path, err)
		}
		if emb == nil {
			log.Printf("no embedding: %s", path)
		}
		// A failed file is stored too, so it is retried only once it changes.
		if _, err := db.Exec("INSERT OR REPLACE INTO track (path, mtime, size, meta, emb, audio, mood) VALUES (?, ?, ?, ?, ?, ?, ?)",
			path, now.mtime, now.size, meta, emb, audio, mood); err != nil {
			return err
		}
		if changed++; changed%reindexEvery == 0 {
			if err := loadIndex(db); err != nil {
				log.Printf("index: %v", err)
			}
		}
		return nil
	})
	if err != nil {
		return false, err
	}
	gone := 0
	if len(seen) > 0 {
		tx, err := db.Begin()
		if err != nil {
			return false, err
		}
		for p := range known {
			if !seen[p] {
				if _, err := tx.Exec("DELETE FROM track WHERE path = ?", p); err != nil {
					tx.Rollback()
					return false, err
				}
				gone++
			}
		}
		if err := tx.Commit(); err != nil {
			return false, err
		}
	}
	log.Printf("scan: %d files, %d new or changed, %d analysed, %d removed", len(seen), changed, analysed, gone)
	return changed > 0 || gone > 0, nil
}

func scanner(db *sql.DB) {
	models, err := loadModels()
	if err != nil {
		log.Fatalf("models: %v", err)
	}
	reload := func() {
		if err := loadIndex(db); err != nil {
			log.Printf("index: %v", err)
		}
	}
	reload()
	for {
		changed, err := scan(db, models)
		if err != nil {
			log.Printf("scan aborted: %v", err)
		}
		if changed {
			reload()
		}
		time.Sleep(scanInterval)
	}
}

func handle(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	reply := func(status int, body any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		json.NewEncoder(w).Encode(body)
	}
	fail := func(err error) {
		var nf notFound
		if errors.As(err, &nf) {
			reply(404, map[string]string{"error": err.Error()})
		} else {
			reply(400, map[string]string{"error": err.Error()})
		}
	}
	count := 10
	if c := q.Get("count"); c != "" {
		n, err := strconv.Atoi(c)
		if err != nil {
			fail(err)
			return
		}
		count = max(1, min(n, 500))
	}
	idx := index.Load()
	if r.URL.Path == "/health" {
		n := 0
		if idx != nil {
			n = len(idx.paths)
		}
		reply(200, map[string]int{"tracks": n})
		return
	}
	if idx == nil {
		fail(notFound("index not loaded"))
		return
	}
	seed := func(p string) (int, error) {
		if !q.Has(p + "size") {
			return 0, notFound(p + "size missing")
		}
		size, err := strconv.ParseInt(q.Get(p+"size"), 10, 64)
		if err != nil {
			return 0, err
		}
		duration := 0.0
		if d := q.Get(p + "duration"); d != "" {
			if duration, err = strconv.ParseFloat(d, 64); err != nil {
				return 0, err
			}
		}
		return idx.find(size, duration, q.Get(p+"title"))
	}
	var res []match
	var err error
	switch r.URL.Path {
	case "/similar":
		var s int
		if s, err = seed(""); err == nil {
			res, err = idx.similar(s, count)
		}
	case "/path":
		var s, e int
		if s, err = seed("start_"); err == nil {
			if e, err = seed("end_"); err == nil {
				res, err = idx.route(s, e, count)
			}
		}
	case "/mood":
		var names []string
		for _, m := range moods {
			names = append(names, m.name)
		}
		if slices.Contains(names, q.Get("name")) {
			res, err = idx.mood(q.Get("name"), count)
		} else {
			err = errors.New("name must be one of " + strings.Join(names, ", "))
		}
	default:
		reply(404, map[string]string{"error": "not found"})
		return
	}
	if err != nil {
		fail(err)
		return
	}
	reply(200, map[string][]match{"matches": res})
}

func main() {
	db, err := openDB(dbPath)
	if err != nil {
		log.Fatal(err)
	}
	if err := initDB(db); err != nil {
		log.Fatal(err)
	}
	go scanner(db)
	log.Fatal(http.ListenAndServe(port, http.HandlerFunc(handle)))
}
