#!/usr/bin/env python3
"""Audio-similarity sidecar for Navidrome: the database, the hourly scan and the
HTTP API the navidrome-sonic plugin queries. analysis.py turns a file into an
embedding and mood scores, index.py ranks tracks by them.

  GET /similar?size=&duration=&title=&count=N    tracks closest to one track
  GET /path?start_size=&...&end_size=&...&count=N a route from one track to another
  GET /mood?name=&count=N                        tracks with the highest score for a mood
  GET /health                                    index size, for the healthcheck

A track is looked up by its file size in bytes, the one key Navidrome hands a
plugin without filesystem access. Files of equal size are told apart by the
duration closest to `duration` (seconds), then by an exact `title`.

Answers are SongRef-shaped (title, artists, album, MBIDs, duration) so Navidrome
matches them to its own track IDs. `python3 test_sonic.py` runs the self-check.
"""

import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# numpy's OpenBLAS otherwise starts a spinning thread per visible CPU.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from analysis import MOODS, audio_hash, embed, load_models, probe  # noqa: E402
from index import Index  # noqa: E402

MUSIC = "/music"
DB = "/data/sonic.db"
# Bumped when a model or the analysis changes: the stored rows are dropped and rebuilt.
ANALYSIS_VERSION = 2
PORT = 8000
SCAN_INTERVAL = 3600
REINDEX_EVERY = 200  # rebuild the index during a long first pass, not only at its end
AUDIO_EXT = {".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".aac", ".wav",
             ".aif", ".aiff", ".wma", ".ape", ".wv", ".mpc", ".dsf", ".dff"}

INDEX = None


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)


def init_db(db):
    if db.execute("PRAGMA user_version").fetchone()[0] != ANALYSIS_VERSION:
        db.execute("DROP TABLE IF EXISTS track")
        db.execute(f"PRAGMA user_version = {ANALYSIS_VERSION}")
    db.execute("CREATE TABLE IF NOT EXISTS track "
               "(path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, meta TEXT, emb BLOB, audio TEXT, mood TEXT)")
    # Queries look tracks up by size, and the scan reuses embeddings by audio hash.
    db.execute("CREATE INDEX IF NOT EXISTS track_size ON track (size)")
    db.execute("CREATE INDEX IF NOT EXISTS track_audio ON track (audio)")


def load_index():
    global INDEX
    INDEX = Index(DB)
    log(f"index: {len(INDEX.paths)} tracks, threshold {INDEX.threshold:.4f}")


def raise_error(e):
    raise e


def scan(db, models):
    known = {p: (m, s) for p, m, s in db.execute("SELECT path, mtime, size FROM track")}
    seen, changed, analysed = set(), 0, 0
    # onerror raises, so an unreadable mount aborts the pass instead of reading as
    # an empty library and dropping every embedding.
    for root, dirs, files in os.walk(MUSIC, onerror=raise_error):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            # "._*" are macOS AppleDouble files that copy the audio extension.
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in AUDIO_EXT:
                continue
            path = os.path.join(root, name)
            st = os.stat(path)
            seen.add(path)
            if known.get(path) == (int(st.st_mtime), st.st_size):
                continue
            emb, mood, meta, audio = None, None, None, None
            try:
                duration, song, album_artist = probe(path)
                meta = json.dumps({"song": song, "albumArtist": album_artist})
                audio = audio_hash(path)
                # The same audio elsewhere, such as the file before it moved, or a
                # copy analysed earlier in this pass.
                emb, mood = db.execute("SELECT emb, mood FROM track WHERE audio = ? AND emb IS NOT NULL LIMIT 1",
                                       (audio,)).fetchone() or (None, None)
                if emb is None:
                    vec, scores = embed(models, path, duration)
                    analysed += 1
                    if vec is not None:
                        emb, mood = vec.tobytes(), json.dumps(scores)
            except Exception as e:  # one bad file must not stop the pass
                log(f"skip {path}: {e}")
            if emb is None:
                log(f"no embedding: {path}")
            # A failed file is stored too, so it is retried only once it changes.
            db.execute("INSERT OR REPLACE INTO track (path, mtime, size, meta, emb, audio, mood) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (path, int(st.st_mtime), st.st_size, meta, emb, audio or "", mood))
            db.commit()
            changed += 1
            if changed % REINDEX_EVERY == 0:
                load_index()
    gone = set(known) - seen
    if seen:
        db.executemany("DELETE FROM track WHERE path = ?", [(p,) for p in gone])
        db.commit()
    log(f"scan: {len(seen)} files, {changed} new or changed, {analysed} analysed, "
        f"{len(gone) if seen else 0} removed")
    return changed or (gone and seen)


def scanner():
    models = load_models()
    db = sqlite3.connect(DB)
    load_index()
    while True:
        try:
            if scan(db, models):
                load_index()
        except Exception as e:
            log(f"scan aborted: {e}")
        time.sleep(SCAN_INTERVAL)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        idx = INDEX
        try:
            count = max(1, min(int(q.get("count", 10)), 500))
            if url.path == "/health":
                return self.reply(200, {"tracks": len(idx.paths) if idx else 0})
            if idx is None:
                raise KeyError("index not loaded")
            seed = lambda p="": idx.find(q[p + "size"], q.get(p + "duration", 0), q.get(p + "title", ""))
            if url.path == "/similar":
                res = idx.similar(seed(), count)
            elif url.path == "/path":
                res = idx.route(seed("start_"), seed("end_"), count)
            elif url.path == "/mood":
                if q.get("name") not in MOODS:
                    raise ValueError(f"name must be one of {', '.join(MOODS)}")
                res = idx.mood(q["name"], count)
            else:
                return self.reply(404, {"error": "not found"})
        except KeyError as e:
            return self.reply(404, {"error": f"not analysed: {e}"})
        except ValueError as e:
            return self.reply(400, {"error": str(e)})
        self.reply(200, {"matches": [{"song": song, "similarity": s} for song, s in res]})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    init_db(db)
    db.close()
    threading.Thread(target=scanner, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
