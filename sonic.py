#!/usr/bin/env python3
"""Audio-similarity sidecar for Navidrome.

Walks /music, computes one MusicNN embedding per audio file and answers
nearest-neighbour queries over HTTP for the navidrome-sonic plugin. The analysis
follows LMS (src/libs/audio/impl/musicnn and
src/libs/services/recommendation/impl/audio-similarity): 16 kHz mono, log-mel
patches of 187 frames, up to 20 evenly spaced patches per track, the mean of their
200-D embeddings; queries run on a whitened 60-D PCA projection with cosine
distance.

  GET /similar?size=&duration=&title=&count=N    tracks closest to one track
  GET /path?start_size=&...&end_size=&...&count=N a route from one track to another
  GET /health                                    index size, for the healthcheck

A track is looked up by its file size in bytes, the one key Navidrome hands a
plugin without filesystem access. Files of equal size are told apart by the
duration closest to `duration` (seconds), then by an exact `title`.

Answers are SongRef-shaped (title, artists, album, MBIDs, duration) so Navidrome
matches them to its own track IDs. `python3 sonic.py test` runs the self-check.
"""

import io
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# numpy's OpenBLAS otherwise starts a spinning thread per visible CPU.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

MUSIC = "/music"
DB = "/data/sonic.db"
MODEL = "/opt/sonic/MSD_musicnn_embedding.onnx"
PORT = 8000
SCAN_INTERVAL = 3600
REINDEX_EVERY = 200  # rebuild the index during a long first pass, not only at its end
AUDIO_EXT = {".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".mp4", ".aac", ".wav",
             ".aif", ".aiff", ".wma", ".ape", ".wv", ".mpc", ".dsf", ".dff"}

# MusicNN front end, as in LMS's MusicNNEmbeddingExtractor.
SR, N_FFT, HOP, MELS, PATCH = 16000, 512, 256, 96, 187
MAX_PATCHES, MIN_PATCH_RMS = 20, 0.003
SPAN = (PATCH - 1) * HOP + N_FFT  # samples one patch reads
# Ranking, as in LMS's AudioSimilarityEngine.
DIMS, OVERSAMPLE, PATH_NEIGHBOURS, WINDOW = 60, 5, 32, 4
W_FIT, W_SMOOTH, W_RELEASE, W_ARTIST = 0.8, 0.2, 0.5, 0.5


def log(*a):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ analysis --

def mel_filterbank():
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    bins = np.floor(N_FFT * hz(np.linspace(mel(0), mel(SR / 2), MELS + 2)) / SR).astype(int)
    for i in range(1, len(bins)):
        bins[i] = max(bins[i], bins[i - 1] + 1)
    fb = np.zeros((MELS, N_FFT // 2 + 1), np.float32)
    for m in range(MELS):
        l, c, r = bins[m:m + 3]
        w = np.concatenate([(np.arange(l, c) - l) / (c - l), (r - np.arange(c, r)) / (r - c)])
        fb[m, l:r] = w / w.sum()
    return fb


FB = mel_filterbank()
WIN = (0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))).astype(np.float32)
POWER_SCALE = 1 / (np.sum(WIN ** 2) * N_FFT)
FRAMES = np.arange(PATCH)[:, None] * HOP + np.arange(N_FFT)


def patches(read, frame_count):
    """Yield SPAN-sample windows of the zero-left-padded signal, laid out like LMS:
    gap, patch, gap, patch, ... until the stream ends. frame_count 0 means unknown."""
    n = min(MAX_PATCHES, frame_count // PATCH)
    gap = (frame_count - n * PATCH) // (n + 1) if n else (0 if frame_count else 2 * PATCH)
    buf, base, start, eof = np.zeros(HOP, np.float32), 0, gap * HOP, False
    while True:
        while base + len(buf) < start + SPAN and not eof:
            chunk = read(1 << 18)
            eof = not chunk
            buf = np.concatenate([buf, np.frombuffer(chunk[:len(chunk) // 4 * 4], "<f4")])
            drop = min(max(start - base, 0), len(buf))
            buf, base = buf[drop:], base + drop
        seg = buf[start - base:start - base + SPAN]
        if len(seg) < SPAN:
            return
        yield seg
        start += (PATCH + gap) * HOP


def log_mel(seg):
    frames = seg[FRAMES]
    power = np.abs(np.fft.rfft(frames * WIN, axis=1)) ** 2 * POWER_SCALE
    rms = np.sqrt(np.mean(frames[:, :HOP] ** 2, axis=1)).mean()
    return np.log10(10000 * power @ FB.T + 1).astype(np.float32), rms


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-of", "json",
                          "-show_entries", "format=duration:format_tags:stream_tags", path],
                         capture_output=True, timeout=120, check=True).stdout
    info = json.loads(out)
    tags = {}
    for t in [info.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in info.get("streams", [])]:
        tags.update({k.lower(): v.strip() for k, v in t.items()})
    get = lambda *keys: next((tags[k] for k in keys if tags.get(k)), "")
    duration = float(info.get("format", {}).get("duration") or 0)
    # Navidrome's fallbacks for untagged files, so its matcher finds them.
    artists = [a.strip() for a in get("artist").split(";") if a.strip()] or ["[Unknown Artist]"]
    album_artist = get("album_artist", "albumartist") or artists[0]
    return duration, {
        "name": get("title") or os.path.splitext(os.path.basename(path))[0],
        "artists": [{"name": a} for a in artists],
        "album": get("album") or "[Unknown Album]",
        "mbid": get("musicbrainz_trackid", "musicbrainz track id"),
        "albumMbid": get("musicbrainz_albumid", "musicbrainz album id"),
        "durationMs": int(duration * 1000),
    }, album_artist


def audio_hash(path):
    """SHA-256 of the first audio stream's packets, copied without decoding: equal
    for a moved or retagged file, so its embedding can be reused."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy",
                          "-f", "hash", "-hash", "sha256", "-"], capture_output=True, timeout=300, check=True).stdout
    return out.decode().strip().split("=", 1)[1]


def embed(session, path, duration):
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path,
                             "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = int(duration * SR)
    frame_count = 1 + (total - N_FFT // 2) // HOP if total >= N_FFT // 2 else 0
    vecs = []
    try:
        for seg in patches(proc.stdout.read, frame_count):
            mel, rms = log_mel(seg)
            if rms >= MIN_PATCH_RMS:
                vecs.append(session.run(None, {"mel_patch": mel[None]})[0][0])
    finally:
        proc.kill()
        proc.wait()
    return np.mean(vecs, axis=0).astype(np.float32) if vecs else None


# --------------------------------------------------------------------- index --

class Index:
    def __init__(self, rows):
        self.paths = [r[0] for r in rows]
        self.by_size = {}  # filled once vectors exist; a query before that is "not analysed"
        meta = [json.loads(r[1]) for r in rows]
        self.songs = [m["song"] for m in meta]
        self.release = [(m["albumArtist"].lower(), m["song"]["album"].lower()) for m in meta]
        self.artists = [{a["name"].lower() for a in m["song"]["artists"]} for m in meta]
        # A row repeats a picked one if it shares its MBID or all its tags, as the
        # same file in a user's own library and a shared one does: Navidrome
        # resolves both answers to the same track.
        self.keys = [{json.dumps(m["song"], sort_keys=True)} | ({m["song"]["mbid"]} if m["song"]["mbid"] else set())
                     for m in meta]
        self.vec, self.threshold = np.zeros((0, DIMS), np.float32), np.inf
        if len(rows) < 2:
            return
        x = np.frombuffer(b"".join(r[2] for r in rows), np.float32).reshape(len(rows), -1)
        # Statistics come from distinct embeddings: copies of one recording say
        # nothing about the library's spread, and many of them would pull the
        # nearest-neighbour threshold to zero.
        uniq = np.unique(x, axis=0).astype(np.float64)
        if len(uniq) < 3:
            return
        mean = uniq.mean(axis=0)
        eigval, eigvec = np.linalg.eigh(np.cov(uniq - mean, rowvar=False))
        top = np.argsort(eigval)[::-1][:DIMS]
        # LMS whitens fully. With few distinct tracks that leaves every pair about
        # equally far apart, so the eigenvalues are shrunk towards their mean by
        # DIMS/distinct: a large library is whitened almost as in LMS, a small one
        # stays close to plain PCA.
        lam = np.maximum(eigval[top], 0)
        basis = eigvec[:, top] / np.sqrt(lam + lam.mean() * DIMS / len(uniq))

        def project(m):
            v = (m - mean) @ basis
            return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)

        self.vec = project(x)
        # Hard limit: mean + 2 sd of nearest-neighbour distances over a sample.
        u = project(uniq)
        sample = random.Random(42).sample(range(len(u)), min(500, len(u)))
        d = (1 - u[sample] @ u.T) / 2
        d[np.arange(len(sample)), sample] = np.inf
        nn = d.min(axis=1)
        self.threshold = nn.mean() + 2 * nn.std(ddof=1)
        for i, r in enumerate(rows):
            self.by_size.setdefault(r[3], []).append(i)

    def find(self, size, duration=0.0, title=""):
        return min(self.by_size[int(size)], key=lambda i: (
            abs(self.songs[i]["durationMs"] / 1000 - float(duration)), self.songs[i]["name"] != title))

    def dist(self, q):
        return (1 - q @ self.vec.T) / 2

    def penalty(self, c, chosen, smooth, fit):
        """LMS's soft constraints: distance to the seed and to the previous track
        (smooth, distances from it), plus repeats of a release or artist among the
        last WINDOW picks."""
        score = W_FIT * fit + W_SMOOTH * float(smooth[c])
        for i, s in enumerate(reversed(chosen[-WINDOW:]), 1):
            score += W_RELEASE * (self.release[s] == self.release[c]) / i
            score += W_ARTIST * bool(self.artists[s] & self.artists[c]) / i
        return score

    def taken(self, picks):
        return set().union(*(self.keys[c] for c in picks))

    def similar(self, seed, count):
        d = self.dist(self.vec[seed])
        d[seed] = np.inf
        ranked = [int(c) for c in np.argsort(d)[:count * OVERSAMPLE] if d[c] <= self.threshold]
        chosen, out, taken = [seed], [], self.taken([seed])
        while len(out) < count:
            ranked = [c for c in ranked if taken.isdisjoint(self.keys[c])]
            if not ranked:
                break
            smooth = self.dist(self.vec[chosen[-1]])
            best = min(ranked, key=lambda c: self.penalty(c, chosen, smooth, float(d[c])))
            chosen.append(best)
            taken |= self.keys[best]
            out.append((best, 1 - float(d[best])))
        return out

    def route(self, start, end, count):
        chosen = [start]
        for i in range(max(count - 2, 0)):
            t = (i + 1) / (count - 1)
            q = self.vec[start] + (self.vec[end] - self.vec[start]) * t
            d = self.dist(q / max(np.linalg.norm(q), 1e-12))
            d[end] = np.inf
            near = [int(c) for c in np.argsort(d)[:PATH_NEIGHBOURS]]
            fit, smooth = self.dist(self.vec[near[0]]), self.dist(self.vec[chosen[-1]])
            taken = self.taken(chosen + [end])
            ok = [c for c in near if taken.isdisjoint(self.keys[c])]
            if ok:
                chosen.append(min(ok, key=lambda c: self.penalty(c, chosen, smooth, float(fit[c]))))
        if count > 1:
            chosen.append(end)
        d = self.dist(self.vec[start])
        return [(c, 1 - float(d[c])) for c in chosen]


INDEX = Index([])


def load_index(db):
    global INDEX
    rows = db.execute("SELECT path, meta, emb, size FROM track WHERE emb IS NOT NULL").fetchall()
    INDEX = Index(rows)
    log(f"index: {len(rows)} tracks, threshold {INDEX.threshold:.4f}")


# ---------------------------------------------------------------------- scan --

def raise_error(e):
    raise e


def scan(db, session):
    known = {p: (m, s, a) for p, m, s, a in db.execute("SELECT path, mtime, size, audio FROM track")}
    by_audio = dict(db.execute("SELECT audio, emb FROM track WHERE audio != '' AND emb IS NOT NULL"))
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
            old = known.get(path)
            if old and old[:2] == (int(st.st_mtime), st.st_size):
                # Rows stored before the audio column get their hash on the first
                # pass, which reads each such file once without decoding it. ""
                # marks a file ffmpeg could not hash, so it is not retried.
                if old[2] is None:
                    try:
                        audio = audio_hash(path)
                    except Exception as e:
                        log(f"skip {path}: {e}")
                        audio = ""
                    db.execute("UPDATE track SET audio = ? WHERE path = ?", (audio, path))
                    db.commit()
                    emb = db.execute("SELECT emb FROM track WHERE path = ?", (path,)).fetchone()[0]
                    if audio and emb is not None:
                        by_audio[audio] = emb
                continue
            emb, meta, audio = None, None, None
            try:
                duration, song, album_artist = probe(path)
                meta = json.dumps({"song": song, "albumArtist": album_artist})
                audio = audio_hash(path)
                emb = by_audio.get(audio)
                if emb is None:
                    vec = embed(session, path, duration)
                    analysed += 1
                    if vec is not None:
                        emb = by_audio[audio] = vec.tobytes()
            except Exception as e:  # one bad file must not stop the pass
                log(f"skip {path}: {e}")
            if emb is None:
                log(f"no embedding: {path}")
            # A failed file is stored too, so it is retried only once it changes.
            db.execute("INSERT OR REPLACE INTO track (path, mtime, size, meta, emb, audio) VALUES (?, ?, ?, ?, ?, ?)",
                       (path, int(st.st_mtime), st.st_size, meta, emb, audio or ""))
            db.commit()
            changed += 1
            if changed % REINDEX_EVERY == 0:
                load_index(db)
    gone = set(known) - seen
    if seen:
        db.executemany("DELETE FROM track WHERE path = ?", [(p,) for p in gone])
        db.commit()
    log(f"scan: {len(seen)} files, {changed} new or changed, {analysed} analysed, "
        f"{len(gone) if seen else 0} removed")
    return changed or (gone and seen)


def scanner():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
    db = sqlite3.connect(DB)
    load_index(db)
    while True:
        try:
            if scan(db, session):
                load_index(db)
        except Exception as e:
            log(f"scan aborted: {e}")
        time.sleep(SCAN_INTERVAL)


# ---------------------------------------------------------------------- http --

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        idx = INDEX
        try:
            count = max(1, min(int(q.get("count", 10)), 500))
            if url.path == "/health":
                return self.reply(200, {"tracks": len(idx.paths)})
            seed = lambda p="": idx.find(q[p + "size"], q.get(p + "duration", 0), q.get(p + "title", ""))
            if url.path == "/similar":
                res = idx.similar(seed(), count)
            elif url.path == "/path":
                res = idx.route(seed("start_"), seed("end_"), count)
            else:
                return self.reply(404, {"error": "not found"})
        except KeyError as e:
            return self.reply(404, {"error": f"not analysed: {e}"})
        except ValueError as e:
            return self.reply(400, {"error": str(e)})
        self.reply(200, {"matches": [{"song": idx.songs[c], "similarity": s} for c, s in res]})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def init_db(db):
    db.execute("CREATE TABLE IF NOT EXISTS track "
               "(path TEXT PRIMARY KEY, mtime INTEGER, size INTEGER, meta TEXT, emb BLOB, audio TEXT)")
    if "audio" not in [c[1] for c in db.execute("PRAGMA table_info(track)")]:
        db.execute("ALTER TABLE track ADD COLUMN audio TEXT")


def main():
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    init_db(db)
    db.close()
    threading.Thread(target=scanner, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


# ---------------------------------------------------------------------- test --

def test():
    # Patch layout: 100 s known -> 20 patches with equal gaps; unknown -> gap of 2 patches.
    sig = np.random.default_rng(0).standard_normal(SR * 100).astype(np.float32)
    frames = 1 + (len(sig) - N_FFT // 2) // HOP
    segs = list(patches(io.BytesIO(sig.tobytes()).read, frames))
    assert len(segs) == 20, len(segs)
    gap = (frames - 20 * PATCH) // 21
    first = gap * HOP - HOP  # padded index -> signal index
    assert np.array_equal(segs[0], sig[first:first + SPAN])
    unknown = list(patches(io.BytesIO(sig.tobytes()).read, 0))
    assert np.array_equal(unknown[1], sig[(5 * PATCH - 1) * HOP:(5 * PATCH - 1) * HOP + SPAN])
    mel, rms = log_mel(segs[0])
    assert mel.shape == (PATCH, MELS) and rms > MIN_PATCH_RMS
    assert abs(FB.sum(axis=1) - 1).max() < 1e-5

    # Ranking: a same-artist neighbour loses to a slightly farther other artist.
    def row(path, artist, vec, size=None, seconds=0, name=None):
        song = {"name": name or path, "artists": [{"name": artist}], "album": name or path, "mbid": "",
                "durationMs": seconds * 1000}
        return (path, json.dumps({"song": song, "albumArtist": artist}), np.asarray(vec, np.float32).tobytes(),
                size if size is not None else hash(path))
    rng = np.random.default_rng(1)
    base = rng.standard_normal(200)
    rows = [row("seed", "A", base), row("same", "A", base + 0.01 * rng.standard_normal(200)),
            row("other", "B", base + 0.02 * rng.standard_normal(200))]
    rows += [row(f"far{i}", f"F{i}", rng.standard_normal(200)) for i in range(80)]
    # Size lookup: three files of one size, told apart by duration, then title.
    rows += [row(name, "C", rng.standard_normal(200), 1000, s) for name, s in (("x", 100), ("y", 200), ("z", 200))]
    # The seed's file again in another library: never offered as similar to itself.
    rows += [row("seed-copy", "A", base, name="seed")]
    assert Index(rows[:2]).by_size == {}
    idx = Index(rows)
    assert idx.paths[idx.find("1000", "199.4", "z")] == "z"
    assert idx.paths[idx.find(1000, 120)] == "x"
    assert idx.paths[idx.find(1000, 200, "y")] == "y"
    assert idx.paths[idx.find(hash("seed"))] == "seed"
    try:
        idx.find(1)
        raise AssertionError("unknown size found")
    except KeyError:
        pass
    picks = [idx.paths[c] for c, _ in idx.similar(0, 2)]
    assert picks == ["other", "same"], picks
    route = [idx.paths[c] for c, _ in idx.route(0, 3, 4)]
    assert route[0] == "seed" and route[-1] == "far0" and len(set(route)) == len(route), route

    # Scan: a moved file and a retagged one keep their embeddings, and rows from
    # before the audio column get a hash without being analysed again.
    global MUSIC
    class Session:
        runs = 0
        def run(self, _, feeds):
            Session.runs += 1
            return [np.full((1, 200), feeds["mel_patch"].mean(), np.float32)]
    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)
    music, MUSIC = MUSIC, tempfile.mkdtemp()
    try:
        ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=5", "-metadata", "title=A", f"{MUSIC}/a.mp3")
        ffmpeg("-f", "lavfi", "-i", "sine=f=660:d=5", "-metadata", "title=B", f"{MUSIC}/b.mp3")
        db = sqlite3.connect(":memory:")
        init_db(db)
        scan(db, Session())
        before = dict(db.execute("SELECT audio, emb FROM track"))
        assert Session.runs == 2 and len(before) == 2 and None not in before.values(), Session.runs
        os.makedirs(f"{MUSIC}/moved")
        os.rename(f"{MUSIC}/a.mp3", f"{MUSIC}/moved/a.mp3")
        ffmpeg("-i", f"{MUSIC}/b.mp3", "-c", "copy", "-metadata", "title=B2", f"{MUSIC}/b2.mp3")
        os.replace(f"{MUSIC}/b2.mp3", f"{MUSIC}/b.mp3")
        os.utime(f"{MUSIC}/b.mp3", (0, 0))
        scan(db, Session())
        rows = {os.path.relpath(p, MUSIC): (json.loads(m)["song"]["name"], a, e)
                for p, m, a, e in db.execute("SELECT path, meta, audio, emb FROM track")}
        assert Session.runs == 2 and sorted(rows) == ["b.mp3", "moved/a.mp3"], (Session.runs, sorted(rows))
        assert rows["b.mp3"][0] == "B2" and all(before[a] == e for _, a, e in rows.values())
        db.execute("UPDATE track SET audio = NULL")
        assert not scan(db, Session()) and Session.runs == 2
        assert dict(db.execute("SELECT audio, emb FROM track")) == before
    finally:
        shutil.rmtree(MUSIC)
        MUSIC = music
    print("ok")


if __name__ == "__main__":
    test() if sys.argv[1:] == ["test"] else main()
