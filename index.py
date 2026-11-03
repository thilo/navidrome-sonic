"""The search index: whitened 60-D vectors of every analysed track, and the queries.

The ranking follows LMS (src/libs/services/recommendation/impl/audio-similarity):
queries run on a whitened PCA projection of the embeddings with cosine distance,
under LMS's soft constraints against repeating a release or an artist. Only the
vectors live in memory; tags and mood scores are read from the database per
query, so the index stays small next to the one being built to replace it.
"""

import json
import random
import sqlite3
from contextlib import closing

import numpy as np

# Ranking, as in LMS's AudioSimilarityEngine.
DIMS, OVERSAMPLE, PATH_NEIGHBOURS, WINDOW = 60, 5, 32, 4
W_FIT, W_SMOOTH, W_RELEASE, W_ARTIST = 0.8, 0.2, 0.5, 0.5


def parse(meta):
    """What a query needs from a track's stored tags."""
    m = json.loads(meta)
    song = m["song"]
    return {"song": song, "release": (m["albumArtist"].lower(), song["album"].lower()),
            "artists": {a["name"].lower() for a in song["artists"]},
            # A row repeats a picked one if it shares its MBID or all its tags, as the
            # same file in a user's own library and a shared one does: Navidrome
            # resolves both answers to the same track.
            "keys": {json.dumps(song, sort_keys=True)} | ({song["mbid"]} if song["mbid"] else set())}


class Rows:
    """One query's view of the track table: a track's parsed tags by index position,
    read once. None for a track removed or emptied since the index was built."""

    def __init__(self, idx):
        self.idx, self.cache = idx, {}
        self.db = sqlite3.connect(idx.db_path)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def __getitem__(self, c):
        if c not in self.cache:
            r = self.db.execute("SELECT meta FROM track WHERE path = ? AND emb IS NOT NULL", (self.idx.paths[c],)).fetchone()
            self.cache[c] = r and parse(r[0])
        return self.cache[c]


class Index:
    def __init__(self, db_path):
        self.db_path = db_path
        self.paths, self.pos = [], {}  # filled once vectors exist; a query before that is "not analysed"
        self.vec, self.threshold = np.zeros((0, DIMS), np.float32), np.inf
        with closing(sqlite3.connect(db_path)) as db:
            db.execute("BEGIN")  # one snapshot for both passes below
            self.build(db)

    def build(self, db):
        rows = db.execute("SELECT path, audio FROM track WHERE emb IS NOT NULL ORDER BY rowid").fetchall()
        # Statistics come from distinct embeddings: copies of one recording say
        # nothing about the library's spread, and many of them would pull the
        # nearest-neighbour threshold to zero.
        keep = np.zeros(len(rows), bool)
        keep[list({a: i for i, (_, a) in enumerate(rows)}.values())] = True
        n = int(keep.sum())
        if n < 3:
            return

        # The N x D matrix is never built: at 1280 dimensions its copies would make
        # up most of the build's memory. One pass over the embeddings sums up the
        # covariance, in float64; a second projects every row.
        def embeddings():
            cur = db.execute("SELECT emb FROM track WHERE emb IS NOT NULL ORDER BY rowid")
            while batch := cur.fetchmany(256):
                yield np.frombuffer(b"".join(e for e, in batch), np.float32).reshape(len(batch), -1)
        s = cov = None
        i = 0
        for e in embeddings():
            b = e[keep[i:i + len(e)]].astype(np.float64)
            i += len(e)
            if cov is None:
                s, cov = b.sum(axis=0), b.T @ b
            else:
                s += b.sum(axis=0)
                cov += b.T @ b
        mean = s / n
        cov -= n * np.outer(mean, mean)
        cov /= n - 1
        eigval, eigvec = np.linalg.eigh(cov)
        del cov
        top = np.argsort(eigval)[::-1][:DIMS]
        # LMS whitens fully. With few distinct tracks that leaves every pair about
        # equally far apart, so the eigenvalues are shrunk towards their mean by
        # DIMS/distinct: a large library is whitened almost as in LMS, a small one
        # stays close to plain PCA.
        lam = np.maximum(eigval[top], 0)
        basis = (eigvec[:, top] / np.sqrt(lam + lam.mean() * DIMS / n)).astype(np.float32)
        mean = mean.astype(np.float32)
        del eigvec
        v = np.concatenate([(e - mean) @ basis for e in embeddings()])
        self.vec = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        # Hard limit: mean + 2 sd of nearest-neighbour distances over a sample.
        u = self.vec[keep]
        sample = random.Random(42).sample(range(len(u)), min(500, len(u)))
        d = u[sample] @ u.T
        d *= -0.5
        d += 0.5
        d[np.arange(len(sample)), sample] = np.inf
        nn = d.min(axis=1)
        self.threshold = nn.mean() + 2 * nn.std(ddof=1)
        self.paths = [p for p, _ in rows]
        self.pos = {p: i for i, p in enumerate(self.paths)}

    def find(self, size, duration=0.0, title=""):
        """The track of that file size, told apart from others of the same size by
        the closest duration (seconds), then an exact title."""
        with Rows(self) as rows:
            found = [self.pos[p] for p, in rows.db.execute("SELECT path FROM track WHERE size = ?", (int(size),))
                     if p in self.pos and rows[self.pos[p]]]
            if not found:
                raise KeyError(int(size))
            return min(found, key=lambda c: (abs(rows[c]["song"]["durationMs"] / 1000 - float(duration)),
                                             rows[c]["song"]["name"] != title))

    def dist(self, q):
        return (1 - q @ self.vec.T) / 2

    def penalty(self, rows, c, chosen, smooth, fit):
        """LMS's soft constraints: distance to the seed and to the previous track
        (smooth, distances from it), plus repeats of a release or artist among the
        last WINDOW picks."""
        score = W_FIT * fit + W_SMOOTH * float(smooth[c])
        for i, s in enumerate(reversed(chosen[-WINDOW:]), 1):
            score += W_RELEASE * (rows[s]["release"] == rows[c]["release"]) / i
            score += W_ARTIST * bool(rows[s]["artists"] & rows[c]["artists"]) / i
        return score

    def similar(self, seed, count):
        with Rows(self) as rows:
            if not rows[seed]:
                raise KeyError(self.paths[seed])
            d = self.dist(self.vec[seed])
            d[seed] = np.inf
            ranked = [int(c) for c in np.argsort(d)[:count * OVERSAMPLE] if d[c] <= self.threshold and rows[int(c)]]
            chosen, out, taken = [seed], [], set(rows[seed]["keys"])
            while len(out) < count:
                ranked = [c for c in ranked if taken.isdisjoint(rows[c]["keys"])]
                if not ranked:
                    break
                smooth = self.dist(self.vec[chosen[-1]])
                best = min(ranked, key=lambda c: self.penalty(rows, c, chosen, smooth, float(d[c])))
                chosen.append(best)
                taken |= rows[best]["keys"]
                out.append((rows[best]["song"], 1 - float(d[best])))
            return out

    def mood(self, name, count):
        """The tracks with the highest score for a mood, each recording once."""
        out, taken = [], set()
        with Rows(self) as rows:
            for meta, score in rows.db.execute(
                    "SELECT meta, json_extract(mood, ?) AS score FROM track WHERE emb IS NOT NULL "
                    "ORDER BY score DESC, rowid", (f"$.{name}",)):
                if len(out) == count:
                    break
                r = parse(meta)
                if taken.isdisjoint(r["keys"]):
                    out.append((r["song"], score))
                    taken |= r["keys"]
        return out

    def route(self, start, end, count):
        with Rows(self) as rows:
            for c in (start, end):
                if not rows[c]:
                    raise KeyError(self.paths[c])
            chosen = [start]
            for i in range(max(count - 2, 0)):
                t = (i + 1) / (count - 1)
                q = self.vec[start] + (self.vec[end] - self.vec[start]) * t
                d = self.dist(q / max(np.linalg.norm(q), 1e-12))
                d[end] = np.inf
                near = [int(c) for c in np.argsort(d)[:PATH_NEIGHBOURS]]
                fit, smooth = self.dist(self.vec[near[0]]), self.dist(self.vec[chosen[-1]])
                taken = set().union(*(rows[c]["keys"] for c in chosen + [end]))
                ok = [c for c in near if rows[c] and taken.isdisjoint(rows[c]["keys"])]
                if ok:
                    chosen.append(min(ok, key=lambda c: self.penalty(rows, c, chosen, smooth, float(fit[c]))))
            if count > 1:
                chosen.append(end)
            d = self.dist(self.vec[start])
            return [(rows[c]["song"], 1 - float(d[c])) for c in chosen]
