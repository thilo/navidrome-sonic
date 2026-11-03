#!/usr/bin/env python3
"""Self-check for the sidecar (sonic.py, analysis.py, index.py) and sonic-decode:
`python3 test_sonic.py`."""

import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile

import sonic  # first: it sets the OpenBLAS thread count before numpy loads
from analysis import DECODE, HOP, MELS, MIN_PATCH_RMS, MOODS, N_FFT, PATCH, SPAN, SR, audio_hash, frames, layout, log_mel, np, probe
from index import Index
from sonic import init_db, scan


def patches(read, frame_count):
    """Yield SPAN-sample windows of the zero-left-padded signal at layout(frame_count)
    until the stream ends: what sonic-decode writes, as the reference."""
    first, step = layout(frame_count)
    buf, base, start, eof = np.zeros(HOP, np.float32), 0, first + HOP, False
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
        start += step


def test():
    # Patch layout: 100 s known -> 20 patches with equal gaps; unknown -> gap of 2 patches.
    sig = np.random.default_rng(0).standard_normal(SR * 100).astype(np.float32)
    frame_count = frames(len(sig) / SR)
    segs = list(patches(io.BytesIO(sig.tobytes()).read, frame_count))
    assert len(segs) == 20, len(segs)
    gap = (frame_count - 20 * PATCH) // 21
    first = gap * HOP - HOP  # padded index -> signal index
    assert np.array_equal(segs[0], sig[first:first + SPAN])
    unknown = list(patches(io.BytesIO(sig.tobytes()).read, 0))
    assert np.array_equal(unknown[1], sig[(5 * PATCH - 1) * HOP:(5 * PATCH - 1) * HOP + SPAN])
    mel, rms = log_mel(segs[0])
    assert mel.shape == (PATCH, MELS) and rms > MIN_PATCH_RMS
    # Front end against Essentia: TensorflowInputMusiCNN() on the first 512 samples of
    # 0.5 sin(1 kHz) + 0.1 sin(150 Hz), bands 3, 10, 30, 31, 32, 60, 95.
    t = np.arange(SPAN) / SR
    mel, _ = log_mel((0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 150 * t)).astype(np.float32))
    bands = mel[0][[3, 10, 30, 31, 32, 60, 95]]
    assert np.allclose(bands, [4.338388, 0.014412, 5.446386, 6.057385, 5.667006, 2.07e-07, 0.0], atol=1e-4), bands

    # Ranking: a same-artist neighbour loses to a slightly farther other artist.
    def row(path, artist, vec, size=None, seconds=0, name=None, happy=0.5):
        song = {"name": name or path, "artists": [{"name": artist}], "album": name or path, "mbid": "",
                "durationMs": seconds * 1000}
        return (path, json.dumps({"song": song, "albumArtist": artist}), np.asarray(vec, np.float32).tobytes(),
                size if size is not None else hash(path), json.dumps({m: happy if m == "happy" else 0.5 for m in MOODS}))
    dbs = []
    def index(rows):  # the embedding bytes stand in for the audio hash
        dbs.append(tempfile.mkdtemp())
        db = sqlite3.connect(f"{dbs[-1]}/sonic.db")
        init_db(db)
        db.executemany("INSERT INTO track (path, meta, emb, audio, size, mood) VALUES (?, ?, ?, ?, ?, ?)",
                       [(p, m, e, e.hex(), s, md) for p, m, e, s, md in rows])
        db.commit()
        db.close()
        return Index(f"{dbs[-1]}/sonic.db")
    names = lambda res: [song["name"] for song, _ in res]
    rng = np.random.default_rng(1)
    base = rng.standard_normal(1280)
    rows = [row("seed", "A", base), row("same", "A", base + 0.01 * rng.standard_normal(1280)),
            row("other", "B", base + 0.02 * rng.standard_normal(1280))]
    rows += [row(f"far{i}", f"F{i}", rng.standard_normal(1280)) for i in range(80)]
    # Size lookup: three files of one size, told apart by duration, then title.
    rows += [row(name, "C", rng.standard_normal(1280), 1000, s) for name, s in (("x", 100), ("y", 200), ("z", 200))]
    # The seed's file again in another library: never offered as similar to itself.
    rows += [row("seed-copy", "A", base, name="seed")]
    assert index(rows[:2]).paths == []
    idx = index(rows)
    assert idx.paths[idx.find("1000", "199.4", "z")] == "z"
    assert idx.paths[idx.find(1000, 120)] == "x"
    assert idx.paths[idx.find(1000, 200, "y")] == "y"
    assert idx.paths[idx.find(hash("seed"))] == "seed"
    try:
        idx.find(1)
        raise AssertionError("unknown size found")
    except KeyError:
        pass
    picks = names(idx.similar(0, 2))
    assert picks == ["other", "same"], picks
    route = names(idx.route(0, 3, 4))
    assert route[0] == "seed" and route[-1] == "far0" and len(set(route)) == len(route), route
    # A track removed after the build is no longer offered.
    with sqlite3.connect(idx.db_path) as db:
        db.execute("DELETE FROM track WHERE path = 'other'")
    assert "other" not in names(idx.similar(0, 10)) + names(idx.route(0, 3, 4)) and idx.similar(0, 10)

    # Moods: highest score first, a recording once even when it is in two libraries.
    moods = index([row("sad", "A", base, happy=0.1), row("glad", "B", base, happy=0.9),
                   row("glad-copy", "B", base, name="glad", happy=0.9), row("ok", "C", base, happy=0.6)])
    assert names(moods.mood("happy", 10)) == ["glad", "ok", "sad"]
    assert [s for _, s in moods.mood("happy", 2)] == [0.9, 0.6]
    for d in dbs:
        shutil.rmtree(d)

    # Scan: a moved file and a retagged one keep their embeddings and mood scores;
    # a database from another analysis version starts over.
    class Session:
        def run(self, _, feeds):
            mels = feeds["serving_default_melspectrogram:0"]
            return [np.full((len(mels), 1280), mels.mean(), np.float32)]
    class Head:  # each head runs once per analysed track
        runs = 0
        def run(self, _, feeds):
            Head.runs += 1
            p = float(feeds["embeddings"].mean()) % 1
            return [np.array([[p, 1 - p]], np.float32)]
    models = {"effnet": Session(), **{m: Head() for m in MOODS}}
    analysed = lambda: Head.runs // len(MOODS)
    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)
    MUSIC = sonic.MUSIC = tempfile.mkdtemp()
    try:
        ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=5", "-metadata", "title=A", f"{MUSIC}/a.mp3")
        ffmpeg("-f", "lavfi", "-i", "sine=f=660:d=5", "-metadata", "title=B", f"{MUSIC}/b.mp3")
        db = sqlite3.connect(":memory:")
        init_db(db)
        scan(db, models)
        before = {a: (e, m) for a, e, m in db.execute("SELECT audio, emb, mood FROM track")}
        assert analysed() == 2 and len(before) == 2 and all(None not in v for v in before.values()), analysed()
        assert set(json.loads(next(iter(before.values()))[1])) == set(MOODS)
        os.makedirs(f"{MUSIC}/moved")
        os.rename(f"{MUSIC}/a.mp3", f"{MUSIC}/moved/a.mp3")
        ffmpeg("-i", f"{MUSIC}/b.mp3", "-c", "copy", "-metadata", "title=B2", f"{MUSIC}/b2.mp3")
        os.replace(f"{MUSIC}/b2.mp3", f"{MUSIC}/b.mp3")
        os.utime(f"{MUSIC}/b.mp3", (0, 0))
        scan(db, models)
        rows = {os.path.relpath(p, MUSIC): (json.loads(m)["song"]["name"], a, (e, md))
                for p, m, a, e, md in db.execute("SELECT path, meta, audio, emb, mood FROM track")}
        assert analysed() == 2 and sorted(rows) == ["b.mp3", "moved/a.mp3"], (analysed(), sorted(rows))
        assert rows["b.mp3"][0] == "B2" and all(before[a] == v for _, a, v in rows.values())
        assert not scan(db, models) and analysed() == 2
        db.execute("PRAGMA user_version = 1")
        init_db(db)
        assert db.execute("SELECT count(*) FROM track").fetchone()[0] == 0
        scan(db, models)
        assert analysed() == 4
    finally:
        shutil.rmtree(MUSIC)

    # sonic-decode against the ffmpeg CLI it replaces: the same packet hash, and the same
    # windows as patches() over the CLI's decode, which is what embed() read before.
    def cli_hash(path):
        out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy",
                              "-f", "hash", "-hash", "sha256", "-"], capture_output=True, check=True).stdout
        return out.decode().strip().split("=", 1)[1]
    def cli_windows(path, frame_count):
        pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path, "-map", "0:a:0",
                              "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return [seg.tobytes() for seg in patches(io.BytesIO(pcm).read, frame_count)]
    def windows(path, frame_count):
        first, step = layout(frame_count)
        out = subprocess.run([DECODE, "windows", path, str(first), str(step), str(SPAN)],
                             capture_output=True, check=True).stdout
        return [out[i:i + SPAN * 4] for i in range(0, len(out), SPAN * 4)]
    tmp = tempfile.mkdtemp()
    try:
        # Layout, through a 16 kHz mono float WAV, which decodes to the signal itself: gaps,
        # unknown duration, and gap 0 (first window at -HOP, overlapping windows).
        short = sig[:(20 * PATCH - 1) * HOP + N_FFT // 2]
        assert layout(20 * PATCH)[0] == -HOP and layout(20 * PATCH)[1] < SPAN
        for s, count in ((sig, frame_count), (sig, 0), (short, 20 * PATCH)):
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                            "-i", "-", "-c:a", "pcm_f32le", f"{tmp}/layout.wav"], input=s.tobytes(), check=True)
            want = [seg.tobytes() for seg in patches(io.BytesIO(s.tobytes()).read, count)]
            assert want and windows(f"{tmp}/layout.wav", count) == want, count

        # Codecs and containers, windowed and decoded in full. 120 s put the windows far enough
        # apart for separate segments, also with FLAC's and WAV's 4096-sample packets.
        ffmpeg("-f", "lavfi", "-i", "aevalsrc=0.4*sin(2*PI*220*t)+0.1*(2*random(0)-1)|"
               "0.4*sin(2*PI*331*t)+0.1*(2*random(1)-1):s=44100:d=120", f"{tmp}/src.wav")
        files = {"a.mp3": ["-c:a", "libmp3lame"], "b.mp3": ["-ar", "48000", "-c:a", "libmp3lame"],
                 "c.flac": [], "d.wav": ["-c:a", "pcm_s16le"], "e.ogg": ["-c:a", "libvorbis"],
                 "f.m4a": ["-c:a", "aac"], "g.opus": ["-c:a", "libopus"], "h.wma": ["-c:a", "wmav2"]}
        for name, args in files.items():
            ffmpeg("-i", f"{tmp}/src.wav", *args, f"{tmp}/{name}")
        # Restarts in full mode and filter graph rebuilds: a chained Ogg and a mp3 whose sample
        # rate and channels change; a flac frame without its sync code, which the decoder
        # drops, so the later samples move up; a mp3 frame without its header, which never
        # becomes a packet.
        def cat(name, *parts):
            with open(f"{tmp}/{name}", "wb") as f:
                for p in parts:
                    f.write(open(f"{tmp}/{p}", "rb").read())
            files[name] = None
        def damage(name, source, at, new):
            """Overwrites the start of the packet nearest `at` seconds."""
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                  "packet=pts_time,pos", "-of", "json", f"{tmp}/{source}"],
                                 capture_output=True, check=True).stdout
            i = int(min(json.loads(out)["packets"], key=lambda p: abs(float(p["pts_time"]) - at))["pos"])
            data = bytearray(open(f"{tmp}/{source}", "rb").read())
            data[i:i + len(new)] = new
            open(f"{tmp}/{name}", "wb").write(data)
            files[name] = None
        ffmpeg("-i", f"{tmp}/src.wav", "-t", "60", "-c:a", "libvorbis", f"{tmp}/x1.ogg")
        ffmpeg("-i", f"{tmp}/src.wav", "-ss", "60", "-ar", "22050", "-ac", "1", "-c:a", "libvorbis", f"{tmp}/x2.ogg")
        cat("chained.ogg", "x1.ogg", "x2.ogg")
        ffmpeg("-i", f"{tmp}/src.wav", "-t", "60", "-c:a", "libmp3lame", f"{tmp}/x1.mp3")
        ffmpeg("-i", f"{tmp}/src.wav", "-ss", "60", "-ar", "48000", "-ac", "1", "-c:a", "libmp3lame",
               "-write_xing", "0", "-id3v2_version", "0", f"{tmp}/x2.mp3")
        cat("rates.mp3", "x1.mp3", "x2.mp3")
        # Between two segments, which only the packet check sees: after window 9 ends and
        # before window 10's preroll (about a second) starts.
        first, step = layout(frames(probe(f"{tmp}/c.flac")[0]))
        at = (first + 9 * step + SPAN) / SR + 0.8
        damage("nosync.flac", "c.flac", at, bytes(2))
        damage("noheader.mp3", "a.mp3", 60, bytes(4))
        for name in files:
            path = f"{tmp}/{name}"
            count = frames(probe(path)[0])
            want = cli_windows(path, count)
            assert audio_hash(path) == cli_hash(path), name
            assert want and windows(path, count) == want, name
    finally:
        shutil.rmtree(tmp)
    print("ok")


if __name__ == "__main__":
    test()
