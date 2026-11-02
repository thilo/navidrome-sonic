#!/usr/bin/env python3
"""Self-check for sonic.py and sonic-decode: `python3 test_sonic.py`."""

import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile

import sonic
from sonic import DECODE, FB, HOP, MELS, MIN_PATCH_RMS, N_FFT, PATCH, SPAN, SR, Index, audio_hash, frames, init_db, layout, log_mel, np, patches, probe, scan


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
    class Session:
        runs = 0
        def run(self, _, feeds):
            Session.runs += 1
            return [np.full((1, 200), feeds["mel_patch"].mean(), np.float32)]
    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)
    MUSIC = sonic.MUSIC = tempfile.mkdtemp()
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
