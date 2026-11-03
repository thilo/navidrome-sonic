"""Audio analysis: one Discogs-EffNet embedding and six mood scores per file.

The front end follows Essentia's TensorflowInputMusiCNN, which EffNet and its mood
heads were trained on: 16 kHz mono, log-mel patches of 128 frames. sonic-decode
(decode.c) supplies the samples of up to 20 evenly spaced patches per track; the
embedding is the mean of their 1280-D outputs, and the heads score that mean.
"""

import json
import os
import subprocess

import numpy as np

MODELS = "/opt/sonic/models"
DECODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sonic-decode")  # built from decode.c
# Mood -> Essentia head and the index of its positive class (from the head's metadata).
MOODS = {"happy": ("mood_happy", 0), "sad": ("mood_sad", 1), "relaxed": ("mood_relaxed", 1),
         "aggressive": ("mood_aggressive", 0), "party": ("mood_party", 1), "danceable": ("danceability", 0)}

# EffNet front end, as in Essentia's TensorflowInputMusiCNN and the model's patch size.
SR, N_FFT, HOP, MELS, PATCH = 16000, 512, 256, 96, 128
SPAN = (PATCH - 1) * HOP + N_FFT  # samples one patch reads
# Patch selection, as in LMS: up to 20 evenly spaced patches, silent ones skipped.
# Essentia runs every patch, 62 frames apart.
MAX_PATCHES, MIN_PATCH_RMS = 20, 0.003


def mel_filterbank():
    """Essentia's MelBands as TensorflowInputMusiCNN sets them: Slaney mel scale,
    triangles over the bin frequencies, each divided by its area in Hz (unit_tri)."""
    slaney = lambda f: np.where(f < 1000, f * 3 / 200, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27))
    hz = lambda m: np.where(m < 15, m * 200 / 3, 1000 * np.exp((m - 15) * np.log(6.4) / 27))
    edges = hz(slaney(np.float64(SR / 2)) * np.arange(MELS + 2) / (MELS + 1))
    f = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    lo, peak, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    tri = np.maximum(0, np.minimum((f - lo) / (peak - lo), (hi - f) / (hi - peak)))
    return (tri / ((hi - lo) / 2)).astype(np.float32)


FB = mel_filterbank()
WIN = (0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))).astype(np.float32)
FRAMES = np.arange(PATCH)[:, None] * HOP + np.arange(N_FFT)


def frames(duration):
    total = int(duration * SR)
    return 1 + (total - N_FFT // 2) // HOP if total >= N_FFT // 2 else 0


def layout(frame_count):
    """Signal sample of the first window and the distance between windows, laid out like
    LMS: gap, patch, gap, patch, ... frame_count 0 means unknown. The first window may
    start at -HOP, in the zero padding."""
    n = min(MAX_PATCHES, frame_count // PATCH)
    gap = (frame_count - n * PATCH) // (n + 1) if n else (0 if frame_count else 2 * PATCH)
    return gap * HOP - HOP, (PATCH + gap) * HOP


def log_mel(seg):
    frames = seg[FRAMES]
    power = np.abs(np.fft.rfft(frames * WIN, axis=1)) ** 2
    rms = np.sqrt(np.mean(frames[:, :HOP] ** 2, axis=1)).mean()
    return np.log10(10000 * power @ FB.T + 1).astype(np.float32), rms


def probe(path):
    """Duration in seconds, the SongRef Navidrome matches by, and the album artist."""
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
    return subprocess.run([DECODE, "hash", path], capture_output=True, timeout=300, check=True).stdout.decode().strip()


def load_models():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    # Otherwise onnxruntime keeps its largest activations allocated between tracks.
    opts.enable_cpu_mem_arena = opts.enable_mem_pattern = False
    load = lambda name: ort.InferenceSession(f"{MODELS}/{name}.onnx", opts, providers=["CPUExecutionProvider"])
    return {"effnet": load("discogs-effnet"), **{mood: load(f"{head}-discogs-effnet-1") for mood, (head, _) in MOODS.items()}}


def embed(models, path, duration):
    """The track's mean embedding and its mood scores, or (None, None) if it is silent."""
    first, step = layout(frames(duration))
    proc = subprocess.Popen([DECODE, "windows", path, str(first), str(step), str(SPAN)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mels = []
    try:
        while len(seg := proc.stdout.read(SPAN * 4)) == SPAN * 4:
            mel, rms = log_mel(np.frombuffer(seg, "<f4"))
            if rms >= MIN_PATCH_RMS:
                mels.append(mel)
        # Its stderr is at most one line, so reading it after stdout cannot block.
        if proc.wait():
            raise RuntimeError(f"sonic-decode: {proc.stderr.read().decode().strip()}")
    finally:
        proc.kill()
        proc.wait()
    if not mels:
        return None, None
    # One patch per run: the same output as one batch, at a fraction of the peak memory.
    out = np.concatenate([models["effnet"].run(["PartitionedCall:1"], {"serving_default_melspectrogram:0": m[None]})[0]
                          for m in mels])
    vec = out.mean(axis=0, keepdims=True).astype(np.float32)
    mood = {m: float(models[m].run(None, {"embeddings": vec})[0][0][pos]) for m, (_, pos) in MOODS.items()}
    return vec[0], mood
