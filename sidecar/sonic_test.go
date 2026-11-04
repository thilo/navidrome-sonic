// Self-check for the sidecar and sonic-decode: `go test`, with ffmpeg and
// sonic-decode (built from decode.c) in this directory.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"
)

func init() { decodeBin, _ = filepath.Abs("sonic-decode") }

func f32bytes(s []float32) []byte {
	b := make([]byte, 4*len(s))
	for i, x := range s {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func floats(b []byte) []float32 {
	s := make([]float32, len(b)/4)
	for i := range s {
		s[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return s
}

// patches returns the span-sample windows of the zero-left-padded f32le signal at
// layout(frameCount) that the signal fills: what sonic-decode writes, as the reference.
func patches(pcm []byte, frameCount int) [][]byte {
	padded := append(make([]byte, 4*hop), pcm[:len(pcm)/4*4]...)
	first, step := layout(frameCount)
	var out [][]byte
	for s := first + hop; 4*(s+span) <= len(padded); s += step {
		out = append(out, padded[4*s:4*(s+span)])
	}
	return out
}

func run(t *testing.T, stdin []byte, name string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Stdin = bytes.NewReader(stdin)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%s %v: %v: %s", name, args, err, stderr.String())
	}
	return out
}

func ffmpeg(t *testing.T, args ...string) {
	t.Helper()
	run(t, nil, "ffmpeg", append([]string{"-nostdin", "-v", "error", "-y"}, args...)...)
}

func names(res []match) (out []string) {
	for _, m := range res {
		out = append(out, m.Song.Name)
	}
	return
}

func TestFrontEnd(t *testing.T) {
	// Patch layout: 100 s known -> 20 patches with equal gaps; unknown -> gap of 2 patches.
	rng := rand.New(rand.NewSource(0))
	sig := make([]float32, sr*100)
	for i := range sig {
		sig[i] = float32(rng.NormFloat64())
	}
	pcm := f32bytes(sig)
	frameCount := frames(float64(len(sig)) / sr)
	segs := patches(pcm, frameCount)
	if len(segs) != 20 {
		t.Fatal(len(segs))
	}
	gap := (frameCount - 20*patch) / 21
	first := gap*hop - hop // padded index -> signal index
	if !bytes.Equal(segs[0], pcm[4*first:4*(first+span)]) {
		t.Fatal("first window")
	}
	at := (5*patch - 1) * hop
	if unknown := patches(pcm, 0); !bytes.Equal(unknown[1], pcm[4*at:4*(at+span)]) {
		t.Fatal("unknown duration")
	}
	mel, rms := logMel(floats(segs[0]))
	if len(mel) != patch*mels || rms <= minPatchRMS {
		t.Fatal(len(mel), rms)
	}
	// Front end against Essentia: TensorflowInputMusiCNN() on the first 512 samples of
	// 0.5 sin(1 kHz) + 0.1 sin(150 Hz), bands 3, 10, 30, 31, 32, 60, 95.
	tone := make([]float32, span)
	for i := range tone {
		x := float64(i) / sr
		tone[i] = float32(0.5*math.Sin(2*math.Pi*1000*x) + 0.1*math.Sin(2*math.Pi*150*x))
	}
	mel, _ = logMel(tone)
	want := map[int]float64{3: 4.338388, 10: 0.014412, 30: 5.446386, 31: 6.057385, 32: 5.667006, 60: 2.07e-07, 95: 0}
	for b, v := range want {
		if math.Abs(float64(mel[b])-v) > 1e-4 {
			t.Errorf("band %d: %g, want %g", b, mel[b], v)
		}
	}
}

func TestTopEigen(t *testing.T) {
	// B Bᵀ with B 200x30: 30 positive eigenvalues, then a cluster of zeros.
	n, k := 200, 60
	rng := rand.New(rand.NewSource(2))
	b := mat.NewDense(n, 30, nil)
	for i := range n {
		for j := range 30 {
			b.Set(i, j, rng.NormFloat64())
		}
	}
	var a mat.Dense
	a.Mul(b, b.T())
	var ref mat.EigenSym
	ref.Factorize(mat.NewSymDense(n, append([]float64(nil), a.RawMatrix().Data...)), false)
	want := ref.Values(nil)
	val, vec := topEigen(append([]float64(nil), a.RawMatrix().Data...), n, k)
	v := mat.NewDense(n, k, vec)
	var av, vtv mat.Dense
	av.Mul(&a, v)
	vtv.Mul(v.T(), v)
	scale := want[n-1]
	for c := range k {
		if math.Abs(val[c]-want[n-1-c]) > 1e-10*scale {
			t.Errorf("eigenvalue %d: %g, want %g", c, val[c], want[n-1-c])
		}
		for i := range n {
			if r := av.At(i, c) - val[c]*v.At(i, c); math.Abs(r) > 1e-9*scale {
				t.Fatalf("residual %d: %g", c, r)
			}
		}
		for p := range k {
			want := 0.0
			if p == c {
				want = 1
			}
			if math.Abs(vtv.At(p, c)-want) > 1e-9 {
				t.Fatalf("VᵀV[%d,%d] = %g", p, c, vtv.At(p, c))
			}
		}
	}
}

type row struct {
	path, artist string
	vec          []float64
	size         int64
	seconds      int64
	name         string
	happy        float64
}

func sizeOf(path string) int64 {
	h := fnv.New32a()
	h.Write([]byte(path))
	return int64(h.Sum32())
}

// newTestIndex builds an index over rows; the embedding bytes stand in for the audio hash.
func newTestIndex(t *testing.T, rows []row) *Index {
	t.Helper()
	db, err := openDB(filepath.Join(t.TempDir(), "sonic.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := initDB(db); err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		name, size := r.name, r.size
		if name == "" {
			name = r.path
		}
		if size == 0 {
			size = sizeOf(r.path)
		}
		s := song{Name: name, Artists: []artist{{r.artist}}, Album: name, DurationMs: r.seconds * 1000}
		meta, _ := json.Marshal(map[string]any{"song": s, "albumArtist": r.artist})
		mood := map[string]float64{}
		for _, m := range moods {
			mood[m.name] = 0.5
		}
		if r.happy != 0 {
			mood["happy"] = r.happy
		}
		md, _ := json.Marshal(mood)
		v := make([]float32, len(r.vec))
		for i, x := range r.vec {
			v[i] = float32(x)
		}
		emb := f32bytes(v)
		if _, err := db.Exec("INSERT INTO track (path, meta, emb, audio, size, mood) VALUES (?, ?, ?, ?, ?, ?)",
			r.path, string(meta), emb, fmt.Sprintf("%x", emb), size, string(md)); err != nil {
			t.Fatal(err)
		}
	}
	idx, err := newIndex(db)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func TestIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	normal := func(scale float64, base []float64) []float64 {
		v := make([]float64, 1280)
		for i := range v {
			v[i] = scale * rng.NormFloat64()
			if base != nil {
				v[i] += base[i]
			}
		}
		return v
	}
	// Ranking: a same-artist neighbour loses to a slightly farther other artist.
	base := normal(1, nil)
	rows := []row{{path: "seed", artist: "A", vec: base}, {path: "same", artist: "A", vec: normal(0.01, base)},
		{path: "other", artist: "B", vec: normal(0.02, base)}}
	for i := range 80 {
		rows = append(rows, row{path: fmt.Sprintf("far%d", i), artist: fmt.Sprintf("F%d", i), vec: normal(1, nil)})
	}
	// Size lookup: three files of one size, told apart by duration, then title.
	for _, r := range []struct {
		name string
		s    int64
	}{{"x", 100}, {"y", 200}, {"z", 200}} {
		rows = append(rows, row{path: r.name, artist: "C", vec: normal(1, nil), size: 1000, seconds: r.s})
	}
	// The seed's file again in another library: never offered as similar to itself.
	rows = append(rows, row{path: "seed-copy", artist: "A", vec: base, name: "seed"})
	if p := newTestIndex(t, rows[:2]).paths; len(p) != 0 {
		t.Fatal(p)
	}
	idx := newTestIndex(t, rows)
	for _, c := range []struct {
		size     int64
		duration float64
		title    string
		want     string
	}{{1000, 199.4, "z", "z"}, {1000, 120, "", "x"}, {1000, 200, "y", "y"}, {sizeOf("seed"), 0, "", "seed"}} {
		if i, err := idx.find(c.size, c.duration, c.title); err != nil || idx.paths[i] != c.want {
			t.Errorf("find %v: %d %v", c, i, err)
		}
	}
	if _, err := idx.find(1, 0, ""); err == nil {
		t.Error("unknown size found")
	}
	res, _ := idx.similar(0, 2)
	if picks := names(res); !slices.Equal(picks, []string{"other", "same"}) {
		t.Error(picks)
	}
	res, _ = idx.route(0, 3, 4)
	route := names(res)
	if route[0] != "seed" || route[len(route)-1] != "far0" || len(slices.Compact(slices.Sorted(slices.Values(route)))) != len(route) {
		t.Error(route)
	}
	// A track removed after the build is no longer offered.
	if _, err := idx.db.Exec("DELETE FROM track WHERE path = 'other'"); err != nil {
		t.Fatal(err)
	}
	sim, _ := idx.similar(0, 10)
	res, _ = idx.route(0, 3, 4)
	if slices.Contains(append(names(sim), names(res)...), "other") || len(sim) == 0 {
		t.Error(names(sim), names(res))
	}

	// Moods: highest score first, a recording once even when it is in two libraries.
	m := newTestIndex(t, []row{{path: "sad", artist: "A", vec: base, happy: 0.1}, {path: "glad", artist: "B", vec: base, happy: 0.9},
		{path: "glad-copy", artist: "B", vec: base, name: "glad", happy: 0.9}, {path: "ok", artist: "C", vec: base, happy: 0.6}})
	res, _ = m.mood("happy", 10)
	if n := names(res); !slices.Equal(n, []string{"glad", "ok", "sad"}) {
		t.Error(n)
	}
	res, _ = m.mood("happy", 2)
	if len(res) != 2 || res[0].Similarity != 0.9 || res[1].Similarity != 0.6 {
		t.Error(res)
	}
}

func TestScan(t *testing.T) {
	// A moved file and a retagged one keep their embeddings and mood scores; a
	// database from another analysis version starts over.
	runs := 0 // each head runs once per analysed track
	models := map[string]model{"effnet": func(in []float32) ([]float32, error) {
		s := 0.0
		for _, x := range in {
			s += float64(x)
		}
		out := make([]float32, 1280)
		for i := range out {
			out[i] = float32(s / float64(len(in)))
		}
		return out, nil
	}}
	for _, md := range moods {
		models[md.name] = func(in []float32) ([]float32, error) {
			runs++
			p := float64(in[0])
			p -= math.Floor(p)
			return []float32{float32(p), float32(1 - p)}, nil
		}
	}
	analysed := func() int { return runs / len(moods) }
	music = t.TempDir()
	ffmpeg(t, "-f", "lavfi", "-i", "sine=f=440:d=5", "-metadata", "title=A", music+"/a.mp3")
	ffmpeg(t, "-f", "lavfi", "-i", "sine=f=660:d=5", "-metadata", "title=B", music+"/b.mp3")
	db, err := openDB(filepath.Join(t.TempDir(), "sonic.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := initDB(db); err != nil {
		t.Fatal(err)
	}
	type stored struct{ name, audio, emb, mood string }
	tracks := func() map[string]stored {
		res, err := db.Query("SELECT path, meta, audio, emb, mood FROM track")
		if err != nil {
			t.Fatal(err)
		}
		defer res.Close()
		out := map[string]stored{}
		for res.Next() {
			var p, meta string
			var s stored
			if err := res.Scan(&p, &meta, &s.audio, &s.emb, &s.mood); err != nil {
				t.Fatal(err)
			}
			s.name = parse(meta).song.Name
			rel, _ := filepath.Rel(music, p)
			out[rel] = s
		}
		return out
	}
	mustScan := func() bool {
		changed, err := scan(db, models)
		if err != nil {
			t.Fatal(err)
		}
		return changed
	}
	mustScan()
	before := map[string]stored{}
	for _, s := range tracks() {
		before[s.audio] = s
	}
	var scores map[string]float64
	json.Unmarshal([]byte(before[tracks()["a.mp3"].audio].mood), &scores)
	if analysed() != 2 || len(before) != 2 || len(scores) != len(moods) {
		t.Fatal(analysed(), before)
	}
	os.Mkdir(music+"/moved", 0o755)
	os.Rename(music+"/a.mp3", music+"/moved/a.mp3")
	ffmpeg(t, "-i", music+"/b.mp3", "-c", "copy", "-metadata", "title=B2", music+"/b2.mp3")
	os.Rename(music+"/b2.mp3", music+"/b.mp3")
	os.Chtimes(music+"/b.mp3", time.Unix(0, 0), time.Unix(0, 0))
	mustScan()
	after := tracks()
	if analysed() != 2 || len(after) != 2 || after["b.mp3"].name != "B2" || after["moved/a.mp3"].name != "A" {
		t.Fatal(analysed(), after)
	}
	for _, s := range after {
		if b := before[s.audio]; b.emb != s.emb || b.mood != s.mood {
			t.Error("embedding not reused", s.name)
		}
	}
	if mustScan() || analysed() != 2 {
		t.Error("unchanged library rescanned", analysed())
	}
	db.Exec("PRAGMA user_version = 1")
	initDB(db)
	if len(tracks()) != 0 {
		t.Fatal("old analysis version kept")
	}
	mustScan()
	if analysed() != 4 {
		t.Error(analysed())
	}
}

// sonic-decode against the ffmpeg CLI it replaces: the same packet hash, and the same
// windows as patches() over the CLI's decode.
func TestDecode(t *testing.T) {
	cliHash := func(path string) string {
		out := run(t, nil, "ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0", "-c", "copy",
			"-f", "hash", "-hash", "sha256", "-")
		return strings.SplitN(strings.TrimSpace(string(out)), "=", 2)[1]
	}
	cliWindows := func(path string, frameCount int) [][]byte {
		return patches(run(t, nil, "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", path, "-map", "0:a:0",
			"-ac", "1", "-ar", strconv.Itoa(sr), "-f", "f32le", "-"), frameCount)
	}
	windows := func(path string, frameCount int) (out [][]byte) {
		first, step := layout(frameCount)
		b := run(t, nil, decodeBin, "windows", path, strconv.Itoa(first), strconv.Itoa(step), strconv.Itoa(span))
		for i := 0; i < len(b); i += span * 4 {
			out = append(out, b[i:i+span*4])
		}
		return
	}
	equal := func(a, b [][]byte) bool { return slices.EqualFunc(a, b, bytes.Equal) }
	tmp := t.TempDir()

	// Layout, through a 16 kHz mono float WAV, which decodes to the signal itself: gaps,
	// unknown duration, and gap 0 (first window at -hop, overlapping windows).
	rng := rand.New(rand.NewSource(0))
	sig := make([]float32, sr*100)
	for i := range sig {
		sig[i] = float32(rng.NormFloat64())
	}
	short := sig[:(20*patch-1)*hop+nFFT/2]
	if first, step := layout(20 * patch); first != -hop || step >= span {
		t.Fatal(first, step)
	}
	for _, c := range []struct {
		s     []float32
		count int
	}{{sig, frames(100)}, {sig, 0}, {short, 20 * patch}} {
		run(t, f32bytes(c.s), "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ar", strconv.Itoa(sr),
			"-ac", "1", "-i", "-", "-c:a", "pcm_f32le", tmp+"/layout.wav")
		want := patches(f32bytes(c.s), c.count)
		if len(want) == 0 || !equal(windows(tmp+"/layout.wav", c.count), want) {
			t.Error("layout", c.count)
		}
	}

	// Codecs and containers, windowed and decoded in full. 120 s put the windows far enough
	// apart for separate segments, also with FLAC's and WAV's 4096-sample packets.
	ffmpeg(t, "-f", "lavfi", "-i", "aevalsrc=0.4*sin(2*PI*220*t)+0.1*(2*random(0)-1)|"+
		"0.4*sin(2*PI*331*t)+0.1*(2*random(1)-1):s=44100:d=120", tmp+"/src.wav")
	files := []string{}
	for _, f := range []struct {
		name string
		args []string
	}{{"a.mp3", []string{"-c:a", "libmp3lame"}}, {"b.mp3", []string{"-ar", "48000", "-c:a", "libmp3lame"}},
		{"c.flac", nil}, {"d.wav", []string{"-c:a", "pcm_s16le"}}, {"e.ogg", []string{"-c:a", "libvorbis"}},
		{"f.m4a", []string{"-c:a", "aac"}}, {"g.opus", []string{"-c:a", "libopus"}}, {"h.wma", []string{"-c:a", "wmav2"}}} {
		ffmpeg(t, append(append([]string{"-i", tmp + "/src.wav"}, f.args...), tmp+"/"+f.name)...)
		files = append(files, f.name)
	}
	// Restarts in full mode and filter graph rebuilds: a chained Ogg and a mp3 whose sample
	// rate and channels change; a flac frame without its sync code, which the decoder
	// drops, so the later samples move up; a mp3 frame without its header, which never
	// becomes a packet.
	read := func(name string) []byte {
		b, err := os.ReadFile(tmp + "/" + name)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	cat := func(name string, parts ...string) {
		var b []byte
		for _, p := range parts {
			b = append(b, read(p)...)
		}
		os.WriteFile(tmp+"/"+name, b, 0o644)
		files = append(files, name)
	}
	// damage overwrites the start of the packet nearest `at` seconds.
	damage := func(name, source string, at float64, patch []byte) {
		var info struct {
			Packets []struct {
				PtsTime string `json:"pts_time"`
				Pos     string
			}
		}
		out := run(t, nil, "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
			"packet=pts_time,pos", "-of", "json", tmp+"/"+source)
		if err := json.Unmarshal(out, &info); err != nil {
			t.Fatal(err)
		}
		pos, best := 0, math.Inf(1)
		for _, p := range info.Packets {
			pts, _ := strconv.ParseFloat(p.PtsTime, 64)
			if d := math.Abs(pts - at); d < best {
				best = d
				pos, _ = strconv.Atoi(p.Pos)
			}
		}
		b := read(source)
		copy(b[pos:], patch)
		os.WriteFile(tmp+"/"+name, b, 0o644)
		files = append(files, name)
	}
	ffmpeg(t, "-i", tmp+"/src.wav", "-t", "60", "-c:a", "libvorbis", tmp+"/x1.ogg")
	ffmpeg(t, "-i", tmp+"/src.wav", "-ss", "60", "-ar", "22050", "-ac", "1", "-c:a", "libvorbis", tmp+"/x2.ogg")
	cat("chained.ogg", "x1.ogg", "x2.ogg")
	ffmpeg(t, "-i", tmp+"/src.wav", "-t", "60", "-c:a", "libmp3lame", tmp+"/x1.mp3")
	ffmpeg(t, "-i", tmp+"/src.wav", "-ss", "60", "-ar", "48000", "-ac", "1", "-c:a", "libmp3lame",
		"-write_xing", "0", "-id3v2_version", "0", tmp+"/x2.mp3")
	cat("rates.mp3", "x1.mp3", "x2.mp3")
	// Between two segments, which only the packet check sees: after window 9 ends and
	// before window 10's preroll (about a second) starts.
	duration, _, _, err := probe(tmp + "/c.flac")
	if err != nil {
		t.Fatal(err)
	}
	first, step := layout(frames(duration))
	damage("nosync.flac", "c.flac", float64(first+9*step+span)/sr+0.8, make([]byte, 2))
	damage("noheader.mp3", "a.mp3", 60, make([]byte, 4))
	for _, name := range files {
		path := tmp + "/" + name
		duration, _, _, err := probe(path)
		if err != nil {
			t.Fatal(err)
		}
		count := frames(duration)
		want := cliWindows(path, count)
		if h, err := audioHash(path); err != nil || h != cliHash(path) {
			t.Error("hash", name, err)
		}
		if len(want) == 0 || !equal(windows(path, count), want) {
			t.Error("windows", name)
		}
	}
}
