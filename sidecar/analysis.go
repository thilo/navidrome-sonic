// Audio analysis: turns a file into an embedding and six mood scores.
//
// EffNet, a network trained on Discogs' genre and style labels, does not read
// raw audio but a spectrogram: how much energy each pitch range has over time.
// The front end builds it exactly as Essentia's TensorflowInputMusiCNN does,
// because EffNet and its mood heads were trained on that input; any other
// would shift every embedding. The audio is resampled to 16 kHz mono and cut
// into overlapping 32 ms frames. Each frame's spectrum (FFT) is summed into
// 96 mel bands, spaced as the ear tells pitches apart, and log-compressed.
// 128 frames, about two seconds, make one patch.
//
// sonic-decode (decode.c) supplies up to 20 patches spread evenly over the
// track, and silent ones are skipped. EffNet maps each patch to 1280 numbers;
// the embedding is their mean, and each mood head, a small network of its
// own, maps that mean to a score between 0 and 1.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/dsp/fourier"
)

const modelDir = "/opt/sonic/models"

// Mood -> the Essentia head that scores it, and which of the head's two
// outputs means yes (from the head's metadata).
var moods = []struct {
	name, head string
	pos        int
}{
	{"happy", "mood_happy", 0}, {"sad", "mood_sad", 1}, {"relaxed", "mood_relaxed", 1},
	{"aggressive", "mood_aggressive", 0}, {"party", "mood_party", 1}, {"danceable", "danceability", 0},
}

// EffNet front end, as in Essentia's TensorflowInputMusiCNN and the model's patch size.
const (
	sr, nFFT, hop, mels, patch = 16000, 512, 256, 96, 128
	span                       = (patch-1)*hop + nFFT // samples one patch reads
	// Patch selection, as in LMS: up to 20 evenly spaced patches, silent ones skipped.
	// Essentia runs every patch, 62 frames apart.
	maxPatches  = 20
	minPatchRMS = 0.003
)

// sonic-decode, built from decode.c, next to the sidecar's binary.
var decodeBin = func() string {
	exe, _ := os.Executable()
	return filepath.Join(filepath.Dir(exe), "sonic-decode")
}()

var (
	fb  = melFilterbank()
	win = func() (w [nFFT]float32) {
		for i := range w {
			w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/(nFFT-1))))
		}
		return
	}()
	fft = fourier.NewFFT(nFFT)
)

// melFilterbank returns Essentia's MelBands as TensorflowInputMusiCNN sets them: Slaney
// mel scale, triangles over the bin frequencies, each divided by its area in Hz (unit_tri).
func melFilterbank() (out [mels][nFFT/2 + 1]float32) {
	slaney := func(f float64) float64 {
		if f < 1000 {
			return f * 3 / 200
		}
		return 15 + math.Log(f/1000)/(math.Log(6.4)/27)
	}
	hz := func(m float64) float64 {
		if m < 15 {
			return m * 200 / 3
		}
		return 1000 * math.Exp((m-15)*math.Log(6.4)/27)
	}
	var edges [mels + 2]float64
	for i := range edges {
		edges[i] = hz(slaney(sr/2) * float64(i) / (mels + 1))
	}
	for m := range out {
		lo, peak, hi := edges[m], edges[m+1], edges[m+2]
		for k := range out[m] {
			f := float64(k) * sr / nFFT
			out[m][k] = float32(math.Max(0, math.Min((f-lo)/(peak-lo), (hi-f)/(hi-peak))) / ((hi - lo) / 2))
		}
	}
	return
}

// frames returns how many frames a track of that duration yields.
func frames(duration float64) int {
	total := int(duration * sr)
	if total < nFFT/2 {
		return 0
	}
	return 1 + (total-nFFT/2)/hop
}

// layout returns the signal sample of the first window and the distance between
// windows, laid out like LMS: gap, patch, gap, patch, ... frameCount 0 means unknown.
// The first window may start at -hop, in the zero padding.
func layout(frameCount int) (first, step int) {
	n := min(maxPatches, frameCount/patch)
	gap := 0
	if n > 0 {
		gap = (frameCount - n*patch) / (n + 1)
	} else if frameCount == 0 {
		gap = 2 * patch
	}
	return gap*hop - hop, (patch + gap) * hop
}

// logMel returns a patch's spectrogram, as log-mel bands (patch x mels,
// row-major), and its loudness as mean frame RMS, which decides whether the
// patch counts as silent.
func logMel(seg []float32) ([]float32, float64) {
	out := make([]float32, patch*mels)
	seq := make([]float64, nFFT)
	coef := make([]complex128, nFFT/2+1)
	power := make([]float64, nFFT/2+1)
	rms := 0.0
	for t := range patch {
		frame := seg[t*hop : t*hop+nFFT]
		sq := 0.0
		for _, x := range frame[:hop] {
			sq += float64(x) * float64(x)
		}
		rms += math.Sqrt(sq / hop)
		for i, x := range frame {
			seq[i] = float64(x * win[i])
		}
		fft.Coefficients(coef, seq)
		for k, c := range coef {
			power[k] = real(c)*real(c) + imag(c)*imag(c)
		}
		for m := range mels {
			s := 0.0
			for k, w := range fb[m] {
				s += power[k] * float64(w)
			}
			out[t*mels+m] = float32(math.Log10(10000*s + 1))
		}
	}
	return out, rms / patch
}

type song struct {
	Name       string   `json:"name"`
	Artists    []artist `json:"artists"`
	Album      string   `json:"album"`
	MBID       string   `json:"mbid"`
	AlbumMBID  string   `json:"albumMbid"`
	DurationMs int64    `json:"durationMs"`
}

type artist struct {
	Name string `json:"name"`
}

// probe returns the duration in seconds, the SongRef Navidrome matches by, and the album artist.
func probe(path string) (float64, song, string, error) {
	out, err := exec.Command("ffprobe", "-v", "error", "-select_streams", "a:0", "-of", "json",
		"-show_entries", "format=duration:format_tags:stream_tags", path).Output()
	if err != nil {
		return 0, song{}, "", fmt.Errorf("ffprobe: %w", err)
	}
	var info struct {
		Format struct {
			Duration string
			Tags     map[string]string
		}
		Streams []struct{ Tags map[string]string }
	}
	if err := json.Unmarshal(out, &info); err != nil {
		return 0, song{}, "", err
	}
	all := []map[string]string{info.Format.Tags}
	for _, st := range info.Streams {
		all = append(all, st.Tags)
	}
	tags := map[string]string{}
	for _, t := range all {
		for k, v := range t {
			tags[strings.ToLower(k)] = strings.TrimSpace(v)
		}
	}
	get := func(keys ...string) string {
		for _, k := range keys {
			if tags[k] != "" {
				return tags[k]
			}
		}
		return ""
	}
	duration, _ := strconv.ParseFloat(info.Format.Duration, 64)
	// Navidrome's fallbacks for untagged files, so its matcher finds them.
	var artists []artist
	for _, a := range strings.Split(get("artist"), ";") {
		if a = strings.TrimSpace(a); a != "" {
			artists = append(artists, artist{a})
		}
	}
	if artists == nil {
		artists = []artist{{"[Unknown Artist]"}}
	}
	albumArtist := get("album_artist", "albumartist")
	if albumArtist == "" {
		albumArtist = artists[0].Name
	}
	s := song{
		Name:       get("title"),
		Artists:    artists,
		Album:      get("album"),
		MBID:       get("musicbrainz_trackid", "musicbrainz track id"),
		AlbumMBID:  get("musicbrainz_albumid", "musicbrainz album id"),
		DurationMs: int64(duration * 1000),
	}
	if s.Name == "" {
		base := filepath.Base(path)
		s.Name = strings.TrimSuffix(base, filepath.Ext(base))
	}
	if s.Album == "" {
		s.Album = "[Unknown Album]"
	}
	return duration, s, albumArtist, nil
}

// audioHash returns the SHA-256 of the first audio stream's packets, copied without
// decoding: equal for a moved or retagged file, so its embedding can be reused.
func audioHash(path string) (string, error) {
	out, err := exec.Command(decodeBin, "hash", path).Output()
	if err != nil {
		return "", fmt.Errorf("sonic-decode: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// A model maps one input tensor to its first output.
type model func(in []float32) ([]float32, error)

// loadModels returns EffNet as "effnet" and the mood heads by mood name.
func loadModels() (map[string]model, error) {
	ort.SetSharedLibraryPath("libonnxruntime.so.1")
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	for _, err := range []error{opts.SetIntraOpNumThreads(1), opts.SetInterOpNumThreads(1),
		// Otherwise onnxruntime keeps its largest activations allocated between tracks.
		opts.SetCpuMemArena(false), opts.SetMemPattern(false)} {
		if err != nil {
			return nil, err
		}
	}
	load := func(name, output string, shape ort.Shape) (model, error) {
		path := fmt.Sprintf("%s/%s.onnx", modelDir, name)
		inputs, outputs, err := ort.GetInputOutputInfoWithOptions(path, opts)
		if err != nil {
			return nil, err
		}
		if output == "" {
			output = outputs[0].Name
		}
		s, err := ort.NewDynamicAdvancedSession(path, []string{inputs[0].Name}, []string{output}, opts)
		if err != nil {
			return nil, err
		}
		return func(in []float32) ([]float32, error) {
			t, err := ort.NewTensor(shape, in)
			if err != nil {
				return nil, err
			}
			defer t.Destroy()
			out := []ort.Value{nil}
			if err := s.Run([]ort.Value{t}, out); err != nil {
				return nil, err
			}
			defer out[0].Destroy()
			return append([]float32(nil), out[0].(*ort.Tensor[float32]).GetData()...), nil
		}, nil
	}
	m := map[string]model{}
	if m["effnet"], err = load("discogs-effnet", "PartitionedCall:1", ort.NewShape(1, patch, mels)); err != nil {
		return nil, err
	}
	for _, md := range moods {
		if m[md.name], err = load(md.head+"-discogs-effnet-1", "", ort.NewShape(1, 1280)); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// embed returns the track's mean embedding and its mood scores, or nil if it is silent.
func embed(models map[string]model, path string, duration float64) ([]float32, map[string]float64, error) {
	first, step := layout(frames(duration))
	cmd := exec.Command(decodeBin, "windows", path, strconv.Itoa(first), strconv.Itoa(step), strconv.Itoa(span))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}
	// One patch at a time: the same output as one batch, at a fraction of the peak memory.
	var sum []float32
	n := 0
	buf := make([]byte, span*4)
	seg := make([]float32, span)
	for {
		if _, err := io.ReadFull(stdout, buf); err != nil {
			break
		}
		for i := range seg {
			seg[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
		}
		mel, rms := logMel(seg)
		if rms < minPatchRMS {
			continue
		}
		out, err := models["effnet"](mel)
		if err != nil {
			cmd.Process.Kill()
			cmd.Wait()
			return nil, nil, err
		}
		if sum == nil {
			sum = make([]float32, len(out))
		}
		for i, x := range out {
			sum[i] += x
		}
		n++
	}
	if err := cmd.Wait(); err != nil {
		return nil, nil, fmt.Errorf("sonic-decode: %s", strings.TrimSpace(stderr.String()))
	}
	if n == 0 {
		return nil, nil, nil
	}
	for i := range sum {
		sum[i] /= float32(n)
	}
	scores := map[string]float64{}
	for _, md := range moods {
		out, err := models[md.name](sum)
		if err != nil {
			return nil, nil, err
		}
		scores[md.name] = float64(out[md.pos])
	}
	return sum, scores, nil
}
