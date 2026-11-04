// The search index: answers "which tracks sound like this one?"
//
// Every analysed track has an embedding, 1280 numbers from EffNet in which
// tracks that sound alike lie close together. The index reduces them to 60
// with PCA over the library: the 60 directions in which its tracks differ
// most, rescaled so each counts about equally ("whitening"). Two tracks are
// similar when their 60-D vectors point the same way (cosine distance), and a
// threshold derived from the library's nearest-neighbour distances drops
// candidates too far off to count.
//
// A mix is picked one track at a time, as in LMS
// (src/libs/services/recommendation/impl/audio-similarity): close to the seed,
// close to the previous pick, and penalised for repeating a release or an
// artist among the last few picks.
//
// Only the vectors live in memory; tags and mood scores are read from the
// database per query, so the index stays small next to the one being built
// to replace it.
package main

import (
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sort"
	"strings"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack/gonum"
	"gonum.org/v1/gonum/mat"
)

// Ranking, as in LMS's AudioSimilarityEngine.
const (
	dims, oversample, pathNeighbours, window = 60, 5, 32, 4
	wFit, wSmooth, wRelease, wArtist         = 0.8, 0.2, 0.5, 0.5
)

// notFound is a seed the index cannot answer for; the API returns 404.
type notFound string

func (e notFound) Error() string { return "not analysed: " + string(e) }

// What a query needs from a track's stored tags.
type track struct {
	song    song
	release [2]string
	artists map[string]bool
	// A row repeats a picked one if it shares its MBID or all its tags, as the
	// same file in a user's own library and a shared one does: Navidrome
	// resolves both answers to the same track.
	keys []string
}

func parse(meta string) *track {
	var m struct {
		Song        song
		AlbumArtist string
	}
	if err := json.Unmarshal([]byte(meta), &m); err != nil {
		panic(err)
	}
	all, _ := json.Marshal(m.Song)
	t := &track{song: m.Song, release: [2]string{strings.ToLower(m.AlbumArtist), strings.ToLower(m.Song.Album)},
		artists: map[string]bool{}, keys: []string{string(all)}}
	for _, a := range m.Song.Artists {
		t.artists[strings.ToLower(a.Name)] = true
	}
	if m.Song.MBID != "" {
		t.keys = append(t.keys, m.Song.MBID)
	}
	return t
}

func (t *track) in(taken map[string]bool) bool {
	for _, k := range t.keys {
		if taken[k] {
			return true
		}
	}
	return false
}

// rows is one query's view of the track table: a track's parsed tags by index
// position, read once. nil for a track removed or emptied since the index was built.
// A database error panics, which net/http turns into a dropped connection.
type rows struct {
	idx   *Index
	cache map[int]*track
}

func (r rows) get(c int) *track {
	if t, ok := r.cache[c]; ok {
		return t
	}
	var meta string
	err := r.idx.db.QueryRow("SELECT meta FROM track WHERE path = ? AND emb IS NOT NULL", r.idx.paths[c]).Scan(&meta)
	var t *track
	if err == nil {
		t = parse(meta)
	} else if err != sql.ErrNoRows {
		panic(err)
	}
	r.cache[c] = t
	return t
}

type Index struct {
	db        *sql.DB
	paths     []string // filled once vectors exist; a query before that is "not analysed"
	pos       map[string]int
	vec       []float32 // len(paths) x dims, unit rows
	threshold float64
}

func newIndex(db *sql.DB) (*Index, error) {
	idx := &Index{db: db, pos: map[string]int{}, threshold: math.Inf(1)}
	tx, err := db.Begin() // one snapshot for both passes below
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	return idx, idx.build(tx)
}

func (idx *Index) build(tx *sql.Tx) error {
	var paths []string
	res, err := tx.Query("SELECT path, audio FROM track WHERE emb IS NOT NULL ORDER BY rowid")
	if err != nil {
		return err
	}
	last := map[string]int{}
	for res.Next() {
		var p, a string
		if err := res.Scan(&p, &a); err != nil {
			return err
		}
		last[a] = len(paths)
		paths = append(paths, p)
	}
	if err := res.Err(); err != nil {
		return err
	}
	// Statistics come from distinct embeddings: copies of one recording say
	// nothing about the library's spread, and many of them would pull the
	// nearest-neighbour threshold to zero.
	keep := make([]bool, len(paths))
	for _, i := range last {
		keep[i] = true
	}
	n := len(last)
	if n < 3 {
		return nil
	}

	// The N x D matrix is never built: at 1280 dimensions its copies would make
	// up most of the build's memory. One pass over the embeddings sums up the
	// covariance, in float64; a second projects every row.
	embeddings := func(f func(i int, e []float32)) error {
		res, err := tx.Query("SELECT emb FROM track WHERE emb IS NOT NULL ORDER BY rowid")
		if err != nil {
			return err
		}
		defer res.Close()
		for i := 0; res.Next(); i++ {
			var b []byte
			if err := res.Scan(&b); err != nil {
				return err
			}
			e := make([]float32, len(b)/4)
			for j := range e {
				e[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*j:]))
			}
			f(i, e)
		}
		return res.Err()
	}
	var sum []float64
	var cov *mat.SymDense
	var batch []float64
	flush := func() {
		if len(batch) > 0 {
			b := mat.NewDense(len(batch)/len(sum), len(sum), batch)
			cov.SymRankK(cov, 1, b.T())
			batch = batch[:0]
		}
	}
	err = embeddings(func(i int, e []float32) {
		if !keep[i] {
			return
		}
		if sum == nil {
			sum, cov = make([]float64, len(e)), mat.NewSymDense(len(e), nil)
		}
		for j, x := range e {
			sum[j] += float64(x)
			batch = append(batch, float64(x))
		}
		if len(batch) == 256*len(sum) {
			flush()
		}
	})
	if err != nil {
		return err
	}
	flush()
	dim := len(sum)
	mean := mat.NewVecDense(dim, sum)
	mean.ScaleVec(1/float64(n), mean)
	cov.SymRankOne(cov, -float64(n), mean)
	cov.ScaleSym(1/float64(n-1), cov)
	// Both triangles, for Dsytrd: SymDense keeps the upper one.
	a := cov.RawSymmetric().Data
	for i := range dim {
		for j := range i {
			a[i*dim+j] = a[j*dim+i]
		}
	}
	cov = nil
	eigval, eigvec := topEigen(a, dim, dims)
	// LMS whitens fully. With few distinct tracks that leaves every pair about
	// equally far apart, so the eigenvalues are shrunk towards their mean by
	// dims/distinct: a large library is whitened almost as in LMS, a small one
	// stays close to plain PCA.
	lam := make([]float64, dims)
	lamMean := 0.0
	for k := range lam {
		lam[k] = math.Max(eigval[k], 0)
		lamMean += lam[k] / dims
	}
	basis := make([]float32, dim*dims) // dim x dims
	for k := range dims {
		s := math.Sqrt(lam[k] + lamMean*dims/float64(n))
		for j := range dim {
			basis[j*dims+k] = float32(eigvec[j*dims+k] / s)
		}
	}
	meanF := make([]float32, dim)
	for j := range meanF {
		meanF[j] = float32(mean.AtVec(j))
	}
	idx.vec = make([]float32, len(paths)*dims)
	err = embeddings(func(i int, e []float32) {
		v := idx.vec[i*dims : (i+1)*dims]
		for j, x := range e {
			x -= meanF[j]
			for k, b := range basis[j*dims : (j+1)*dims] {
				v[k] += x * b
			}
		}
		normalize(v)
	})
	if err != nil {
		return err
	}
	// Hard limit: mean + 2 sd of nearest-neighbour distances over a sample.
	var u []int
	for i, k := range keep {
		if k {
			u = append(u, i)
		}
	}
	sample := rand.New(rand.NewSource(42)).Perm(len(u))[:min(500, len(u))]
	nn := make([]float64, len(sample))
	for s, i := range sample {
		nn[s] = math.Inf(1)
		for j, c := range u {
			if j != i {
				nn[s] = math.Min(nn[s], (1-dot(idx.vec[u[i]*dims:], idx.vec[c*dims:]))/2)
			}
		}
	}
	m, sd := 0.0, 0.0
	for _, d := range nn {
		m += d / float64(len(nn))
	}
	for _, d := range nn {
		sd += (d - m) * (d - m) / float64(len(nn)-1)
	}
	idx.threshold = m + 2*math.Sqrt(sd)
	idx.paths = paths
	for i, p := range paths {
		idx.pos[p] = i
	}
	return nil
}

// topEigen returns the k largest eigenvalues of the symmetric n×n matrix a
// (row-major, overwritten), descending, and their eigenvectors as the columns of
// an n×k matrix. gonum's Dsyev forms all n eigenvectors, most of the build's time
// at 1280 dimensions; this follows LAPACK's Dsyevx instead: tridiagonal form, all
// eigenvalues, inverse iteration for k of them, back-transformation.
func topEigen(a []float64, n, k int) (val, vec []float64) {
	var impl gonum.Implementation
	d, e, tau := make([]float64, n), make([]float64, n-1), make([]float64, n-1)
	work := []float64{0}
	impl.Dsytrd(blas.Lower, n, a, n, d, e, tau, work, -1)
	work = make([]float64, int(work[0]))
	impl.Dsytrd(blas.Lower, n, a, n, d, e, tau, work, len(work))
	all := append([]float64(nil), d...)
	impl.Dsterf(n, all, append([]float64(nil), e...)) // ascending
	norm := 0.0
	for i := range n {
		r := math.Abs(d[i])
		if i > 0 {
			r += math.Abs(e[i-1])
		}
		if i < n-1 {
			r += math.Abs(e[i])
		}
		norm = math.Max(norm, r)
	}
	eps := 0x1p-52 * norm
	val, vec = make([]float64, k), make([]float64, n*k)
	rng := rand.New(rand.NewSource(1))
	x := make([]float64, n)
	// solve overwrites x with (T - shift I)⁻¹ x, or reports a zero pivot.
	solve := func(shift float64) bool {
		dl, du, dd := slices.Clone(e), slices.Clone(e), make([]float64, n)
		for i := range dd {
			dd[i] = d[i] - shift
		}
		return impl.Dgtsv(n, 1, dl, dd, du, x, 1)
	}
	shift := math.Inf(1)
	for c := range k {
		val[c] = all[n-1-c]
		// Equal eigenvalues get distinct shifts, as in Dstein.
		shift = math.Min(val[c], shift-10*eps)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		for range 4 {
			// A zero pivot means the shift is exact for a leading block; moving it
			// off by eps is enough, and the partly reduced x still serves as a start.
			for !solve(shift) {
				shift -= eps
			}
			unit(x)
			// Orthogonal to the vectors found so far, which inverse iteration alone
			// does not give within a cluster of eigenvalues.
			for p := range c {
				s := 0.0
				for i := range x {
					s += x[i] * vec[i*k+p]
				}
				for i := range x {
					x[i] -= s * vec[i*k+p]
				}
			}
			unit(x)
		}
		for i, v := range x {
			vec[i*k+c] = v
		}
	}
	// vec = Q vec, Q = diag(1, H_0 ... H_{n-2}) with the reflectors Dsytrd left
	// below the subdiagonal, as in LAPACK's Dormtr.
	impl.Dormqr(blas.Left, blas.NoTrans, n-1, k, n-1, a[n:], n, tau, vec[k:], k, work, -1)
	work = make([]float64, int(work[0]))
	impl.Dormqr(blas.Left, blas.NoTrans, n-1, k, n-1, a[n:], n, tau, vec[k:], k, work, len(work))
	return val, vec
}

func unit(x []float64) {
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	if s = math.Sqrt(s); s > 0 {
		for i := range x {
			x[i] /= s
		}
	}
}

func dot(a, b []float32) float64 {
	s := float32(0)
	for k := range dims {
		s += a[k] * b[k]
	}
	return float64(s)
}

func normalize(v []float32) {
	norm := math.Max(math.Sqrt(dot(v, v)), 1e-12)
	for k := range v {
		v[k] = float32(float64(v[k]) / norm)
	}
}

func (idx *Index) rows() rows { return rows{idx, map[int]*track{}} }

// find returns the track of that file size, told apart from others of the same size
// by the closest duration (seconds), then an exact title.
func (idx *Index) find(size int64, duration float64, title string) (int, error) {
	r := idx.rows()
	res, err := idx.db.Query("SELECT path FROM track WHERE size = ?", size)
	if err != nil {
		return 0, err
	}
	var found []int
	for res.Next() {
		var p string
		if err := res.Scan(&p); err != nil {
			return 0, err
		}
		if c, ok := idx.pos[p]; ok {
			found = append(found, c)
		}
	}
	if err := res.Err(); err != nil {
		return 0, err
	}
	best, bestD, bestT := -1, 0.0, true
	for _, c := range found {
		t := r.get(c)
		if t == nil {
			continue
		}
		d, other := math.Abs(float64(t.song.DurationMs)/1000-duration), t.song.Name != title
		if best < 0 || d < bestD || d == bestD && !other && bestT {
			best, bestD, bestT = c, d, other
		}
	}
	if best < 0 {
		return 0, notFound(fmt.Sprint(size))
	}
	return best, nil
}

func (idx *Index) dist(q []float32) []float64 {
	d := make([]float64, len(idx.paths))
	for c := range d {
		d[c] = (1 - dot(q, idx.vec[c*dims:])) / 2
	}
	return d
}

func (idx *Index) at(c int) []float32 { return idx.vec[c*dims : (c+1)*dims] }

func argsort(d []float64) []int {
	order := make([]int, len(d))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return d[order[a]] < d[order[b]] })
	return order
}

// penalty is LMS's soft constraints: distance to the seed and to the previous track
// (smooth, distances from it), plus repeats of a release or artist among the last
// window picks.
func penalty(r rows, c int, chosen []int, smooth []float64, fit float64) float64 {
	score := wFit*fit + wSmooth*smooth[c]
	t := r.get(c)
	for i := 1; i <= min(window, len(chosen)); i++ {
		s := r.get(chosen[len(chosen)-i])
		if s.release == t.release {
			score += wRelease / float64(i)
		}
		for a := range s.artists {
			if t.artists[a] {
				score += wArtist / float64(i)
				break
			}
		}
	}
	return score
}

// best returns the candidate with the lowest penalty, the first on ties.
func best(cands []int, score func(c int) float64) int {
	b, bs := cands[0], score(cands[0])
	for _, c := range cands[1:] {
		if s := score(c); s < bs {
			b, bs = c, s
		}
	}
	return b
}

type match struct {
	Song       song    `json:"song"`
	Similarity float64 `json:"similarity"`
}

func (idx *Index) similar(seed, count int) ([]match, error) {
	r := idx.rows()
	if r.get(seed) == nil {
		return nil, notFound(idx.paths[seed])
	}
	d := idx.dist(idx.at(seed))
	d[seed] = math.Inf(1)
	var ranked []int
	for _, c := range argsort(d)[:min(count*oversample, len(d))] {
		if d[c] <= idx.threshold && r.get(c) != nil {
			ranked = append(ranked, c)
		}
	}
	chosen, out, taken := []int{seed}, []match{}, map[string]bool{}
	for _, k := range r.get(seed).keys {
		taken[k] = true
	}
	for len(out) < count {
		var left []int
		for _, c := range ranked {
			if !r.get(c).in(taken) {
				left = append(left, c)
			}
		}
		if ranked = left; len(ranked) == 0 {
			break
		}
		smooth := idx.dist(idx.at(chosen[len(chosen)-1]))
		b := best(ranked, func(c int) float64 { return penalty(r, c, chosen, smooth, d[c]) })
		chosen = append(chosen, b)
		for _, k := range r.get(b).keys {
			taken[k] = true
		}
		out = append(out, match{r.get(b).song, 1 - d[b]})
	}
	return out, nil
}

// mood returns the tracks with the highest stored score for a mood, each
// recording once. It does not use the vectors.
func (idx *Index) mood(name string, count int) ([]match, error) {
	res, err := idx.db.Query("SELECT meta, json_extract(mood, ?) AS score FROM track WHERE emb IS NOT NULL "+
		"ORDER BY score DESC, rowid", "$."+name)
	if err != nil {
		return nil, err
	}
	defer res.Close()
	out, taken := []match{}, map[string]bool{}
	for len(out) < count && res.Next() {
		var meta string
		var score sql.NullFloat64
		if err := res.Scan(&meta, &score); err != nil {
			return nil, err
		}
		if t := parse(meta); !t.in(taken) {
			out = append(out, match{t.song, score.Float64})
			for _, k := range t.keys {
				taken[k] = true
			}
		}
	}
	return out, res.Err()
}

// route returns a path from start to end: it steps along the straight line
// between their vectors and at each step picks a nearby track under the same
// constraints as a mix.
func (idx *Index) route(start, end, count int) ([]match, error) {
	r := idx.rows()
	for _, c := range []int{start, end} {
		if r.get(c) == nil {
			return nil, notFound(idx.paths[c])
		}
	}
	chosen := []int{start}
	q := make([]float32, dims)
	for i := range max(count-2, 0) {
		t := float32(i+1) / float32(count-1)
		for k := range q {
			q[k] = idx.at(start)[k] + (idx.at(end)[k]-idx.at(start)[k])*t
		}
		normalize(q)
		d := idx.dist(q)
		d[end] = math.Inf(1)
		near := argsort(d)[:min(pathNeighbours, len(d))]
		fit, smooth := idx.dist(idx.at(near[0])), idx.dist(idx.at(chosen[len(chosen)-1]))
		taken := map[string]bool{}
		for _, c := range append(chosen, end) {
			for _, k := range r.get(c).keys {
				taken[k] = true
			}
		}
		var ok []int
		for _, c := range near {
			if t := r.get(c); t != nil && !t.in(taken) {
				ok = append(ok, c)
			}
		}
		if len(ok) > 0 {
			chosen = append(chosen, best(ok, func(c int) float64 { return penalty(r, c, chosen, smooth, fit[c]) }))
		}
	}
	if count > 1 {
		chosen = append(chosen, end)
	}
	d := idx.dist(idx.at(start))
	out := make([]match, len(chosen))
	for i, c := range chosen {
		out[i] = match{r.get(c).song, 1 - d[c]}
	}
	return out, nil
}
