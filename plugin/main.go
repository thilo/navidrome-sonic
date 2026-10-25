// Navidrome plugin that answers sonic-similarity requests and Instant Mix
// (getSimilarSongs2) from the navidrome-sonic sidecar.
//
// The sidecar knows tracks by file, Navidrome by its own ID. A seed goes out as
// the file size, with duration and title to tell apart files of equal size: the
// Matcher service resolves the ID to a track, which carries all three without
// filesystem access. Results come back as tags (title, artists, album, MBIDs,
// duration), which Navidrome matches to its own tracks within the caller's
// libraries.
package main

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strconv"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/metadata"
	"github.com/navidrome/navidrome/plugins/pdk/go/sonicsimilarity"
	"github.com/navidrome/navidrome/plugins/pdk/go/types"
)

const sidecar = "http://navidrome-sonic:8000"

type plugin struct{}

func init() {
	sonicsimilarity.Register(plugin{})
	metadata.Register(plugin{})
}

func main() {}

// seed adds the sidecar's lookup keys for a Navidrome track to params, each name prefixed.
func seed(params url.Values, prefix, id string) error {
	tracks, err := host.MatcherMatchSongs([]types.SongRef{{ID: id}}, host.MatchOptions{})
	if err != nil {
		return err
	}
	if len(tracks) == 0 || tracks[0] == nil {
		return fmt.Errorf("track %s not found", id)
	}
	t := tracks[0]
	params.Set(prefix+"size", strconv.FormatInt(t.Size, 10))
	params.Set(prefix+"duration", strconv.FormatFloat(t.Duration, 'f', -1, 64))
	params.Set(prefix+"title", t.Title)
	return nil
}

func query(endpoint string, params url.Values) (sonicsimilarity.SonicSimilarityResponse, error) {
	var out sonicsimilarity.SonicSimilarityResponse
	resp, err := host.HTTPSend(host.HTTPRequest{Method: "GET", URL: sidecar + endpoint + "?" + params.Encode(), TimeoutMs: 10000})
	if err != nil {
		return out, err
	}
	if resp.StatusCode != 200 {
		return out, fmt.Errorf("navidrome-sonic %s: %d %s", endpoint, resp.StatusCode, resp.Body)
	}
	return out, json.Unmarshal(resp.Body, &out)
}

func (plugin) GetSonicSimilarTracks(req sonicsimilarity.GetSonicSimilarTracksRequest) (sonicsimilarity.SonicSimilarityResponse, error) {
	params := url.Values{"count": {strconv.Itoa(int(req.Count))}}
	if err := seed(params, "", req.Song.ID); err != nil {
		return sonicsimilarity.SonicSimilarityResponse{}, err
	}
	return query("/similar", params)
}

func (plugin) FindSonicPath(req sonicsimilarity.FindSonicPathRequest) (sonicsimilarity.SonicSimilarityResponse, error) {
	params := url.Values{"count": {strconv.Itoa(int(req.Count))}}
	if err := seed(params, "start_", req.StartSong.ID); err != nil {
		return sonicsimilarity.SonicSimilarityResponse{}, err
	}
	if err := seed(params, "end_", req.EndSong.ID); err != nil {
		return sonicsimilarity.SonicSimilarityResponse{}, err
	}
	return query("/path", params)
}

// GetSimilarSongsByTrack serves Instant Mix and getSimilarSongs2. An error (track
// not analysed yet, sidecar down) lets Navidrome fall through to the next agent.
func (p plugin) GetSimilarSongsByTrack(req metadata.SimilarSongsByTrackRequest) (*metadata.SimilarSongsResponse, error) {
	res, err := p.GetSonicSimilarTracks(sonicsimilarity.GetSonicSimilarTracksRequest{Song: types.SongRef{ID: req.ID}, Count: req.Count})
	if err != nil {
		return nil, err
	}
	songs := make([]types.SongRef, len(res.Matches))
	for i, m := range res.Matches {
		songs[i] = m.Song
	}
	return &metadata.SimilarSongsResponse{Songs: songs}, nil
}
