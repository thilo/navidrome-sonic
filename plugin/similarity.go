// Similar tracks: the OpenSubsonic getSonicSimilarTracks and findSonicPath
// calls, and Instant Mix and getSimilarSongs2 through the metadata agent.
//
// Navidrome knows a track by its ID, the sidecar by its file. seed resolves
// the ID with the Matcher service, which returns the file size, duration and
// title without filesystem access, and sends those. The match is not scoped to
// a user: Navidrome has already checked that the caller can see the seed, and
// it filters the answer to the caller's libraries itself.
package main

import (
	"fmt"
	"net/url"
	"strconv"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/metadata"
	"github.com/navidrome/navidrome/plugins/pdk/go/sonicsimilarity"
	"github.com/navidrome/navidrome/plugins/pdk/go/types"
)

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
//
// Navidrome matches the answer within the requesting user's libraries, drops
// repeats and trims it to the requested count. The plugin does not learn who
// asks, so it requests three times as many: tracks in libraries that user
// cannot open then leave room for ones in the libraries they share.
func (p plugin) GetSimilarSongsByTrack(req metadata.SimilarSongsByTrackRequest) (*metadata.SimilarSongsResponse, error) {
	res, err := p.GetSonicSimilarTracks(sonicsimilarity.GetSonicSimilarTracksRequest{Song: types.SongRef{ID: req.ID}, Count: 3 * req.Count})
	if err != nil {
		return nil, err
	}
	return &metadata.SimilarSongsResponse{Songs: songs(res)}, nil
}
