// Navidrome plugin that answers sonic-similarity requests, Instant Mix
// (getSimilarSongs2) and weekly mood playlists from the navidrome-sonic sidecar.
// This file registers the plugin and holds the sidecar client; similarity.go
// serves the similarity requests, moods.go the playlists.
package main

import (
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/lifecycle"
	"github.com/navidrome/navidrome/plugins/pdk/go/metadata"
	"github.com/navidrome/navidrome/plugins/pdk/go/scheduler"
	"github.com/navidrome/navidrome/plugins/pdk/go/sonicsimilarity"
	"github.com/navidrome/navidrome/plugins/pdk/go/types"
)

const sidecar = "http://navidrome-sonic:8000"

type plugin struct{}

func init() {
	sonicsimilarity.Register(plugin{})
	metadata.Register(plugin{})
	lifecycle.Register(plugin{})
	scheduler.Register(plugin{})
}

func main() {}

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

// songs returns the tracks of a sidecar answer, best match first.
func songs(res sonicsimilarity.SonicSimilarityResponse) []types.SongRef {
	out := make([]types.SongRef, len(res.Matches))
	for i, m := range res.Matches {
		out[i] = m.Song
	}
	return out
}
