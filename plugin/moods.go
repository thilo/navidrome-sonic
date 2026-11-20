// Mood playlists. Every second day, and once 5 minutes after the plugin loads,
// each user the plugin is granted gets six playlists ("Happy Mix" and so on): 50
// tracks drawn at random from those the sidecar scores at least 0.8 for that mood
// among the tracks in the libraries that user can open. A playlist is found again
// by owner, name and the comment "navidrome-sonic", and its tracks are replaced;
// renaming it or editing the comment detaches it.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/url"
	"strconv"
	"strings"

	"github.com/navidrome/navidrome/plugins/pdk/go/host"
	"github.com/navidrome/navidrome/plugins/pdk/go/scheduler"
)

const (
	// Odd days of the month, so a 31st is followed by a run on the 1st.
	moodCron   = "0 4 */2 * *"
	moodSize   = 50
	moodMarker = "navidrome-sonic"
	// The mood heads output a probability; a track needs this much to count as that mood.
	moodMin = 0.8
	// The sidecar does not know which tracks a user can open, so random pages are
	// matched until the playlist is full. A user who can open under 5% of the
	// tracks that pass moodMin can end up with a shorter playlist.
	moodPage, moodPages = 100, 10
	// Navidrome cuts every plugin call off after 30 seconds, so each user and mood
	// is its own scheduled call, started this many seconds after the previous one.
	moodSpacing = 30
)

// moods maps the sidecar's mood names to playlist names.
var moods = [][2]string{
	{"happy", "Happy Mix"},
	{"sad", "Sad Mix"},
	{"relaxed", "Relaxed Mix"},
	{"aggressive", "Aggressive Mix"},
	{"party", "Party Mix"},
	{"danceable", "Dance Mix"},
}

// OnInit schedules the mood playlists. Navidrome drops a plugin's schedules when
// it unloads, so the fixed IDs never collide.
func (plugin) OnInit() error {
	if _, err := host.SchedulerScheduleRecurring(moodCron, "moods", "sonic-moods"); err != nil {
		return err
	}
	_, err := host.SchedulerScheduleOneTime(300, "moods", "sonic-moods-now")
	return err
}

// OnCallback fans the "moods" payload out into one call per user and mood, with
// the payload "<mood> <username>". Navidrome logs a returned error.
func (plugin) OnCallback(req scheduler.SchedulerCallbackRequest) error {
	if req.Payload != "moods" {
		mood, user, _ := strings.Cut(req.Payload, " ")
		for _, m := range moods {
			if m[0] == mood {
				if err := refreshMood(user, mood, m[1]); err != nil {
					return fmt.Errorf("%s for %s: %w", m[1], user, err)
				}
			}
		}
		return nil
	}
	users, err := host.UsersGetUsers()
	if err != nil {
		return err
	}
	var errs []error
	var delay int32
	for _, u := range users {
		for _, m := range moods {
			if _, err := host.SchedulerScheduleOneTime(delay, m[0]+" "+u.UserName, ""); err != nil {
				errs = append(errs, err)
			}
			delay += moodSpacing
		}
	}
	return errors.Join(errs...)
}

// refreshMood replaces the tracks of user's playlist for mood, creating it if
// needed. Tracks in libraries the user cannot open, and copies that match the
// same track, drop out in the match. One seed keeps a refresh's pages apart.
func refreshMood(user, mood, name string) error {
	sample := url.Values{"name": {mood}, "min": {strconv.FormatFloat(moodMin, 'f', -1, 64)},
		"seed": {strconv.FormatInt(rand.Int64(), 10)}, "count": {strconv.Itoa(moodPage)}}
	seen := map[string]bool{}
	var ids []string
	for page := 0; page < moodPages && len(ids) < moodSize; page++ {
		sample.Set("page", strconv.Itoa(page))
		res, err := query("/mood", sample)
		if err != nil {
			return err
		}
		if len(res.Matches) == 0 {
			break
		}
		tracks, err := host.MatcherMatchSongs(songs(res), host.MatchOptions{Username: user})
		if err != nil {
			return err
		}
		for _, t := range tracks {
			if t != nil && !seen[t.ID] {
				seen[t.ID] = true
				ids = append(ids, t.ID)
			}
		}
	}
	if len(ids) == 0 {
		return nil
	}
	params := url.Values{"u": {user}, "songId": ids[:min(moodSize, len(ids))]}

	var lists struct {
		Playlists struct {
			Playlist []struct{ ID, Name, Comment, Owner string }
		}
	}
	if err := subsonic("getPlaylists?"+url.Values{"u": {user}}.Encode(), &lists); err != nil {
		return err
	}
	for _, p := range lists.Playlists.Playlist {
		if p.Owner == user && p.Name == name && p.Comment == moodMarker {
			// createPlaylist with a playlistId replaces that playlist's songs.
			params.Set("playlistId", p.ID)
			return subsonic("createPlaylist?"+params.Encode(), nil)
		}
	}
	params.Set("name", name)
	var created struct{ Playlist struct{ ID string } }
	if err := subsonic("createPlaylist?"+params.Encode(), &created); err != nil {
		return err
	}
	// createPlaylist takes no comment, so the marker is set afterwards.
	return subsonic("updatePlaylist?"+url.Values{"u": {user}, "playlistId": {created.Playlist.ID}, "comment": {moodMarker}}.Encode(), nil)
}

// subsonic calls Navidrome's Subsonic API and decodes a successful response into out.
func subsonic(uri string, out any) error {
	body, err := host.SubsonicAPICall(uri)
	if err != nil {
		return err
	}
	var env struct {
		Response json.RawMessage `json:"subsonic-response"`
	}
	var status struct {
		Status string
		Error  struct{ Message string }
	}
	if err := json.Unmarshal([]byte(body), &env); err != nil {
		return err
	}
	if err := json.Unmarshal(env.Response, &status); err != nil {
		return err
	}
	if status.Status != "ok" {
		return fmt.Errorf("%s: %s", strings.SplitN(uri, "?", 2)[0], status.Error.Message)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(env.Response, out)
}
