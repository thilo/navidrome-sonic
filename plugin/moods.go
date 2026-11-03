// Mood playlists. Every Monday, and once 5 minutes after the plugin loads, each
// user the plugin is granted gets six playlists ("Happy Mix" and so on): 30
// tracks drawn at random from the 150 the sidecar scores highest for that mood
// among the tracks in the libraries that user can open. A playlist is found again
// by owner, name and the comment marker, and its tracks are replaced; renaming or
// editing the comment detaches it.
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
	moodCron   = "0 4 * * 1"
	moodPool   = 150
	moodSize   = 30
	moodMarker = "navidrome-sonic"
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

// refreshMood replaces the tracks of user's playlist for mood, creating it if needed.
func refreshMood(user, mood, name string) error {
	res, err := query("/mood", url.Values{"name": {mood}, "count": {strconv.Itoa(2 * moodPool)}})
	if err != nil {
		return err
	}
	tracks, err := host.MatcherMatchSongs(songs(res), host.MatchOptions{Username: user})
	if err != nil {
		return err
	}
	seen := map[string]bool{}
	var ids []string
	for _, t := range tracks {
		if t == nil || seen[t.ID] {
			continue
		}
		seen[t.ID] = true
		if ids = append(ids, t.ID); len(ids) == moodPool {
			break
		}
	}
	if len(ids) == 0 {
		return nil
	}
	rand.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
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
			params.Set("playlistId", p.ID)
			return subsonic("createPlaylist?"+params.Encode(), nil)
		}
	}
	params.Set("name", name)
	var created struct{ Playlist struct{ ID string } }
	if err := subsonic("createPlaylist?"+params.Encode(), &created); err != nil {
		return err
	}
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
