# navidrome-sonic

Audio-similarity recommendations for [Navidrome](https://www.navidrome.org/):
Instant Mix in the web UI, `getSimilarSongs2`, and the OpenSubsonic
`getSonicSimilarTracks` / `findSonicPath` calls, ranked by what the tracks sound
like.

## Features

- Similar tracks and Instant Mix from the audio itself, not from tags or play
  history.
- Independent of file paths: the plugin needs no filesystem access, and
  Navidrome and the sidecar may mount the library at different paths (see
  [Track lookup](#track-lookup)).
- Each file's audio is analysed once. A moved, renamed or retagged file keeps
  its embedding, and the same file in several libraries is analysed once and
  offered once.
- Results per user: each user gets tracks from the libraries they can open,
  shared libraries included (see
  [Several libraries and users](#several-libraries-and-users)).
- Runs on one CPU thread next to Navidrome, without a database server or GPU.

## How it works

Two parts, built into one image:

- `sonic.py`, a sidecar that walks the music library once an hour, computes a
  MusicNN embedding for every file with new audio (the model and analysis
  [LMS](https://github.com/epoupon/lms) uses) and answers nearest-neighbour
  queries over HTTP. Embeddings live in SQLite under `/data`, each with a hash
  of the audio stream, which paths and tags don't affect. Runs on one thread
  with Python, numpy, onnxruntime and ffmpeg.
- `plugin/`, a Navidrome plugin (`sonic.ndp`) that implements the
  SonicSimilarity capability and the `GetSimilarSongsByTrack` metadata agent by
  asking the sidecar.

On start the container copies `sonic.ndp` into `/plugins`, which Navidrome
mounts as its plugin folder.

## Track lookup

Navidrome and the sidecar know tracks by different keys. Navidrome's track IDs
are random, assigned when a library is first scanned, so the sidecar cannot
derive them; the sidecar knows files.

For a seed track the plugin asks Navidrome's matcher for the track by ID. The
answer carries the file size, duration and title, and the plugin sends those to
the sidecar, which looks the track up by size, takes the closest duration among
files of equal size and then an exact title. Answers go back as tags (title,
artists, album, MBIDs, duration), which Navidrome matches to its own tracks.

The manifest requests `matcher` for that lookup, `library` because Navidrome
requires it alongside `matcher`, and `http` for the sidecar. It does not request
`library.filesystem`: the lookup works without it, and with it Navidrome would
also give the plugin track paths and read access to every library file.

## Compose

```yaml
services:
  navidrome:
    image: deluan/navidrome:0.64.2
    environment:
      ND_PLUGINS_ENABLED: "true"
      ND_PLUGINS_FOLDER: /plugins
      # First agent: answers Instant Mix; the others take over for tracks it
      # has no answer for.
      ND_AGENTS: sonic,deezer,lastfm,listenbrainz
    volumes:
      - music:/music:ro
      - navidrome_data:/data
      - navidrome_plugins:/plugins:ro
    networks: [default, sonic]
    depends_on:
      navidrome-sonic:
        condition: service_healthy

  navidrome-sonic:
    build: https://github.com/thilo/navidrome-sonic.git#v0.1.0
    init: true
    # Must be able to read the music files.
    user: "1000:1000"
    volumes:
      - music:/music:ro
      - navidrome_sonic_data:/data
      - navidrome_plugins:/plugins
    networks: [sonic]
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"]
      interval: 30s

networks:
  # The sidecar's API has no authentication.
  sonic:
    internal: true

volumes:
  music:
  navidrome_data:
  navidrome_sonic_data:
  navidrome_plugins:
```

The plugin reaches the sidecar at `http://navidrome-sonic:8000`, so the service
keeps that name, on a private Docker network.

## Several libraries and users

Mount every Navidrome library somewhere below the sidecar's `/music`; it
analyses whatever it finds there as one collection.

Results are per user. The sidecar answers from all libraries, and Navidrome
matches the answer within the libraries the requesting user can open: a track
in someone else's library drops out, a track in a shared library stays. For
Instant Mix the plugin asks for three times the requested count, so dropped
tracks leave room for shared ones; `getSonicSimilarTracks` and `findSonicPath`
return at most the requested count before that filtering.

## Enabling the plugin

Navidrome loads a new plugin disabled. Enable it once after the first start,
and again whenever `sonic.ndp` changes, which Navidrome detects by its hash.
The plugin looks up the seed track in the libraries it is granted, so grant it
all of them:

```bash
docker compose exec navidrome navidrome plugin edit sonic --all-libraries
docker compose exec navidrome navidrome plugin enable sonic
docker compose restart navidrome
```

The package is built reproducibly, so rebuilding the image without plugin
changes keeps it enabled.

Until a track is analysed, Instant Mix for it falls back to the agents after
`sonic` in `ND_AGENTS`.

## Test

```bash
python3 sonic.py test
```

The Docker build runs it too.

## License

GPL-3.0 (see [LICENSE](LICENSE)); the plugin links Navidrome's GPL-3.0 PDK.
The bundled model file has its own origin, see [NOTICE](NOTICE).
