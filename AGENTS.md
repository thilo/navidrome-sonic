# AGENTS.md

Context for anyone — human or agent — changing this repo, limited to what the
files themselves don't tell you. How the plugin and sidecar fit together is in
the [README](README.md); the reasons behind single settings sit in comments
beside them.

## Scope

Built for small boxes: one thread, no database server, a single image. Changes
that need more memory, more cores or another service have to justify it. No
speculative configuration: no options, env vars or endpoints without a caller.

## Constraints that aren't visible in the code

- **The plugin hash is part of the interface.** Navidrome disables a plugin
  whose `sonic.ndp` hash changed until an admin enables it again. Any change
  under `plugin/` (code, `go.mod`, manifest) forces that on every install;
  changes elsewhere must leave the hash alone. Check with two builds of the
  `plugin` stage, one with `--no-cache`, and compare
  `sha256sum /sonic.ndp`.
- **The front end follows Essentia; patch selection and ranking follow LMS.**
  The front end in `sidecar/analysis.go` reproduces Essentia's `TensorflowInputMusiCNN`,
  which EffNet and its heads were trained on (`sonic_test.go` checks it against
  values from Essentia); `layout()` and the patch and ranking constants
  reproduce LMS's. Don't tune them; a deliberate departure gets a comment saying
  how and why it differs (see the whitening shrinkage in `index.go`).
- **Navidrome does the per-user filtering.** The plugin is not told who asks;
  Navidrome matches the returned tags within the caller's libraries. Don't
  filter by user or library in the plugin or the sidecar.
- **The models are vendored.** `models/` holds plain files in git; their
  origin, licence and checksums are in `NOTICE`, and `convert-effnet.sh`
  rebuilds `discogs-effnet.onnx` from Essentia's TensorFlow graph. A change to
  a model or the analysis bumps `analysisVersion` in `sonic.go`, which makes
  the sidecar drop and rebuild its database.

## References

The Navidrome version the PDK in `plugin/go.mod` was cut from is the
authority: its `plugins/README.md` (permissions, host services) and the PDK
source. The user docs at navidrome.org describe less than that README.

## Testing

- `go test` in `sidecar/` runs the self-check; it needs ffmpeg and
  `sonic-decode` in that directory (built from `decode.c` against ffmpeg 8's
  libraries, as in the Dockerfile), and the Docker build runs it too. It
  compares `sonic-decode` byte for byte with the ffmpeg CLI, so an ffmpeg
  update that changes its output fails the build.
- There is no Go toolchain requirement on the host; the plugin builds in the
  Dockerfile's `plugin` stage, the sidecar in its `sidecar` stage.
- End to end means a Navidrome with the plugin enabled, calling
  `getSonicSimilarTracks`, `getSimilarSongs2` and `findSonicPath`; for
  library-scoping changes, with two users who can open different libraries.

## Committed text (comments, README, commit messages)

- Comments are factual and terse: what a setting does and why. No grading
  adjectives, no selling, no decision history.
- One place per rationale; point at it instead of repeating it.
- Nothing unique to one deployment: no hostnames, addresses, dates or measured
  sizes. A measurement that justifies a setting goes in one line beside it.
- Commit subject: one imperative line, no trailing period. Body: one paragraph
  of three or four lines on what was wrong and why the fix works.
- No `Co-Authored-By` trailer for an AI assistant unless asked.
- Before committing, scan the whole diff for grading adjectives, dates,
  measured numbers, host facts and duplicated rationale.
