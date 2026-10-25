# The sidecar and, alongside it, the Navidrome plugin that queries it. The
# container copies sonic.ndp into /plugins, Navidrome's plugin folder, on start.

# Navidrome disables a plugin whose file hash changed until an admin enables it
# again, so the package is built byte for byte reproducibly: a pinned toolchain,
# -trimpath, fixed mtimes and no extra zip fields.
FROM golang:1.25.14-alpine AS plugin
RUN apk add --no-cache zip
WORKDIR /src
COPY plugin/ .
RUN GOOS=wasip1 GOARCH=wasm go build -trimpath -buildmode=c-shared -o plugin.wasm . \
 && touch -t 198001010000 manifest.json plugin.wasm \
 && zip -X -j /sonic.ndp manifest.json plugin.wasm

# Alpine 3.24 packages the same onnxruntime release the LMS image runs the model with.
FROM alpine:3.24
RUN apk add --no-cache python3 py3-numpy py3-onnxruntime ffmpeg
# MusicNN embedding model; see NOTICE for its origin.
COPY MSD_musicnn_embedding.onnx /opt/sonic/
COPY --from=plugin /sonic.ndp /opt/sonic/
COPY sonic.py /opt/sonic/
RUN python3 /opt/sonic/sonic.py test
CMD ["sh", "-c", "cp /opt/sonic/sonic.ndp /plugins/ && exec python3 /opt/sonic/sonic.py"]
