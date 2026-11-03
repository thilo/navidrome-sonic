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

# sonic-decode, linked against the ffmpeg libraries of the same Alpine release as the
# runtime stage.
FROM alpine:3.24 AS decode
RUN apk add --no-cache build-base ffmpeg-dev
COPY decode.c /src/
RUN cc -O2 -Wall -o /sonic-decode /src/decode.c \
    $(pkg-config --cflags --libs libavformat libavcodec libavfilter libavutil)

FROM alpine:3.24
RUN apk add --no-cache python3 py3-numpy py3-onnxruntime ffmpeg
# EffNet and its mood heads; see NOTICE for their origin and licence.
COPY models/ /opt/sonic/models/
COPY --from=plugin /sonic.ndp /opt/sonic/
COPY --from=decode /sonic-decode /opt/sonic/
COPY sonic.py analysis.py index.py /opt/sonic/
# The self-check runs against the image's libraries but is not part of it.
RUN --mount=type=bind,source=test_sonic.py,target=/tmp/test_sonic.py \
    PYTHONPATH=/opt/sonic PYTHONDONTWRITEBYTECODE=1 python3 /tmp/test_sonic.py
CMD ["sh", "-c", "cp /opt/sonic/sonic.ndp /plugins/ && exec python3 /opt/sonic/sonic.py"]
