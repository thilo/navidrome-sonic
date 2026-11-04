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

# The sidecar and sonic-decode, linked against the libraries of the same Alpine
# release as the runtime stage. The self-check runs against those libraries but
# is not part of the image.
FROM alpine:3.24 AS sidecar
RUN apk add --no-cache build-base ffmpeg ffmpeg-dev go
WORKDIR /src
COPY sidecar/go.mod sidecar/go.sum ./
RUN go mod download
COPY sidecar/*.go ./
RUN --mount=type=bind,source=decode.c,target=/tmp/decode.c \
    cc -O2 -Wall -o sonic-decode /tmp/decode.c \
    $(pkg-config --cflags --libs libavformat libavcodec libavfilter libavutil) \
 && go build -trimpath -o sonic . \
 && go test .

FROM alpine:3.24
RUN apk add --no-cache ffmpeg onnxruntime
# EffNet and its mood heads; see NOTICE for their origin and licence.
COPY models/ /opt/sonic/models/
COPY --from=plugin /sonic.ndp /opt/sonic/
COPY --from=sidecar /src/sonic /src/sonic-decode /opt/sonic/
CMD ["sh", "-c", "cp /opt/sonic/sonic.ndp /plugins/ && exec /opt/sonic/sonic"]
