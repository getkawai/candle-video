# candle-video Go binding

Native Go binding for `candle-video` using Rust FFI (`cdylib`) + `dlopen/dlsym`.

## Build Rust shared library

Run from repository root:

```bash
cargo build --release
```

Library output:
- macOS: `target/release/libcandle_video.dylib`
- Linux: `target/release/libcandle_video.so`

Prebuilt binaries are also published in GitHub Releases:
- `libcandle_video.dylib-darwin-arm64`
- `libcandle_video.so-linux-amd64`

You can override path with env var:

```bash
export CANDLE_VIDEO_LIB_PATH=/absolute/path/to/libcandle_video.dylib
```

## Run Go example

```bash
cd go
go run ./examples/ltx-video \
  --repo ../ \
  --weights /path/to/models/ltx-video \
  --output ./output
```

## Programmatic usage

```go
err := candlevideo.Generate(context.Background(), "/path/to/candle-video", candlevideo.GenerateOptions{
    Prompt:       "A cinematic drone shot of a waterfall in Iceland",
    LocalWeights: "/path/to/models/ltx-video",
    OutputDir:    "./output",
    GIF:          true,
})
```
