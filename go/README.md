# candle-video Go wrapper

This directory provides a small Go API for running the Rust LTX-Video pipeline from Go.

It wraps the existing Rust example command:

```bash
cargo run --example ltx-video --release -- ...
```

## Requirements

- Go 1.24+
- Rust toolchain + cargo
- Model weights (local path recommended)

## Quick start

```bash
cd go
go run ./examples/ltx-video \
  --repo ../ \
  --weights /path/to/models/ltx-video \
  --output ./output
```

## Programmatic usage

```go
err := candlevideo.Generate(context.Background(), candlevideo.GenerateOptions{
    RepoDir:      "/path/to/candle-video",
    Prompt:       "A cinematic drone shot of a waterfall in Iceland",
    LocalWeights: "/path/to/models/ltx-video",
    OutputDir:    "./output",
    GIF:          true,
})
```

## Notes

- `Height` and `Width` must be divisible by 32.
- This wrapper executes `cargo` as a subprocess and forwards CLI args.
- For production, you can replace this with direct FFI/cgo binding later.
