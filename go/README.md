# candle-video Go binding

Native Go binding for `candle-video` using Rust FFI (`cdylib`) + `dlopen/dlsym`.

## Download sources (jelas)

### Native library (`libcandle_video`)

Download dari GitHub Releases:
- Release page: `https://github.com/getkawai/candle-video/releases`
- Latest release saat ini: `https://github.com/getkawai/candle-video/releases/tag/auto-13-5b2541a51fd2585b85b84b8e2527abd91143f8a8`

Direct download:
- Linux AMD64: `https://github.com/getkawai/candle-video/releases/download/auto-13-5b2541a51fd2585b85b84b8e2527abd91143f8a8/libcandle_video-linux-amd64.tar.gz`
- macOS ARM64: `https://github.com/getkawai/candle-video/releases/download/auto-13-5b2541a51fd2585b85b84b8e2527abd91143f8a8/libcandle_video-darwin-arm64.tar.gz`
- Windows AMD64: `https://github.com/getkawai/candle-video/releases/download/auto-13-5b2541a51fd2585b85b84b8e2527abd91143f8a8/libcandle_video-windows-amd64.tar.gz`

Contoh Linux:

```bash
mkdir -p /opt/candle-video/lib
curl -fL "https://github.com/getkawai/candle-video/releases/download/auto-13-5b2541a51fd2585b85b84b8e2527abd91143f8a8/libcandle_video-linux-amd64.tar.gz" -o /tmp/libcandle_video-linux-amd64.tar.gz
tar -xzf /tmp/libcandle_video-linux-amd64.tar.gz -C /opt/candle-video/lib
```

### Model weights

Download dari Hugging Face model repo:
- `https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled`

Contoh clone semua file model:

```bash
git lfs install
git clone https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled /opt/candle-video/models/LTX-Video-0.9.8-2B-distilled
```

## Build Rust shared library

Run from repository root:

```bash
cargo build --release
```

Library output:
- macOS: `target/release/libcandle_video.dylib`
- Linux: `target/release/libcandle_video.so`
- Windows: `target/release/candle_video.dll`

Prebuilt binaries are also published in GitHub Releases:
- `libcandle_video-darwin-arm64.tar.gz`
- `libcandle_video-linux-amd64.tar.gz`
- `libcandle_video-windows-amd64.tar.gz`

Production releases are published from Git tags (`v*`) via GitHub Actions.

## Library path resolution

Path library native diselesaikan dengan urutan prioritas berikut:
1. `candlevideo.SetLibraryPath(...)`
2. environment variable `CANDLE_VIDEO_LIB_PATH`
3. default path dari `repoDir` (`target/release/...`)

Path model lokal diselesaikan dengan urutan prioritas berikut:
1. `GenerateOptions.LocalWeights` (explicit per call)
2. `candlevideo.SetModelPath(...)`
3. fallback ke `ModelID` (download/resolve model)

Override via environment variable:

```bash
export CANDLE_VIDEO_LIB_PATH=/absolute/path/to/libcandle_video.so
```

CI production gates:
- ABI symbol verification on Linux.
- Go test + vet + race (Linux) and test + vet (Windows).
- `govulncheck` on Go module.
- Consumer smoke test against latest released `libcandle_video-linux-amd64.tar.gz`.
- Hugging Face download smoke (`model_index.json` dari `oxide-lab/LTX-Video-0.9.8-2B-distilled`).
- Optional real E2E generate from Hugging Face via workflow dispatch (`run_hf_e2e=true`) on CPU runner (`ubuntu-latest`).
  Hasil video E2E di-upload sebagai artifact `hf-e2e-generated-video` pada run Actions.

## Run Go example

```bash
cd go
go run ./examples/ltx-video \
  --repo ../ \
  --lib-path ../target/release/libcandle_video.so \
  --model-path /path/to/models/ltx-video \
  --output ./output
```

`--lib-path` akan mengisi `CANDLE_VIDEO_LIB_PATH` dari input CLI.
`--model-path` akan mengisi override model path (`candlevideo.SetModelPath`).

## High-end consumer computer (quality profile)

Untuk PC/laptop high-end (contoh RTX 4080/4090, RAM 32GB+, NVMe), gunakan GPU (`CPU=false`) dan parameter lebih tinggi:

- Resolusi: `768x768` sampai `1024x576`
- Frames: `49` sampai `97`
- Steps: `12` sampai `24`
- Simpan model lokal (`LocalWeights`) agar tidak download ulang

Contoh kode:

```go
candlevideo.SetLibraryPath("/absolute/path/to/libcandle_video.so")
candlevideo.SetModelPath("/absolute/path/to/models/ltx-video")

err := candlevideo.Generate(context.Background(), "/path/to/candle-video", candlevideo.GenerateOptions{
    Prompt:       "A cinematic aerial shot of a futuristic city at golden hour",
    OutputDir:    "./output-high",
    Height:       768,
    Width:        768,
    NumFrames:    97,
    Steps:        16,
    GuidanceScale: 3.0,
    GIF:          true,
    CPU:          false,
})
```

Catatan:
- Jika VRAM tidak cukup, turunkan urutan ini: `NumFrames` -> `Height/Width` -> `Steps`.
- Untuk output terbaik, jalankan job ini di mesin lokal/GPU dedicated, bukan CPU-only CI.

## Programmatic usage

```go
import "github.com/getkawai/candle-video/go/candlevideo"

candlevideo.SetLibraryPath("/absolute/path/to/libcandle_video.so")
candlevideo.SetModelPath("/absolute/path/to/models/ltx-video")

err := candlevideo.Generate(context.Background(), "/path/to/candle-video", candlevideo.GenerateOptions{
    Prompt:       "A cinematic drone shot of a waterfall in Iceland",
    LocalWeights: "/path/to/models/ltx-video",
    OutputDir:    "./output",
    GIF:          true,
})
```
