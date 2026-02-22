package candlevideo

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strconv"
)

// GenerateOptions maps to candle-video ltx-video example flags.
type GenerateOptions struct {
	RepoDir        string
	Prompt         string
	NegativePrompt string
	LocalWeights   string
	UnifiedWeights string
	OutputDir      string
	LTXVersion     string
	Height         int
	Width          int
	NumFrames      int
	Steps          int
	GuidanceScale  float64
	CPU            bool
	Seed           uint64
	GIF            bool
	Frames         bool
	VAETiling      bool
	VAESlicing     bool
	UseBF16T5      bool
}

func (o *GenerateOptions) normalize() {
	if o.RepoDir == "" {
		o.RepoDir = "."
	}
	if o.Prompt == "" {
		o.Prompt = "A video of a cute cat playing with a yarn ball"
	}
	if o.LTXVersion == "" {
		o.LTXVersion = "0.9.8-2b-distilled"
	}
	if o.OutputDir == "" {
		o.OutputDir = "output"
	}
	if o.Height == 0 {
		o.Height = 512
	}
	if o.Width == 0 {
		o.Width = 768
	}
	if o.NumFrames == 0 {
		o.NumFrames = 97
	}
	if o.Steps == 0 {
		o.Steps = 7
	}
	if o.GuidanceScale == 0 {
		o.GuidanceScale = 1.0
	}
}

// Generate runs the Rust example binary via `cargo run --example ltx-video`.
func Generate(ctx context.Context, opts GenerateOptions) error {
	opts.normalize()

	if opts.Height%32 != 0 || opts.Width%32 != 0 {
		return fmt.Errorf("height and width must be divisible by 32")
	}

	repoDir, err := filepath.Abs(opts.RepoDir)
	if err != nil {
		return fmt.Errorf("resolve repo dir: %w", err)
	}

	args := []string{"run", "--example", "ltx-video", "--release", "--", "--prompt", opts.Prompt,
		"--negative-prompt", opts.NegativePrompt,
		"--ltxv-version", opts.LTXVersion,
		"--height", strconv.Itoa(opts.Height),
		"--width", strconv.Itoa(opts.Width),
		"--num-frames", strconv.Itoa(opts.NumFrames),
		"--steps", strconv.Itoa(opts.Steps),
		"--guidance-scale", strconv.FormatFloat(opts.GuidanceScale, 'f', -1, 64),
		"--output-dir", opts.OutputDir,
	}

	if opts.LocalWeights != "" {
		args = append(args, "--local-weights", opts.LocalWeights)
	}
	if opts.UnifiedWeights != "" {
		args = append(args, "--unified-weights", opts.UnifiedWeights)
	}
	if opts.CPU {
		args = append(args, "--cpu")
	}
	if opts.Seed != 0 {
		args = append(args, "--seed", strconv.FormatUint(opts.Seed, 10))
	}
	if opts.GIF {
		args = append(args, "--gif")
	}
	if opts.Frames {
		args = append(args, "--frames")
	}
	if opts.VAETiling {
		args = append(args, "--vae-tiling")
	}
	if opts.VAESlicing {
		args = append(args, "--vae-slicing")
	}
	if opts.UseBF16T5 {
		args = append(args, "--use-bf16-t5")
	}

	cmd := exec.CommandContext(ctx, "cargo", args...)
	cmd.Dir = repoDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("candle-video failed: %w\n%s", err, string(out))
	}
	return nil
}
