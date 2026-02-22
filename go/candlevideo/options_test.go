package candlevideo

import "testing"

func TestGenerateOptionsNormalizeDefaults(t *testing.T) {
	opts := GenerateOptions{}
	opts.normalize()

	if opts.Prompt == "" {
		t.Fatal("Prompt should be defaulted")
	}
	if opts.LTXVersion != "0.9.8-2b-distilled" {
		t.Fatalf("unexpected default LTXVersion: %q", opts.LTXVersion)
	}
	if opts.ModelID != "oxide-lab/LTX-Video-0.9.8-2B-distilled" {
		t.Fatalf("unexpected default ModelID: %q", opts.ModelID)
	}
	if opts.OutputDir != "output" {
		t.Fatalf("unexpected default OutputDir: %q", opts.OutputDir)
	}
	if opts.Height != 512 || opts.Width != 768 {
		t.Fatalf("unexpected default resolution: %dx%d", opts.Width, opts.Height)
	}
	if opts.NumFrames != 97 {
		t.Fatalf("unexpected default NumFrames: %d", opts.NumFrames)
	}
	if opts.Steps != 7 {
		t.Fatalf("unexpected default Steps: %d", opts.Steps)
	}
	if !opts.GIF {
		t.Fatal("GIF should be enabled by default when GIF and Frames are both false")
	}
}

func TestGenerateOptionsNormalizePreservesProvidedValues(t *testing.T) {
	opts := GenerateOptions{
		Prompt:     "hello",
		LTXVersion: "custom",
		ModelID:    "model/x",
		OutputDir:  "out",
		Height:     640,
		Width:      960,
		NumFrames:  33,
		Steps:      12,
		Frames:     true,
	}
	opts.normalize()

	if opts.Prompt != "hello" {
		t.Fatalf("Prompt changed unexpectedly: %q", opts.Prompt)
	}
	if opts.LTXVersion != "custom" {
		t.Fatalf("LTXVersion changed unexpectedly: %q", opts.LTXVersion)
	}
	if opts.ModelID != "model/x" {
		t.Fatalf("ModelID changed unexpectedly: %q", opts.ModelID)
	}
	if opts.OutputDir != "out" {
		t.Fatalf("OutputDir changed unexpectedly: %q", opts.OutputDir)
	}
	if opts.Height != 640 || opts.Width != 960 {
		t.Fatalf("resolution changed unexpectedly: %dx%d", opts.Width, opts.Height)
	}
	if opts.NumFrames != 33 {
		t.Fatalf("NumFrames changed unexpectedly: %d", opts.NumFrames)
	}
	if opts.Steps != 12 {
		t.Fatalf("Steps changed unexpectedly: %d", opts.Steps)
	}
	if opts.GIF {
		t.Fatal("GIF should stay false when Frames is true")
	}
}
