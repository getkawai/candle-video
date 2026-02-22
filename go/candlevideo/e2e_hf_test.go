package candlevideo

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGenerateE2EFromHuggingFace(t *testing.T) {
	if os.Getenv("CANDLE_VIDEO_RUN_HF_E2E") != "1" {
		t.Skip("set CANDLE_VIDEO_RUN_HF_E2E=1 to enable")
	}

	repoDir := os.Getenv("CANDLE_VIDEO_REPO_DIR")
	if repoDir == "" {
		t.Fatal("CANDLE_VIDEO_REPO_DIR is required")
	}

	outputDir := filepath.Join(t.TempDir(), "output")
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
	defer cancel()

	err := Generate(ctx, repoDir, GenerateOptions{
		Prompt:    "A short cinematic shot of ocean waves at sunset",
		OutputDir: outputDir,
		Height:    256,
		Width:     256,
		NumFrames: 17,
		Steps:     1,
		GIF:       true,
		CPU:       false,
	})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	entries, err := os.ReadDir(outputDir)
	if err != nil {
		t.Fatalf("ReadDir failed: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("no output files generated")
	}
}
