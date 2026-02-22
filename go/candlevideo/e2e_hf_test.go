package candlevideo

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
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

	outputDir := os.Getenv("CANDLE_VIDEO_E2E_OUTPUT_DIR")
	if outputDir == "" {
		outputDir = filepath.Join(t.TempDir(), "output")
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("MkdirAll failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Minute)
	defer cancel()

	cpu := true
	if v := os.Getenv("CANDLE_VIDEO_E2E_CPU"); v != "" {
		if parsed, err := strconv.ParseBool(v); err == nil {
			cpu = parsed
		}
	}

	modelID := os.Getenv("CANDLE_VIDEO_E2E_MODEL_ID")
	if modelID == "" {
		modelID = "oxide-lab/LTX-Video-0.9.8-2B-distilled"
	}

	err := Generate(ctx, repoDir, GenerateOptions{
		Prompt:    "A short cinematic shot of ocean waves at sunset",
		ModelID:   modelID,
		OutputDir: outputDir,
		Height:    128,
		Width:     128,
		NumFrames: 9,
		Steps:     1,
		GIF:       true,
		CPU:       cpu,
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
