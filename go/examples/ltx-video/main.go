package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/getkawai/candle-video/go/candlevideo"
)

func main() {
	repoDir := flag.String("repo", "../..", "Path to candle-video repository")
	prompt := flag.String("prompt", "A cinematic wide shot of ocean waves crashing into dark rocks at sunset", "Text prompt")
	weights := flag.String("weights", "", "Path to local model directory (required for offline use)")
	modelPath := flag.String("model-path", "", "Path to local model directory (alias of --weights)")
	unified := flag.String("unified", "", "Path to unified safetensors file (optional)")
	output := flag.String("output", "output", "Output directory")
	libPath := flag.String("lib-path", "", "Path to shared library; sets CANDLE_VIDEO_LIB_PATH")
	cpu := flag.Bool("cpu", false, "Run on CPU")
	flag.Parse()

	if *libPath != "" {
		if err := os.Setenv("CANDLE_VIDEO_LIB_PATH", *libPath); err != nil {
			log.Fatalf("failed setting CANDLE_VIDEO_LIB_PATH: %v", err)
		}
	}
	if *modelPath != "" {
		candlevideo.SetModelPath(*modelPath)
	}

	err := candlevideo.Generate(context.Background(), *repoDir, candlevideo.GenerateOptions{
		Prompt:         *prompt,
		LocalWeights:   *weights,
		UnifiedWeights: *unified,
		OutputDir:      *output,
		GIF:            true,
		CPU:            *cpu,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("done, output in %s\n", *output)
}
