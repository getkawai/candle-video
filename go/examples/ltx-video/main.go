package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/getkawai/candle-video/go/candlevideo"
)

func main() {
	repoDir := flag.String("repo", "../..", "Path to candle-video repository")
	prompt := flag.String("prompt", "A cinematic wide shot of ocean waves crashing into dark rocks at sunset", "Text prompt")
	weights := flag.String("weights", "", "Path to local model directory (required for offline use)")
	unified := flag.String("unified", "", "Path to unified safetensors file (optional)")
	output := flag.String("output", "output", "Output directory")
	cpu := flag.Bool("cpu", false, "Run on CPU")
	flag.Parse()

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
