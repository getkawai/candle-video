package candlevideo

import (
	"os"
	"testing"
)

func TestRealLibraryHealthcheck(t *testing.T) {
	libPath := os.Getenv("CANDLE_VIDEO_LIB_PATH")
	if libPath == "" {
		t.Skip("CANDLE_VIDEO_LIB_PATH is not set")
	}

	if err := Init("."); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	if !Healthcheck() {
		t.Fatal("Healthcheck failed")
	}
}
