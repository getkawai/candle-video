package candlevideo

import "testing"

func TestSetLibraryPath(t *testing.T) {
	SetLibraryPath("/tmp/libcandle_video.so")
	if got := getLibraryPathOverride(); got != "/tmp/libcandle_video.so" {
		t.Fatalf("unexpected override path: %q", got)
	}

	SetLibraryPath("")
	if got := getLibraryPathOverride(); got != "" {
		t.Fatalf("override should be cleared, got: %q", got)
	}
}

func TestSetModelPath(t *testing.T) {
	SetModelPath("/tmp/models")
	if got := getModelPathOverride(); got != "/tmp/models" {
		t.Fatalf("unexpected model path override: %q", got)
	}

	SetModelPath("")
	if got := getModelPathOverride(); got != "" {
		t.Fatalf("model path override should be cleared, got: %q", got)
	}
}
