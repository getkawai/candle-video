package candlevideo

import (
	"context"
	"strings"
	"testing"
)

func TestGenerateRejectsInvalidDimensions(t *testing.T) {
	err := Generate(context.Background(), ".", GenerateOptions{
		Height: 513,
		Width:  768,
	})
	if err == nil {
		t.Fatal("expected error for non-divisible dimensions, got nil")
	}
	if !strings.Contains(err.Error(), "divisible by 32") {
		t.Fatalf("expected divisibility error, got: %v", err)
	}
}

func TestVersionBeforeInit(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("Version should return a stable fallback when not initialized")
	}
}
