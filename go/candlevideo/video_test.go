package candlevideo

import (
	"context"
	"errors"
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

func TestGenerateHonorsCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := Generate(ctx, ".", GenerateOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context canceled, got: %v", err)
	}
}
