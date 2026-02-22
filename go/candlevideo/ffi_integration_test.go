package candlevideo

import (
	"os"
	"strings"
	"sync"
	"testing"
)

func TestFFIContractWithRealLibrary(t *testing.T) {
	if os.Getenv("CANDLE_VIDEO_LIB_PATH") == "" {
		t.Skip("CANDLE_VIDEO_LIB_PATH is not set")
	}

	if err := Init("."); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	if !Healthcheck() {
		t.Fatal("Healthcheck failed")
	}
	if Version() == "unknown" {
		t.Fatal("Version should not be unknown after successful Init")
	}

	err := generateFromJSONPayload([]byte("{invalid-json"))
	if err == nil {
		t.Fatal("expected native JSON parsing error, got nil")
	}
	if !strings.Contains(err.Error(), "invalid JSON config") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestInitConcurrentCalls(t *testing.T) {
	if os.Getenv("CANDLE_VIDEO_LIB_PATH") == "" {
		t.Skip("CANDLE_VIDEO_LIB_PATH is not set")
	}

	const workers = 16
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := Init("."); err != nil {
				errCh <- err
			}
		}()
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		if err != nil {
			t.Fatalf("concurrent Init returned error: %v", err)
		}
	}
}
