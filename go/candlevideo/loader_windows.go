//go:build windows

package candlevideo

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
)

var (
	initialized bool
	dll         *syscall.DLL
	initMu      sync.Mutex

	fnGenerate *syscall.Proc
	fnHealth   *syscall.Proc
	fnLastErr  *syscall.Proc
	fnVersion  *syscall.Proc
)

func defaultLibPath(repoDir string) string {
	base := filepath.Join(repoDir, "target", "release")
	return filepath.Join(base, "candle_video.dll")
}

func Init(repoDir string) error {
	initMu.Lock()
	defer initMu.Unlock()

	if initialized {
		return nil
	}

	libPath := os.Getenv("CANDLE_VIDEO_LIB_PATH")
	if libPath == "" {
		libPath = defaultLibPath(repoDir)
	}

	if _, err := os.Stat(libPath); err != nil {
		return fmt.Errorf("library not found at %s: %w", libPath, err)
	}

	loaded, err := syscall.LoadDLL(libPath)
	if err != nil {
		return fmt.Errorf("LoadDLL failed: %w", err)
	}
	dll = loaded

	loadProc := func(name string) (*syscall.Proc, error) {
		proc, findErr := dll.FindProc(name)
		if findErr != nil {
			return nil, fmt.Errorf("symbol not found: %s: %w", name, findErr)
		}
		return proc, nil
	}

	if fnGenerate, err = loadProc("candle_video_generate"); err != nil {
		return err
	}
	if fnHealth, err = loadProc("candle_healthcheck"); err != nil {
		fnHealth = nil
	}
	if fnLastErr, err = loadProc("candle_last_error"); err != nil {
		return err
	}
	if fnVersion, err = loadProc("candle_binding_version"); err != nil {
		return err
	}

	initialized = true
	return nil
}

func lastError() string {
	return "unknown error (windows loader fallback)"
}

func Version() string {
	return "unknown"
}

func Healthcheck() bool {
	if !initialized || fnHealth == nil {
		return false
	}
	r1, _, _ := fnHealth.Call()
	return int32(r1) == 1
}
