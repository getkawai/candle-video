//go:build windows

package candlevideo

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
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
	if !initialized || fnLastErr == nil {
		return "unknown error"
	}
	addr, _, _ := fnLastErr.Call()
	return cStringFromPtr(addr)
}

func Version() string {
	if !initialized || fnVersion == nil {
		return "unknown"
	}
	addr, _, _ := fnVersion.Call()
	v := cStringFromPtr(addr)
	if v == "" {
		return "unknown"
	}
	return v
}

func Healthcheck() bool {
	if !initialized || fnHealth == nil {
		return false
	}
	r1, _, _ := fnHealth.Call()
	return int32(r1) == 1
}

func cStringFromPtr(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	const maxLen = 4096
	buf := make([]byte, 0, 64)
	for i := 0; i < maxLen; i++ {
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(i)))
		if b == 0 {
			return string(buf)
		}
		buf = append(buf, b)
	}
	return string(buf)
}
