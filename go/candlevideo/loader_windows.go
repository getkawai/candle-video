//go:build windows

package candlevideo

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

var (
	initialized bool
	dll         *syscall.DLL

	fnGenerate *syscall.Proc
	fnLastErr  *syscall.Proc
	fnVersion  *syscall.Proc
)

func defaultLibPath(repoDir string) string {
	base := filepath.Join(repoDir, "target", "release")
	return filepath.Join(base, "candle_video.dll")
}

func Init(repoDir string) error {
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
	if fnLastErr, err = loadProc("candle_last_error"); err != nil {
		return err
	}
	if fnVersion, err = loadProc("candle_binding_version"); err != nil {
		return err
	}

	initialized = true
	return nil
}

func cStringFromPtr(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	bytes := make([]byte, 0, 128)
	for i := uintptr(0); ; i++ {
		b := *(*byte)(unsafe.Pointer(ptr + i))
		if b == 0 {
			break
		}
		bytes = append(bytes, b)
	}
	return string(bytes)
}

func lastError() string {
	if fnLastErr == nil {
		return "unknown error"
	}
	r1, _, _ := fnLastErr.Call()
	msg := cStringFromPtr(r1)
	if msg == "" {
		return "unknown error"
	}
	return msg
}

func Version() string {
	if !initialized || fnVersion == nil {
		return "unknown"
	}
	r1, _, _ := fnVersion.Call()
	v := cStringFromPtr(r1)
	if v == "" {
		return "unknown"
	}
	return v
}
