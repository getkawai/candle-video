//go:build darwin || linux

package candlevideo

/*
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include <dlfcn.h>
#include "candle.h"

void* open_lib(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}

void* get_sym(void* handle, const char* name) {
    return dlsym(handle, name);
}

char* get_dlerror() {
    return dlerror();
}
*/
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"
)

var (
	initialized bool
	dlHandle    unsafe.Pointer
	initMu      sync.Mutex

	fnGenerate unsafe.Pointer
	fnHealth   unsafe.Pointer
	fnLastErr  unsafe.Pointer
	fnVersion  unsafe.Pointer
)

func defaultLibPath(repoDir string) string {
	base := filepath.Join(repoDir, "target", "release")
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(base, "libcandle_video.dylib")
	default:
		return filepath.Join(base, "libcandle_video.so")
	}
}

func Init(repoDir string) error {
	initMu.Lock()
	defer initMu.Unlock()

	if initialized {
		return nil
	}

	libPath := os.Getenv("CANDLE_VIDEO_LIB_PATH")
	if override := getLibraryPathOverride(); override != "" {
		libPath = override
	}
	if libPath == "" {
		libPath = defaultLibPath(repoDir)
	}

	if _, err := os.Stat(libPath); err != nil {
		return fmt.Errorf("library not found at %s: %w", libPath, err)
	}

	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.open_lib(cPath)
	if handle == nil {
		return fmt.Errorf("dlopen failed: %s", C.GoString(C.get_dlerror()))
	}
	dlHandle = handle

	load := func(name string) (unsafe.Pointer, error) {
		cName := C.CString(name)
		defer C.free(unsafe.Pointer(cName))
		sym := C.get_sym(dlHandle, cName)
		if sym == nil {
			return nil, fmt.Errorf("symbol not found: %s", name)
		}
		return sym, nil
	}

	var err error
	if fnGenerate, err = load("candle_video_generate"); err != nil {
		return err
	}
	if fnHealth, err = load("candle_healthcheck"); err != nil {
		fnHealth = nil
	}
	if fnLastErr, err = load("candle_last_error"); err != nil {
		return err
	}
	if fnVersion, err = load("candle_binding_version"); err != nil {
		return err
	}

	initialized = true
	return nil
}

func lastError() string {
	if fnLastErr == nil {
		return "unknown error"
	}
	msg := C.call_candle_last_error(fnLastErr)
	if msg == nil {
		return "unknown error"
	}
	return C.GoString(msg)
}

func Version() string {
	if !initialized || fnVersion == nil {
		return "unknown"
	}
	v := C.call_candle_binding_version(fnVersion)
	if v == nil {
		return "unknown"
	}
	return C.GoString(v)
}

func Healthcheck() bool {
	if !initialized || fnHealth == nil {
		return false
	}
	return C.call_candle_healthcheck(fnHealth) == 1
}
