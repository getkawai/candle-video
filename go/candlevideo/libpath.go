package candlevideo

import "sync"

var (
	libPathMu       sync.RWMutex
	libPathOverride string

	modelPathMu       sync.RWMutex
	modelPathOverride string
)

// SetLibraryPath overrides dynamic library location used by Init.
// Empty value clears the override and falls back to env/default path resolution.
func SetLibraryPath(path string) {
	libPathMu.Lock()
	defer libPathMu.Unlock()
	libPathOverride = path
}

func getLibraryPathOverride() string {
	libPathMu.RLock()
	defer libPathMu.RUnlock()
	return libPathOverride
}

// SetModelPath overrides default local model directory used by Generate.
// It is applied only when GenerateOptions.LocalWeights is empty.
// Empty value clears the override.
func SetModelPath(path string) {
	modelPathMu.Lock()
	defer modelPathMu.Unlock()
	modelPathOverride = path
}

func getModelPathOverride() string {
	modelPathMu.RLock()
	defer modelPathMu.RUnlock()
	return modelPathOverride
}
