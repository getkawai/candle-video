//go:build darwin || linux

package candlevideo

/*
#include <stdlib.h>
#include "candle.h"
*/
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"
)

// Generate executes native Rust pipeline via shared-library FFI.
func Generate(ctx context.Context, repoDir string, opts GenerateOptions) error {
	_ = ctx

	opts.normalize()

	if opts.Height%32 != 0 || opts.Width%32 != 0 {
		return fmt.Errorf("height and width must be divisible by 32")
	}

	if err := Init(repoDir); err != nil {
		return err
	}

	payload, err := json.Marshal(opts)
	if err != nil {
		return fmt.Errorf("marshal options: %w", err)
	}

	return generateFromJSONPayload(payload)
}

func generateFromJSONPayload(payload []byte) error {
	cCfg := C.CString(string(payload))
	defer C.free(unsafe.Pointer(cCfg))

	ret := C.call_candle_video_generate(fnGenerate, cCfg)
	if ret != 0 {
		return fmt.Errorf("video generation failed: %s", lastError())
	}

	return nil
}
