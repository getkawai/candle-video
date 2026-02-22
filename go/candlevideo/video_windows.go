//go:build windows

package candlevideo

import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"
)

// Generate executes native Rust pipeline via shared-library FFI.
func Generate(ctx context.Context, repoDir string, opts GenerateOptions) error {
	if err := ctx.Err(); err != nil {
		return err
	}

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

	if err := ctx.Err(); err != nil {
		return err
	}

	return generateFromJSONPayload(payload)
}

func generateFromJSONPayload(payload []byte) error {
	buf := append(payload, 0)
	ret, _, _ := fnGenerate.Call(uintptr(unsafe.Pointer(&buf[0])))
	if int32(ret) != 0 {
		return fmt.Errorf("video generation failed: %s", lastError())
	}

	return nil
}
