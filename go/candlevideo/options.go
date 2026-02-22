package candlevideo

// GenerateOptions maps to Rust FFI JSON config.
type GenerateOptions struct {
	Prompt         string  `json:"prompt"`
	NegativePrompt string  `json:"negative_prompt,omitempty"`
	LTXVersion     string  `json:"ltxv_version,omitempty"`
	ModelID        string  `json:"model_id,omitempty"`
	LocalWeights   string  `json:"local_weights,omitempty"`
	UnifiedWeights string  `json:"unified_weights,omitempty"`
	OutputDir      string  `json:"output_dir,omitempty"`
	Height         int     `json:"height,omitempty"`
	Width          int     `json:"width,omitempty"`
	NumFrames      int     `json:"num_frames,omitempty"`
	Steps          int     `json:"steps,omitempty"`
	GuidanceScale  float32 `json:"guidance_scale,omitempty"`
	CPU            bool    `json:"cpu,omitempty"`
	Seed           uint64  `json:"seed,omitempty"`
	GIF            bool    `json:"gif,omitempty"`
	Frames         bool    `json:"frames,omitempty"`
	VAETiling      bool    `json:"vae_tiling,omitempty"`
	VAESlicing     bool    `json:"vae_slicing,omitempty"`
	UseBF16T5      bool    `json:"use_bf16_t5,omitempty"`
}

func (o *GenerateOptions) normalize() {
	if o.LocalWeights == "" {
		if override := getModelPathOverride(); override != "" {
			o.LocalWeights = override
		}
	}

	if o.Prompt == "" {
		o.Prompt = "A video of a cute cat playing with a yarn ball"
	}
	if o.LTXVersion == "" {
		o.LTXVersion = "0.9.8-2b-distilled"
	}
	if o.ModelID == "" {
		o.ModelID = "oxide-lab/LTX-Video-0.9.8-2B-distilled"
	}
	if o.OutputDir == "" {
		o.OutputDir = "output"
	}
	if o.Height == 0 {
		o.Height = 512
	}
	if o.Width == 0 {
		o.Width = 768
	}
	if o.NumFrames == 0 {
		o.NumFrames = 97
	}
	if o.Steps == 0 {
		o.Steps = 7
	}
	if !o.GIF && !o.Frames {
		o.GIF = true
	}
}
