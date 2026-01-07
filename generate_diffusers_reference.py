import torch
import sys
import os

# Add local diffusers source to path
sys.path.insert(0, os.path.join(os.getcwd(), "tp", "diffusers", "src"))

from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# Path to the model snapshot
model_path = r"c:\candle-video\models\models--Lightricks--LTX-Video-0.9.5"
output_dir = "reference_output"
os.makedirs(output_dir, exist_ok=True)

prompt = "The waves crash against the jagged rocks of the shoreline, sending spray high into the air.The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon."
negative_prompt = "worst quality, bad quality, deformation, low resolution"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading pipeline from {model_path}...")
pipe = LTXPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16, 
).to(device)

# Fixed seed for reproducibility
seed = 42
generator = torch.Generator(device).manual_seed(seed)

print(f"Generating video...")
# Matching the parameters from sample_ltx
# steps=30, height=512, width=768, num_frames=9 (which is 2 frames in latent space if compression is 4?)
# Actually LTX compression is 8? Let's check.
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=9, # This should likely be 9, 17, 25... (8*k + 1)
    num_inference_steps=30,
    guidance_scale=3.0,
    generator=generator,
    output_type="np", # Get numpy array
).frames[0]

# Save frames as images for side-by-side comparison
import PIL.Image as Image
for i, frame in enumerate(video):
    img = Image.fromarray((frame * 255).astype("uint8"))
    img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

print(f"Done! Reference frames saved to {output_dir}")
