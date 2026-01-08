import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5"
    
    print(f"Loading pipeline from {model_path}...")
    pipeline = LTXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        variant=None
    )
    # Remove .to(device) because cpu_offload handles it
    
    print("Enabling CPU offload and optimizations...")
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_tiling()


    prompt = "A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show."
    negative_prompt = "" 
    seed = 267805598632705
    
    print(f"Generating with seed {seed}...")
    generator = torch.Generator(device=device).manual_seed(seed)

    video = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=768,
        height=512,
        num_frames=97,
        num_inference_steps=30,
        guidance_scale=3.0,
        generator=generator,
        output_type="pt"
    ).frames[0]

    output_path = "output_diffusers.mp4"
    export_to_video(video, output_path, fps=24)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    run_inference()
