import torch
from transformers import T5EncoderModel, T5Tokenizer
from safetensors.torch import save_file
import os

def capture_t5():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float32 to be safe for comparison, though model might be bf16
    dtype = torch.float32

    model_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/text_encoder"
    tokenizer_path = "c:/candle-video/models/models--Lightricks--LTX-Video-0.9.5/text_encoder"

    print(f"Loading T5 from {model_path}...")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    prompt = "A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show."
    
    print("Encoding prompt...")
    max_length = 128
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

    print(f"Output shape: {last_hidden_state.shape}")

    save_data = {
        "input_ids": input_ids.cpu().contiguous(),
        "attention_mask": attention_mask.cpu().contiguous(),
        "output": last_hidden_state.cpu().contiguous().to(torch.float32) # Ensure F32 for comparison
    }

    os.makedirs("reference_output", exist_ok=True)
    save_file(save_data, "reference_output/t5_ref.safetensors")
    print("Saved reference_output/t5_ref.safetensors")

if __name__ == "__main__":
    capture_t5()
