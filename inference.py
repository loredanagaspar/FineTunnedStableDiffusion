from diffusers import StableDiffusionPipeline
import torch

# Load your fine-tuned LoRA model
model_id = "your_huggingface_username/loredana-personal-portrait-lora"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Generate image
prompt = "Loredana Gaspar in a professional setting, corporate portrait"
image = pipe(prompt).images[0]
image.save("generated_portrait.png")