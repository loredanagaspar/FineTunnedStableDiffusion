import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
import wandb

# Hardcoded configuration
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
ARROW_DATA_PATH = "/root/FineTunnedStableDiffusion/img/loredana_dataset"
INSTANCE_PROMPT = "a photo of sks woman"
OUTPUT_DIR = "/root/FineTunnedStableDiffusion/loredana_lora_results"
WANDB_PROJECT = "loredana_portrait_lora"
TRAIN_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
RESOLUTION = 512

class ArrowImageCaptionDataset(Dataset):
    def __init__(self, dataset_path, prompt_key="text", image_key="image"):
        self.dataset = load_from_disk(dataset_path)
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example[self.image_key].convert("RGB")
        prompt = example[self.prompt_key]
        image = self.transform(image)
        return image, prompt

def setup_environment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wandb.init(
        project=WANDB_PROJECT,
        name="loredana_lora_training",
        config={"model": MODEL_NAME, "data": ARROW_DATA_PATH}
    )

def apply_lora_to_unet(unet):
    for _, module in unet.named_modules():
        if hasattr(module, 'set_attn_processor'):
            module.set_attn_processor(LoRAAttnProcessor2_0())

def train():
    setup_environment()

    dataset = ArrowImageCaptionDataset(ARROW_DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
    unet = pipeline.unet
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    apply_lora_to_unet(unet)
    unet.train()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=LEARNING_RATE)

    for epoch in range(TRAIN_EPOCHS):
        for step, (images, captions) in enumerate(dataloader):

            images = images.to("cuda", dtype=torch.float16)
            text_inputs = tokenizer(list(captions), padding="max_length", truncation=True, max_length=77, return_tensors="pt")
            input_ids = text_inputs.input_ids.to("cuda")
            attention_mask = text_inputs.attention_mask.to("cuda")

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]
            
            text_embeds = encoder_hidden_states
            time_ids = torch.tensor(
                [[0, 0, RESOLUTION, RESOLUTION, RESOLUTION, RESOLUTION]] * images.shape[0],
                device=text_embeds.device,
                dtype=text_embeds.dtype
            )
            time_ids = time_ids[:, None, :].expand(-1, text_embeds.shape[1], -1)
            if step==0 and epoch == 0:
                torch.save(text_embeds, "debug_text_embeds.pt")
                torch.save(time_ids, "debug_time_ids.pt")
                print("Debug files saved.")
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            model_pred = unet(
                noisy_images,
                timesteps,
                encoder_hidden_states=None,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False
            )[0]

            loss = torch.nn.functional.mse_loss(model_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                wandb.log({"loss": loss.item()})

    pipeline.save_pretrained(OUTPUT_DIR)
    wandb.finish()

if __name__ == "__main__":
    train()
