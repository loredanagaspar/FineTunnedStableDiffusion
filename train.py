from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from peft import LoraConfig
import torch
from accelerate import Accelerator
from tqdm import tqdm
from data_utils import train_loader, test_loader
import wandb

# Initialize

wandb.init(project="SDXL-LoRA-Training", config={
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "lora_rank":8,
    "batch_size": train_loader.batch_size,
})

# Detect CUDA and print GPU info
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

accelerator = Accelerator(
    gradient_accumulation_steps=2,
    mixed_precision="fp16"
    log_with="wandb"
)
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,cache_dir=os.getenv("HF_HOME", "./cache"))
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.float16, cache_dir=os.getenv("HF_HOME", "./cache"))
pipe.to(accelerator.device)

# LoRA Config
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # SDXL attention layers
    lora_dropout=0.1,
    bias="none"
)
pipe.unet.add_adapter(lora_config)

# Optimizer and LR Scheduler
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))

# Prepare with Accelerator
pipe.unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
    pipe.unet, optimizer, train_loader, lr_scheduler
)

# Training Loop
num_epochs = 5  
global_step=0

for epoch in range(num_epochs):
    pipe.unet.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for step, batch in enumerate(progress_bar):
        images, captions = batch
        images = images.to(accelerator.device)
        
        # Forward pass (SDXL expects latents)
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215
        
        # Compute loss
        with accelerator.accumulate(pipe.unet):
            loss = pipe(
                captions, 
                latents=latents,
                num_inference_steps=1,  # Shortcut for training
                return_dict=False
            ).loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        #Logging
        global_step+= 1
        logs={
            "loss":loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "epoch": epoch + (step+1)/len(train_loader),
            "step": global_step

        }
        progress_bar.set_postfix(**{k: f"{v:.4f}" for k, v in logs.items()})

        #Log to W&B
        if accelerator.is_main_process:
            wandb.log(logs)


# Save LoRA
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    pipe.unet.save_pretrained("./lora_weights", safe_serialization=True)
    wandb.finish()