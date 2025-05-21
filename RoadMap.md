🛣️ Roadmap: Image Deblurring using Stable Diffusion 1.5 + ControlNet + LoRA

🔧 Prerequisites
✅ Environment
Hardware: 24GB VRAM GPU (e.g., A6000, RTX 3090/4090)

OS: Linux or WSL2 preferred (Windows works but trickier)

Python: 3.9 or 3.10

Virtual environment: venv or conda recommended

✅ Dependencies
Install these:

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets opencv-python scikit-image
pip install peft bitsandbytes

🧱 Task 1: Data Preparation
🔹 Step 1.1: Download a Clean Image Dataset
Example: Use CelebA or COCO from Hugging Face.

from datasets import load_dataset
dataset = load_dataset("celebA", split="train[:1000]")  # small subset for demo

🔹 Step 1.2: Generate Synthetic Blurred Versions
For each image:

Apply Gaussian blur

Save both blurred.png and sharp.png

import cv2
import os

def apply_gaussian_blur(img, kernel_size=9):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def save_blurred_dataset(images, output_dir="blurred_dataset"):
    os.makedirs(f"{output_dir}/sharp", exist_ok=True)
    os.makedirs(f"{output_dir}/blurred", exist_ok=True)
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blurred = apply_gaussian_blur(img)
        cv2.imwrite(f"{output_dir}/sharp/{i:04d}.png", img)
        cv2.imwrite(f"{output_dir}/blurred/{i:04d}.png", blurred)


🤖 Task 2: Model Setup — Stable Diffusion + ControlNet
🔹 Step 2.1: Load Stable Diffusion 1.5

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.to("cuda")

🔹 Step 2.2: Use Control Input (like Canny edges)
Use blurred image → extract edges → condition model

import numpy as np
def canny_edge(img):
    return cv2.Canny(img, 100, 200)


🧠 Task 3: Fine-Tuning with LoRA
🔹 Step 3.1: Convert SD to LoRA-compatible
Use Hugging Face peft or kohya-ss backend.

🔹 Step 3.2: LoRA Configuration
Set:

Rank = 4 or 8 (LoRA size)

Learning rate = 1e-4 to 2e-4

Epochs = 5–10

Batch size = 2–4 (fit in 24GB with half precision)

🔹 Step 3.3: Training Loop
Train to minimize pixel loss or latent loss between outputs and ground truth sharp image.

🧪 Task 4: Evaluation — SSIM / PSNR / Visuals
🔹 Step 4.1: Compute SSIM/PSNR

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compare_metrics(img1, img2):
    return {
        "ssim": ssim(img1, img2, channel_axis=-1),
        "psnr": psnr(img1, img2)
    }

🔹 Step 4.2: Visual Samples
Log:

Input (blurred)

Output (deblurred)

Ground truth (sharp)
Use matplotlib or export with gradio for quick UI.

🧪 Task 5: Export & Reuse
🔹 Step 5.1: Save LoRA Weights
Export the fine-tuned adapter for re-use:

model.save_pretrained("controlnet-lora-deblur/")