
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
import matplotlib.pyplot as plt

# --- 3. Load Dataset and Tokenizer ---

dataset = load_from_disk("/root/FineTunnedStableDiffusion/data")
tokenizer_1 = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer", use_fast=False
)
tokenizer_2 = AutoTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", use_fast=False
)

# --- 4. Split Dataset into Train and Test ---
split_dataset = dataset.train_test_split(test_size=3, seed=42)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# --- 5. Define PyTorch Dataset Class ---
class LoraDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer_1, tokenizer_2):
        self.dataset = hf_dataset
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.image_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.image_transform(self.dataset[idx]["image"])
        caption = self.dataset[idx]["text"]
        token_1 = self.tokenizer_1(
            caption, truncation=True, padding="max_length", max_length=77, return_tensors="pt"
        )
        token_2 = self.tokenizer_2(
            caption, truncation=True, padding="max_length", max_length=77, return_tensors="pt"
        )
        return image, token_1.input_ids.squeeze(0), token_2.input_ids.squeeze(0)

# --- 6. Instantiate Datasets and DataLoaders ---
train_data = LoraDataset(train_dataset, tokenizer_1, tokenizer_2)
test_data = LoraDataset(test_dataset, tokenizer_1, tokenizer_2)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# --- 7. Inspect Sample Output ---
sample_image, sample_input_ids_1, sample_input_ids_2 = train_data[0]
print("Image tensor shape:", sample_image.shape)
print("Image tensor range:", sample_image.min().item(), "to", sample_image.max().item())
print("Tokenized text (input_ids_1):", sample_input_ids_1.tolist())
print("Tokenized text (input_ids_2):", sample_input_ids_2.tolist())
