import os
import shutil
import subprocess
import torch
from diffusers import StableDiffusionPipeline

# Check available disk space
total, used, free = shutil.disk_usage("/")
print(f"Total: {total // (2**30)} GB")
print(f"Used: {used // (2**30)} GB")
print(f"Free: {free // (2**30)} GB")

required_space_gb = 4  # 3438.17 MB is approximately 4 GB

if free < required_space_gb * (2**30):
    raise RuntimeError("Not enough free disk space to download the model files.")

# Clear transformers cache
cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared transformers cache.")

# Install accelerate if not already installed
try:
    from accelerate import Accelerator
except ImportError:
    subprocess.check_call(["pip", "install", "accelerate"])
    from accelerate import Accelerator

# Load the pre-trained Stable Diffusion model from Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(model_id, low_cpu_mem_usage=True)
pipe = pipe.to(device)

# Define the text prompt
prompt = "images "

# Generate an image
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]

# Save the generated image
image.save("generated_image.png")

# Display the generated image
image.show()

