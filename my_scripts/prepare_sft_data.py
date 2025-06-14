import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# === Config ===
use_cot = False  # Set to True to preserve <think> and <answer> format

# Load dataset
dataset = load_dataset("laolao77/ViRFT_CLS_flower_4_shot", split="train")

# Define output folders
output_root = os.path.abspath("/app/shared_data/raja/oxford_flowers/sft_data/")
image_output_dir = os.path.join(output_root, "images")
json_output_path = os.path.join(output_root, f"sft_dataset_cot_{use_cot}.json")

# Ensure folders exist
os.makedirs(image_output_dir, exist_ok=True)

# Template prompt if CoT is disabled
DEFAULT_PROMPT = "<image>This is an image containing a flower. Please identify the species of the flower based on the image. Only output the name of the species without any additional text."

sft_data = []

for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing examples"):
    pil_img: Image.Image = example["image"]
    original_prompt = example["problem"]
    original_response = example["solution"]

    # Save image
    image_filename = f"image_{idx:05d}.png"
    image_path = os.path.abspath(os.path.join(image_output_dir, image_filename))
    pil_img.save(image_path)

    if use_cot:
        prompt = f"<image>{original_prompt}"
        response = original_response
    else:
        prompt = DEFAULT_PROMPT
        # Strip <answer>...</answer> tags
        if "<answer>" in original_response and "</answer>" in original_response:
            start = original_response.find("<answer>") + len("<answer>")
            end = original_response.find("</answer>")
            response = original_response[start:end].strip()
        else:
            response = original_response.strip()

    sft_entry = {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response}
        ],
        "images": [image_path]
    }

    sft_data.append(sft_entry)


# Save to JSON
with open(json_output_path, "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"✅ SFT JSON saved to: {json_output_path}")
print(f"✅ Images saved under: {image_output_dir}")
