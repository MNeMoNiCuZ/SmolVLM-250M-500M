import os
import torch
import math
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

# User Configuration
MODEL_SIZE = "500M"  # Options: "256M" or "500M"
OVERWRITE_FILES = True  # Set to False to skip existing files
PREFIX = ""  # Text to prepend to the output
SUFFIX = ""  # Text to append to the output
OUTPUT_FORMAT = ".txt"  # Set desired output format, e.g., ".txt", ".md"
BATCH_SIZE = 10  # Number of images to process at the same time. 10 is good for a 24gb 3090
REMOVE_PREAMBLE = True  # Set to True to remove user/assistant preamble

CUSTOM_PROMPT = "Describe this image in detail."

# Example prompts
# Brief description: "describe"
# Long description: "Describe this image in detail."

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''}")

# Initialize processor and model
model_path = f"HuggingFaceTB/SmolVLM-{MODEL_SIZE}-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    _attn_implementation="eager",
)

model.to(DEVICE)

# Define input folder
input_folder = os.path.join(os.path.dirname(__file__), "input")

# Collect all valid images
valid_images = []
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
            image_path = os.path.join(root, filename)
            output_path = os.path.splitext(image_path)[0] + OUTPUT_FORMAT

            if OVERWRITE_FILES or not os.path.exists(output_path):
                valid_images.append((image_path, output_path))

print(f"\nUsing model: {model_path}")
print(f"Total valid images to process: {len(valid_images)}")
batch_count = math.ceil(len(valid_images)/BATCH_SIZE)
print(f"About to process {batch_count} {'batch' if batch_count == 1 else 'batches'}\n")

# Batch processing with progress bar
for i in tqdm(range(0, len(valid_images), BATCH_SIZE), desc="Processing Batches"):
    batch = valid_images[i:i + BATCH_SIZE]

    images = [Image.open(image_path).convert("RGB") for image_path, _ in batch]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CUSTOM_PROMPT}
            ]
        } for _ in images
    ]

    # Preprocess
    prompts = [processor.apply_chat_template([msg], add_generation_prompt=True) for msg in messages]
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_ids = generated_ids.to(DEVICE)
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # Process and save outputs
    for (image_path, output_path), output_text in zip(batch, generated_texts):
        if REMOVE_PREAMBLE and "Assistant:" in output_text:
            output_text = output_text.split("Assistant:", 1)[-1].strip()

        output_text = f"{PREFIX}{output_text}{SUFFIX}"

        with open(output_path, "w") as f:
            f.write(output_text)
