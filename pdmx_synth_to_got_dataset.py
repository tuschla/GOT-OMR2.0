import os
import json
import functools
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# === CONFIGURATION ===
OUTPUT_DIR = "./datasets/PDMX_synth_GOT"
TARGET_SPLITS = ["train", "val", "test"]
NUM_WORKERS = 64  # Start with 4; increase only if RAM allows
# =====================

# Global variable to hold the dataset so workers can access it via memory mapping
dataset = None


def process_single_item(index, split, images_dir):
    """Worker function: Accesses the global dataset by index only."""
    global dataset

    try:
        example = dataset[split][index]
        image_obj = example["image"]
        transcription = example["transcription"]
        filename = example["filename"]

        if not os.path.splitext(filename)[1]:
            filename = f"{filename}.png"

        image_path = os.path.join(images_dir, filename)

        # Save the image
        if not os.path.exists(image_path):
            image_obj.save(image_path)

        return {
            "id": f"pdmx_{split}_{index}",
            "image": filename,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nConvert this musical score to ABC notation:",
                },
                {"from": "gpt", "value": transcription},
            ],
        }
    except Exception as e:
        return None


def main():
    global dataset

    print("Loading dataset (streaming/mapping)...")
    try:
        # We load the dataset here so it's available in the parent memory space
        dataset = load_dataset("guangyangmusic/PDMX-Synth")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    for split in TARGET_SPLITS:
        if split not in dataset:
            continue

        print(f"\n--- Processing '{split}' ({len(dataset[split])} items) ---")

        split_images_dir = os.path.join(OUTPUT_DIR, split, "images")
        os.makedirs(split_images_dir, exist_ok=True)

        indices = range(len(dataset[split]))

        # Use functools to pre-fill the split and directory arguments
        worker_func = functools.partial(
            process_single_item, split=split, images_dir=split_images_dir
        )

        got_data = []

        # Process in parallel using only indices to keep memory usage low
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # chunksize=10 prevents the executor from flooding the RAM with too many tasks at once
            results = list(
                tqdm(
                    executor.map(worker_func, indices, chunksize=256),
                    total=len(indices),
                )
            )

            # Clean up None values from failed entries
            got_data = [r for r in results if r is not None]

        # Save JSON
        json_output_path = os.path.join(OUTPUT_DIR, f"{split}.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(got_data, f, indent=2)

        print(f"✅ Split '{split}' saved to {json_output_path}")


if __name__ == "__main__":
    main()
