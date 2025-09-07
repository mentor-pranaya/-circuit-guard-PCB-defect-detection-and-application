import os
import argparse
import shutil
import random
from glob import glob
from PIL import Image

def prepare_dataset(input_dir, output_dir, img_size):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Folders
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect classes
    classes = [c for c in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, c))]
    print(f"Classes found: {classes}")

    total_count = 0
    per_class_counts = {}

    for cls in classes:
        img_paths = glob(os.path.join(input_dir, cls, "*"))
        random.shuffle(img_paths)

        n = len(img_paths)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val

        per_class_counts[cls] = n

        # Output dirs
        for split, paths in zip(
            ["train", "val", "test"],
            [img_paths[:n_train], img_paths[n_train:n_train+n_val], img_paths[n_train+n_val:]]
        ):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for p in paths:
                img = Image.open(p).convert("RGB").resize((img_size, img_size))
                out_path = os.path.join(split_dir, os.path.basename(p))
                img.save(out_path)

        total_count += n

    print("✅ Processed dataset created")
    print(f"Per-class counts: {per_class_counts}")
    print(f"Total images: {total_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    prepare_dataset(args.input_dir, args.output_dir, args.img_size)
