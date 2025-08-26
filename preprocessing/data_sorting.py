import os
import shutil
from argparse import ArgumentParser


import random


def copy_and_structure_valentini(src_dir, dest_dir):
    # Remove existing .data directory if it exists
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    # Prepare mapping for folder names
    folder_map = {}
    if os.path.exists(src_dir):
        for folder_name in os.listdir(src_dir):
            if "test" in folder_name:
                split = "test"
            elif "train" in folder_name:
                split = "train"
            else:
                continue  # Only process train and test

            if "clean" in folder_name:
                sub = "clean"
            elif "noisy" in folder_name:
                sub = "noisy"
            else:
                continue

            folder_map[(split, sub)] = os.path.join(src_dir, folder_name)

        # Handle test set (copy as is)
        for sub in ["clean", "noisy"]:
            src_subdir = folder_map.get(("test", sub))
            dest_subdir = os.path.join(dest_dir, "test", sub)
            os.makedirs(dest_subdir, exist_ok=True)
            if src_subdir and os.path.exists(src_subdir):
                for file in os.listdir(src_subdir):
                    src_file = os.path.join(src_subdir, file)
                    dest_file = os.path.join(dest_subdir, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dest_file)

        # Handle train set: split into train/valid (80/20)
        train_clean_dir = folder_map.get(("train", "clean"))
        train_noisy_dir = folder_map.get(("train", "noisy"))
        if train_clean_dir and train_noisy_dir:
            clean_files = sorted(
                [
                    f
                    for f in os.listdir(train_clean_dir)
                    if os.path.isfile(os.path.join(train_clean_dir, f))
                ]
            )
            noisy_files = sorted(
                [
                    f
                    for f in os.listdir(train_noisy_dir)
                    if os.path.isfile(os.path.join(train_noisy_dir, f))
                ]
            )
            # Assume files are paired by name
            paired_files = [(c, c) for c in clean_files if c in noisy_files]
            random.shuffle(paired_files)
            split_idx = int(0.8 * len(paired_files))
            train_pairs = paired_files[:split_idx]
            valid_pairs = paired_files[split_idx:]

            for split_name, pairs in zip(
                ["train", "valid"], [train_pairs, valid_pairs]
            ):
                for sub, dir_path in zip(
                    ["clean", "noisy"], [train_clean_dir, train_noisy_dir]
                ):
                    dest_subdir = os.path.join(dest_dir, split_name, sub)
                    os.makedirs(dest_subdir, exist_ok=True)
                for clean_file, noisy_file in pairs:
                    # Copy clean
                    src_clean = os.path.join(train_clean_dir, clean_file)
                    dest_clean = os.path.join(dest_dir, split_name, "clean", clean_file)
                    shutil.copy2(src_clean, dest_clean)
                    # Copy noisy
                    src_noisy = os.path.join(train_noisy_dir, noisy_file)
                    dest_noisy = os.path.join(dest_dir, split_name, "noisy", noisy_file)
                    shutil.copy2(src_noisy, dest_noisy)
        else:
            print("Warning: Could not find both clean and noisy train folders.")
    else:
        print(f"Warning: {src_dir} does not exist.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--basedir",
        type=str,
        required=True,
        help="Path to Valentini dataset root directory",
    )
    args = parser.parse_args()

    dest_data_dir = os.path.join(os.getcwd(), ".data")
    copy_and_structure_valentini(args.basedir, dest_data_dir)
    print(f"Data copied and structured in {dest_data_dir}")
