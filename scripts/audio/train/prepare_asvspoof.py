#!/usr/bin/env python3
"""
ASVspoof 2019 LA Dataset Preprocessing
将 ASVspoof 协议文件转换为我们的训练格式

原始格式:
ASVspoof2019_LA_train/
├── flac/
│   ├── LA_T_1000137.flac
│   └── ...
└── ASVspoof2019_LA_cm_protocols/
    └── ASVspoof2019.LA.cm.train.trn.txt

协议文件格式:
SPEAKER_ID AUDIO_FILE_NAME - ATTACK_TYPE LABEL
LA_T_1000137 LA_T_1000137 - - bonafide
LA_T_1000265 LA_T_1000265 - A07 spoof

目标格式:
processed/
├── train/
│   ├── bonafide/
│   │   ├── LA_T_1000137.flac
│   │   └── ...
│   └── spoof/
│       ├── LA_T_1000265.flac
│       └── ...
├── dev/
│   ├── bonafide/
│   └── spoof/
└── eval/
    ├── bonafide/
    └── spoof/
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_protocol_file(protocol_path):
    """
    Parse ASVspoof protocol file
    Returns: dict mapping audio_file -> label (bonafide/spoof)
    """
    audio_labels = {}

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # Format: SPEAKER_ID AUDIO_FILE_NAME - ATTACK_TYPE LABEL
            audio_file = parts[1]
            label = parts[4]  # bonafide or spoof

            audio_labels[audio_file] = label

    return audio_labels


def organize_dataset(input_root, output_root, split='train'):
    """
    Organize ASVspoof dataset into bonafide/spoof structure

    Args:
        input_root: Path to ASVspoof2019_LA_{split}/
        output_root: Path to output directory
        split: train/dev/eval
    """
    print(f"\n{'='*60}")
    print(f"Processing {split} split")
    print(f"{'='*60}")

    # Paths
    audio_dir = os.path.join(input_root, f'ASVspoof2019_LA_{split}', 'flac')
    protocol_dir = os.path.join(input_root, f'ASVspoof2019_LA_cm_protocols')

    if split == 'train':
        protocol_file = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.train.trn.txt')
    elif split == 'dev':
        protocol_file = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.dev.trl.txt')
    elif split == 'eval':
        protocol_file = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.eval.trl.txt')
    else:
        raise ValueError(f"Unknown split: {split}")

    # Check paths exist
    if not os.path.exists(audio_dir):
        print(f"[Error] Audio directory not found: {audio_dir}")
        return

    if not os.path.exists(protocol_file):
        print(f"[Error] Protocol file not found: {protocol_file}")
        return

    # Parse protocol file
    print(f"[1/3] Parsing protocol file: {protocol_file}")
    audio_labels = parse_protocol_file(protocol_file)
    print(f"      Found {len(audio_labels)} audio files")

    # Count labels
    bonafide_count = sum(1 for label in audio_labels.values() if label == 'bonafide')
    spoof_count = sum(1 for label in audio_labels.values() if label == 'spoof')
    print(f"      - Bonafide: {bonafide_count}")
    print(f"      - Spoof: {spoof_count}")

    # Create output directories
    output_split_dir = os.path.join(output_root, split)
    bonafide_dir = os.path.join(output_split_dir, 'bonafide')
    spoof_dir = os.path.join(output_split_dir, 'spoof')

    os.makedirs(bonafide_dir, exist_ok=True)
    os.makedirs(spoof_dir, exist_ok=True)

    # Copy files
    print(f"[2/3] Copying audio files to {output_split_dir}")

    copied_bonafide = 0
    copied_spoof = 0
    missing_files = []

    for audio_file, label in tqdm(audio_labels.items(), desc=f"  Copying {split}"):
        src_path = os.path.join(audio_dir, f"{audio_file}.flac")

        if not os.path.exists(src_path):
            missing_files.append(audio_file)
            continue

        if label == 'bonafide':
            dst_path = os.path.join(bonafide_dir, f"{audio_file}.flac")
            shutil.copy2(src_path, dst_path)
            copied_bonafide += 1
        elif label == 'spoof':
            dst_path = os.path.join(spoof_dir, f"{audio_file}.flac")
            shutil.copy2(src_path, dst_path)
            copied_spoof += 1

    print(f"[3/3] Summary:")
    print(f"      - Copied {copied_bonafide} bonafide files")
    print(f"      - Copied {copied_spoof} spoof files")

    if missing_files:
        print(f"      - Warning: {len(missing_files)} files not found")
        print(f"        (First 5: {missing_files[:5]})")

    print(f"      ✓ {split} split completed!")


def main():
    parser = argparse.ArgumentParser(description='Preprocess ASVspoof 2019 LA dataset')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to ASVspoof2019_LA root directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--splits', type=str, default='train,dev,eval',
                        help='Splits to process (comma-separated)')

    args = parser.parse_args()

    # Process each split
    splits = args.splits.split(',')

    print("=" * 60)
    print("ASVspoof 2019 LA Dataset Preprocessing")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Splits: {splits}")

    for split in splits:
        split = split.strip()
        organize_dataset(args.input, args.output, split)

    print("\n" + "=" * 60)
    print("All splits processed successfully!")
    print("=" * 60)
    print(f"\nOutput structure:")
    print(f"{args.output}/")
    for split in splits:
        print(f"├── {split.strip()}/")
        print(f"│   ├── bonafide/")
        print(f"│   └── spoof/")

    print(f"\nYou can now train with:")
    print(f"python train_audio_forgery.py \\")
    print(f"    --dataset_root {args.output}/train \\")
    print(f"    --backbone nvidia/MambaVision-T-1K \\")
    print(f"    --stage1_epochs 50 \\")
    print(f"    --stage2_epochs 50")


if __name__ == '__main__':
    main()
