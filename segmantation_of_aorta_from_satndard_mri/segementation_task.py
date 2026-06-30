# -*- coding: utf-8 -*-

"""
nnUNet v2 - Aorta Segmentation Pipeline (Refactored)
"""

import os
import json
import shutil
import subprocess
from glob import glob
from collections import OrderedDict

import torch


# -----------------------------
# System Info
# -----------------------------
def print_system_info():
    print("=== System Info ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()


# -----------------------------
# Directory Setup
# -----------------------------
def create_directories(base_dir):
    nnunet_raw = os.path.join(base_dir, "nnUNet_raw")
    nnunet_preprocessed = os.path.join(base_dir, "nnUNet_preprocessed")
    nnunet_results = os.path.join(base_dir, "nnUNet_results")

    for d in [nnunet_raw, nnunet_preprocessed, nnunet_results]:
        os.makedirs(d, exist_ok=True)

    os.environ['nnUNet_raw'] = nnunet_raw
    os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
    os.environ['nnUNet_results'] = nnunet_results

    print("Directories created.")
    return nnunet_raw


# -----------------------------
# Dataset Structure
# -----------------------------
def create_dataset_structure(nnunet_raw, dataset_id=201, dataset_name="AORTA_MRI"):
    dataset_dir = os.path.join(nnunet_raw, f"Dataset{dataset_id}_{dataset_name}")
    print(dataset_dir)
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")

    for d in [dataset_dir, images_tr, labels_tr, images_ts]:
        os.makedirs(d, exist_ok=True)

    print(f"Dataset structure created at: {dataset_dir}")

    return dataset_dir, images_tr, labels_tr, images_ts


# -----------------------------
# File Copy Utility
# -----------------------------
def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# -----------------------------
# Dataset Processing (YOUR LOGIC ✅)
# -----------------------------
def process_dataset(images_source, labels_source, images_tr, labels_tr):

    image_files = sorted(glob(os.path.join(images_source, "*.nii.gz")))

    print(f"Images source: {images_source}")
    print(f"Labels source: {labels_source}")
    print(f"Found images: {len(image_files)}")

    processed_count = 0
    skipped_count = 0
    errors = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)

        # match label name (keep your logic)
        label_name = img_name.replace('_0000', '')
        label_path = os.path.join(labels_source, label_name)

        print(f"Checking: {img_name} → {label_name}")

        if not os.path.exists(label_path):
            errors.append(f"Missing label for {img_name}: {label_path}")
            skipped_count += 1
            continue

        dst_img = os.path.join(images_tr, img_name)
        dst_label = os.path.join(labels_tr, label_name)

        if not os.path.exists(dst_img):
            copy_file(img_path, dst_img)

        if not os.path.exists(dst_label):
            copy_file(label_path, dst_label)

        processed_count += 1

    print(f"\nProcessed: {processed_count} image-label pairs")
    print(f"Skipped: {skipped_count}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")


# -----------------------------
# dataset.json Creation
# -----------------------------
def create_dataset_json(dataset_dir, images_tr, labels_tr):
    label_files = sorted([f for f in os.listdir(labels_tr) if f.endswith(".nii.gz")])

    training_entries = []
    for lab in label_files:
        img = lab.replace(".nii.gz", "_0000.nii.gz")

        training_entries.append({
            "image": f"./imagesTr/{img}",
            "label": f"./labelsTr/{lab}"
        })

    dataset_dict = OrderedDict({
        "name": "AORTA_MRI",
        "description": "Aorta Segmentation",
        "reference":"",
        "licence":"",
        "tensorImageSize": "3D",
        "modality": {"0": "MRI"},
        "channel_names": {"0": "MRI"},
        "file_ending": ".nii.gz",
        "labels": {
            "background": 0,
            "aorta": 1
        },
        "numTraining": len(training_entries),
        "training": training_entries,
        "overwrite_image_reader_writer":"SimpleITKIO"
    })

    json_path = os.path.join(dataset_dir, "dataset.json")

    with open(json_path, "w") as f:
        json.dump(dataset_dict, f, indent=4)

    print(f"dataset.json created: {json_path}")


# -----------------------------
# nnUNet Commands
# -----------------------------
def run_preprocessing(dataset_id):
    print("\nRunning preprocessing...")

    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c 3d_fullres --verify_dataset_integrity"
    print(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print("\n--- STDOUT ---")
    print(result.stdout)

    print("\n--- STDERR ---")
    if result.stderr:
        print("\n--- STDERR ---")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Preprocessing failed!")
    else:
        print(f"\nReturn code: {result.returncode}")


def train_model(dataset_id, fold=0):
    print("\nStarting training...")

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['nnUNet_n_proc_DA'] = '1'

    cmd = f"nnUNetv2_train {dataset_id} 3d_fullres {fold}"
    print(cmd)

    subprocess.run(cmd, shell=True, check=True)


# -----------------------------
# Main
# -----------------------------
def main():
    print("🚀 Script started\n")
    torch.multiprocessing.set_sharing_strategy('file_system')
    print_system_info()

    base_dir = os.getcwd()
    raw_data_dir = os.path.join(base_dir, "train_data")

    images_source = os.path.join(raw_data_dir, "ao_imgs")
    labels_source = os.path.join(raw_data_dir, "ao_segs")

    nnunet_raw = create_directories(base_dir)

    dataset_id = 201
    dataset_dir, images_tr, labels_tr, _ = create_dataset_structure(nnunet_raw, dataset_id)

    process_dataset(images_source, labels_source, images_tr, labels_tr)

    create_dataset_json(dataset_dir, images_tr, labels_tr)

    run_preprocessing(dataset_id)

    train_model(dataset_id, fold=0)


if __name__ == "__main__":
    main()
