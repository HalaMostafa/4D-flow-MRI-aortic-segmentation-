import subprocess
import os
import shutil
from glob import glob

DATASET_ID = 201


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

def create_dataset_structure(nnunet_raw, dataset_id=DATASET_ID, dataset_name="AORTA_MRI"):
    dataset_dir = os.path.join(nnunet_raw, f"Dataset{dataset_id}_{dataset_name}")
    print(dataset_dir)
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")

    for d in [dataset_dir, images_ts,labels_ts]:
        os.makedirs(d, exist_ok=True)

    print(f"Dataset structure created at: {dataset_dir}")

    return dataset_dir,images_ts,labels_ts

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


# -----------------------------
# Dataset Processing (YOUR LOGIC ✅)
# -----------------------------
def process_dataset(images_source, images_ts):

    image_files = sorted(glob(os.path.join(images_source, "*.nii.gz")))

    print(f"Images source: {images_source}")
    print(f"Found images: {len(image_files)}")

    processed_count = 0
    skipped_count = 0
    errors = []

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        dst_img = os.path.join(images_ts, img_name)

        if not os.path.exists(dst_img):
            copy_file(img_path, dst_img)

        processed_count += 1

    print(f"\nProcessed: {processed_count} image-testing ")
    print(f"Skipped: {skipped_count}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")



def run_inference(input_image, output_path, dataset_id=DATASET_ID):
    """
    Run nnUNet inference on a single image.
    
    Args:
        input_image: Path to input .nii.gz image
        output_path: Path to save prediction
        dataset_id: Dataset ID used for training
    """
    cmd = f'nnUNetv2_predict -i "{input_image}" -o "{output_path}" -d {dataset_id} -c 3d_fullres -f 0 -chk checkpoint_best.pth'
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0




def main():

    base_dir = os.getcwd()
    nnunet_raw = create_directories(base_dir)
    raw_data_dir = os.path.join(base_dir, "sample_data")
    images_source = os.path.join(raw_data_dir, "ao_imgs")
    dataset_dir, images_ts, labels_ts = create_dataset_structure(nnunet_raw)
    process_dataset(images_source, images_ts)
    print("PATH TO IMGS",images_ts)
    print("OUTPUT",labels_ts)
    run_inference(images_ts, labels_ts)

   


if __name__ == "__main__":
    main()