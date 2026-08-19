import sys
import glob
import re
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Paths (DOCKER SAFE ✅)
# ----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent   # racelette/
ZIP_ROOT = HERE / "RACELETTE"  # scripts/RACELETTE

# ----------------------------------------------------------------------------
# Add repo scripts (adjust if needed)
# ----------------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT / "Scripts"))

from dataHandler import (
    read_4dflow_frames,
    save_4dflow_nifti,
    load_4dflow_nifti,
)
from flowProcessing import extract_velocities, extract_ivsds

# ----------------------------------------------------------------------------
# ZIP extraction
# ----------------------------------------------------------------------------

def extract_zip(zip_path, extract_root):
    extract_path = extract_root / zip_path.stem

    if not extract_path.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
    else:
        print(f"Already extracted: {zip_path.name}")

    return extract_path


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _sorted_pv(directory, pattern):
    files = sorted(
        glob.glob(str(directory / pattern)),
        key=lambda f: int(re.search(r'_(\d+)\.pv$', f).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"No '{pattern}' files found in {directory}")
    return [Path(f) for f in files]


def _dir_size_mb(directory):
    return sum(f.stat().st_size for f in Path(directory).iterdir() if f.is_file()) / 1024 ** 2


def _mb(path):
    return Path(path).stat().st_size / 1024 ** 2


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _save_fig(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

FORMATS = [
    ('raw', False),
    ('gzip', True),
]

zip_files = sorted(ZIP_ROOT.glob("VirtualSubject_n*.zip"))

if not zip_files:
    raise FileNotFoundError(f"No ZIP files found in {ZIP_ROOT}")

for zip_file in zip_files:

    print("\n" + "=" * 60)
    print(f"Processing {zip_file.name}")
    print("=" * 60)

    subject_name = zip_file.stem

    # ✅ Extract inside same folder
    subject_dir = extract_zip(zip_file, ZIP_ROOT)

    # ✅ Your TRUE structure (double folder)
    DATA_DIR = subject_dir / subject_name / "Data"

    if not DATA_DIR.exists():
        # fallback (robust)
        candidates = list(subject_dir.rglob("7p_4DFlow_*.pv"))
        if not candidates:
            print("No PV files found. Skipping.")
            continue
        DATA_DIR = candidates[0].parent

    print(f"Using DATA_DIR: {DATA_DIR}")

    # ✅ Output per subject
    EXPORT_DIR = HERE / "Exported" / subject_name
    RESULTS_DIR = HERE / "Results" / subject_name

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ✅ Skip processed
    if (EXPORT_DIR / "mask.nii.gz").exists() or (EXPORT_DIR / "mask.nii").exists():
        print("Already processed. Skipping.")
        continue

    # ------------------------------------------------------------------------
    # Load MRI
    # ------------------------------------------------------------------------

    print("Loading MRI frames …")

    dims, spacing, mask, signal, params = read_4dflow_frames(
        str(DATA_DIR / '7p_4DFlow_*.pv')
    )

    n_enc = signal.shape[3]
    n_frames = signal.shape[4]

    kvs = np.asarray(params['kvs']).ravel().astype(np.float64) if 'kvs' in params else None

    mri_files = _sorted_pv(DATA_DIR, '7p_4DFlow_*.pv')
    pv_mri_size = sum(_mb(f) for f in mri_files)

    print(f"{len(mri_files)} frames | {n_enc} enc | {pv_mri_size:.1f} MB")

    mag_ref = np.abs(signal).astype(np.float32)
    phase_ref = np.angle(signal).astype(np.float32)

    if kvs is not None:
        vel_ref = extract_velocities(signal, kvs).astype(np.float32)
        ivsd_ref = extract_ivsds(signal, kvs).astype(np.float32)

    # ------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------

    results = {}

    for label, compressed in tqdm(FORMATS, desc="Formats"):

        ext = '.nii.gz' if compressed else '.nii'
        out_dir = EXPORT_DIR / f"mri_{label}"

        save_4dflow_nifti(out_dir, dims, spacing, mask, signal, params,
                          compressed=compressed)

        dims_r, spacing_r, mask_r, signal_r, _ = load_4dflow_nifti(
            out_dir, compressed=compressed)

        err_mag = _max_err(mag_ref, np.abs(signal_r))
        err_phase = _max_err(phase_ref, np.angle(signal_r))

        results[label] = {
            "size": _dir_size_mb(out_dir),
            "err_mag": err_mag,
            "err_phase": err_phase,
        }

    # ------------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------------

    labels = list(results.keys())
    sizes = [results[k]['size'] for k in labels]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes)

    ax.set_title(subject_name)
    ax.set_ylabel("Size (MB)")

    fig.tight_layout()
    _save_fig(fig, RESULTS_DIR / "sizes.png")

print("\n✅ ALL SUBJECTS DONE")