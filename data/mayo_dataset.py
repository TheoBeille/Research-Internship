"""
mayo_dataset.py

Extraction and preparation of the Mayo Clinic dataset (LDCT-and-Projection-data,
downloaded through idc-index) for the CT reconstruction pipeline.

Pipeline:
    1. Explore available DICOM series for each patient (full dose, low dose, projections)
    2. Loads the "Full Dose Images" series (ground truth u_true), sorted by physical position
    3. Converts to Hounsfield units, applies windowing, and normalizes to [0, 1]
    4. Caches each patient as .npy (avoids decoding the DICOM files on every run)
    5. Provides a torch.utils.data.Dataset ready to connect to data/dataset.py

Usage:
    python mayo_dataset.py --explore                  # inspect available series
    python mayo_dataset.py --build-cache              # build the .npy cache
    python mayo_dataset.py --build-cache --size 128   # optional resized variant

Then in your training code:
    from mayo_dataset import MayoSliceDataset
    train_ds = MayoSliceDataset(["L004", "L006", "L012", "L019"], cache_dir="./data/mayo_cache_512")
    test_ds  = MayoSliceDataset(["L014"], cache_dir="./data/mayo_cache_512")
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

try:
    from skimage.transform import resize as sk_resize
except ImportError:
    sk_resize = None  # only needed when --size is used


# --------------------------------------------------------------------------
# Default configuration - adjust to your directory layout
# --------------------------------------------------------------------------

DEFAULT_ROOT = Path("./mayo_data/ldct_and_projection_data")
DEFAULT_CACHE_DIR = Path("./data/mayo_cache_512")
DEFAULT_HU_WINDOW = (-160, 240)  # standard abdominal window

DEFAULT_TRAIN_PATIENTS = ["L004", "L006", "L012", "L019"]
DEFAULT_TEST_PATIENTS = ["L014"]


# --------------------------------------------------------------------------
# 1. Explore DICOM series
# --------------------------------------------------------------------------

def explore_patient(patient_dir: Path):
    """Display available series for a patient (description, number of slices)."""
    print(f"\n=== {patient_dir.name} ===")
    for study_dir in sorted(patient_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            dcm_files = list(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            desc = getattr(ds, "SeriesDescription", "N/A")
            print(f"  {series_dir.name[:20]}...  desc='{desc}'  n_slices={len(dcm_files)}")


def explore_all(root: Path, patients):
    for p in patients:
        explore_patient(root / p)


# --------------------------------------------------------------------------
# 2. Select a series and load the volume
# --------------------------------------------------------------------------

def find_series_dir(patient_dir: Path, keyword: str = "full dose images") -> Path:
    """Find the series directory whose SeriesDescription contains `keyword`.

    Matching is case-insensitive because the collection is not homogeneous:
    most patients have 'Full Dose Images', while others have
    'Full Dose Projections' with an uppercase P.
    """
    for study_dir in patient_dir.iterdir():
        if not study_dir.is_dir():
            continue
        for series_dir in study_dir.iterdir():
            if not series_dir.is_dir():
                continue
            dcm_files = list(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue
            ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            desc = getattr(ds, "SeriesDescription", "").lower()
            if keyword in desc:
                return series_dir
    raise FileNotFoundError(
        f"No series containing '{keyword}' found in {patient_dir}"
    )


def load_series_volume(series_dir: Path) -> np.ndarray:
    """Load all slices in a series, sorted by physical position (Z axis),
    and convert them to Hounsfield units using RescaleSlope/RescaleIntercept."""
    files = list(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    volume = volume * slope + intercept
    return volume  # (n_slices, H, W) in HU


# --------------------------------------------------------------------------
# 3. Preprocessing: windowing, normalization, optional resizing
# --------------------------------------------------------------------------

def load_mayo_slices(
    patient_dir: Path,
    hu_window=DEFAULT_HU_WINDOW,
    size: int | None = None,
) -> np.ndarray:
    """Return a patient's 'Full Dose Images' slices, normalized to [0, 1].

    If `size` is None, keep the native resolution (512x512 for this collection).
    If `size` is given, resize each slice to (size, size).
    """
    series_dir = find_series_dir(patient_dir, "full dose images")
    volume = load_series_volume(series_dir)  # (N, H, W) en HU

    lo, hi = hu_window
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)

    if size is None:
        return volume.astype(np.float32)

    if sk_resize is None:
        raise ImportError(
            "scikit-image is required for resizing: pip install scikit-image"
        )

    out = np.zeros((volume.shape[0], size, size), dtype=np.float32)
    for i in range(volume.shape[0]):
        out[i] = sk_resize(volume[i], (size, size), anti_aliasing=True)
    return out


# --------------------------------------------------------------------------
# 4. Build the .npy cache (one file per patient)
# --------------------------------------------------------------------------

def build_cache(
    root: Path,
    patients,
    cache_dir: Path,
    hu_window=DEFAULT_HU_WINDOW,
    size: int | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in patients:
        cache_file = cache_dir / f"{p}.npy"
        if cache_file.exists():
            print(f"{p}: cache already exists ({cache_file}), skipping")
            continue
        slices = load_mayo_slices(root / p, hu_window=hu_window, size=size)
        # write atomically: a crash/interrupt/out-of-space event during
        # np.save must never leave a truncated *.npy under the final name,
        # since that later fails with a confusing "cannot reshape" error.
        # (tmp name must already end in ".npy" -- np.save appends it otherwise)
        tmp_file = cache_dir / f"{p}.tmp.npy"
        np.save(tmp_file, slices)
        os.replace(tmp_file, cache_file)
        print(f"{p}: {slices.shape} -> {cache_file}")


# --------------------------------------------------------------------------
# 5. PyTorch dataset - connection point for the existing pipeline
# --------------------------------------------------------------------------

class MayoSliceDataset(Dataset):
    """Dataset of real Mayo Clinic CT slices, one entry = one 2D slice.

    Directly replace a seed-based phantom generator:
    instead of `phantom = generate_from_seed(seed)`, use
    `u_true = mayo_dataset[i]` to obtain a normalized (1, H, W) tensor.
    """

    def __init__(self, patients, cache_dir=DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.index = []  # list of (patient, slice_idx)
        self._volumes = {}

        for p in patients:
            cache_file = self.cache_dir / f"{p}.npy"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Cache missing for {p}: {cache_file}. "
                    f"Run build_cache() or `python mayo_dataset.py --build-cache` first."
                )
            try:
                vol = np.load(cache_file)
            except ValueError as e:
                raise ValueError(
                    f"Corrupted cache for {p} ({cache_file}, "
                    f"{cache_file.stat().st_size} bytes on disk): {e}. "
                    f"The file was likely truncated during a copy or sync. "
                    f"Delete it and rerun "
                    f"`python data/mayo_dataset.py --build-cache --patients {p}` "
                    f"to rebuild it from the DICOM sources."
                ) from e
            self._volumes[p] = vol
            self.index.extend([(p, i) for i in range(vol.shape[0])])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        patient, i = self.index[idx]
        img = self._volumes[patient][i]  # (H, W), already normalized [0, 1]
        return torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract the Mayo Clinic LDCT dataset")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                         help="Root directory for downloaded data (idc-index)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                         help="Output directory for the .npy cache")
    parser.add_argument("--patients", nargs="+",
                         default=DEFAULT_TRAIN_PATIENTS + DEFAULT_TEST_PATIENTS,
                         help="Patients to process (e.g. L004 L006 L012)")
    parser.add_argument("--size", type=int, default=None,
                         help="Resize dimension (default: native resolution, e.g. 512)")
    parser.add_argument("--hu-min", type=float, default=DEFAULT_HU_WINDOW[0])
    parser.add_argument("--hu-max", type=float, default=DEFAULT_HU_WINDOW[1])
    parser.add_argument("--explore", action="store_true",
                         help="Display available series per patient without loading data")
    parser.add_argument("--build-cache", action="store_true",
                         help="Build the .npy cache for the selected patients")
    args = parser.parse_args()

    if args.explore:
        explore_all(args.root, args.patients)
        return

    if args.build_cache:
        build_cache(
            root=args.root,
            patients=args.patients,
            cache_dir=args.cache_dir,
            hu_window=(args.hu_min, args.hu_max),
            size=args.size,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()