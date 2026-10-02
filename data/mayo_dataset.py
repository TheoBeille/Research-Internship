"""
Mayo Clinic low-dose CT dataset (LDCT-and-Projection-data, downloaded with
mayo_download.py).

Build the cache once (one .npy file per patient, slices normalised to [0, 1]):

    python data/mayo_dataset.py                # native 512x512
    python data/mayo_dataset.py --size 128     # resized variant

Then use build_train_test_data_mayo() to get the training / test instances.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path("./data/mayo_data/ldct_and_projection_data")
DEFAULT_CACHE_DIR = Path("./data/mayo_cache_512")
DEFAULT_PATIENTS = ["L004", "L006", "L012", "L019", "L014"]
HU_WINDOW = (-160, 240)     # standard abdominal window


# ============================================================
# DICOM -> .npy cache
# ============================================================

def find_series_dir(patient_dir, keyword="full dose images"):
    """Series directory whose SeriesDescription contains `keyword`
    (case-insensitive)."""
    for series_dir in Path(patient_dir).glob("*/*"):
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        header = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        if keyword in getattr(header, "SeriesDescription", "").lower():
            return series_dir
    raise FileNotFoundError(f"No series containing '{keyword}' found in {patient_dir}")


def load_mayo_slices(patient_dir, size=None):
    """Full-dose slices of one patient, sorted along z, windowed and
    normalised to [0, 1]. Shape (n_slices, H, W); resized if `size` is given."""
    series_dir = find_series_dir(patient_dir)
    slices = [pydicom.dcmread(f) for f in series_dir.glob("*.dcm")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # raw values -> Hounsfield units
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)

    lo, hi = HU_WINDOW
    volume = (np.clip(volume, lo, hi) - lo) / (hi - lo)

    if size is not None:
        from skimage.transform import resize
        volume = np.stack([resize(s, (size, size), anti_aliasing=True) for s in volume])
    return volume.astype(np.float32)


def build_cache(root, patients, cache_dir, size=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in patients:
        cache_file = cache_dir / f"{p}.npy"
        if cache_file.exists():
            print(f"{p}: cache already exists, skipping")
            continue
        slices = load_mayo_slices(Path(root) / p, size=size)
        # write to a temporary file first so that an interrupted run never
        # leaves a truncated file under the final name
        tmp_file = cache_dir / f"{p}.tmp.npy"
        np.save(tmp_file, slices)
        os.replace(tmp_file, cache_file)
        print(f"{p}: {slices.shape} -> {cache_file}")


# ============================================================
# Dataset
# ============================================================

class MayoSliceDataset(Dataset):
    """One entry = one 2D slice, as a (1, H, W) tensor in [0, 1]."""

    def __init__(self, patients, cache_dir=DEFAULT_CACHE_DIR):
        slices = []
        for p in patients:
            cache_file = Path(cache_dir) / f"{p}.npy"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Cache missing for {p}: {cache_file}. "
                    f"Run `python data/mayo_dataset.py` first.")
            slices.append(np.load(cache_file))
        self.slices = np.concatenate(slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return torch.from_numpy(self.slices[idx]).float().unsqueeze(0)


def build_train_test_data_mayo(train_patients, test_patients, params, device,
                               n_angles=180, cache_dir=DEFAULT_CACHE_DIR,
                               noise_level=0.0, max_train_slices=None,
                               max_test_slices=None):
    """
    Returns (train_data, test_data), two lists of tuples

        (initial_state, clean, functions)

    one per slice. Never put the same patient in both lists: neighbouring
    slices of one patient are strongly correlated.
    """
    from Algo_setuptorch import get_setup, build_algo_functions

    def build_split(patients, max_slices):
        dataset = MayoSliceDataset(patients, cache_dir)
        n = len(dataset) if max_slices is None else min(max_slices, len(dataset))
        # evenly spaced slices, so that all patients and body regions are covered
        indices = np.linspace(0, len(dataset) - 1, n).round().astype(int)
        data = []
        for i in indices:
            setup = get_setup(params.size, n_angles=n_angles, seed=int(i),
                              noise_level=noise_level, device=device,
                              phantom_array=dataset[int(i)].squeeze(0).numpy())
            functions = build_algo_functions(setup, params)
            data.append((setup["initial_state"], setup["phantom"], functions))
        return data

    return (build_split(train_patients, max_train_slices),
            build_split(test_patients, max_test_slices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Mayo .npy cache")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    parser.add_argument("--size", type=int, default=None,
                        help="resize slices to size x size (default: native 512)")
    args = parser.parse_args()
    build_cache(args.root, args.patients, args.cache_dir, args.size)
