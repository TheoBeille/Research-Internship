import numpy as np
import torch

from Algo_setuptorch import get_setup, build_algo_functions


def load_sample(size, seed, params, device, n_angles=180,
                noise_range=(0.0, 0.1)):
    """
    Build one tomographic sample.

    Returns:
        initial_state : back-projection A^T y   
        clean         : ground-truth phantom   
        functions     
    """
    rng = np.random.default_rng(seed)
    noise_level = float(rng.uniform(*noise_range))
    
    setup = get_setup(size, n_angles=n_angles, seed=seed,
                      noise_level=noise_level, device=device)

    functions = build_algo_functions(setup, params)

    initial_state = setup["initial_state"].to(device)   # [1,1,H,W] back-projection
    clean         = setup["phantom"].to(device)         # [1,1,H,W] ground truth

    return initial_state, clean, functions


def build_dataset(size, seeds, params, device, n_angles=180,
                  noise_range=(0.0, 0.1)):
    """Full in-memory dataset as a list of (initial_state, clean, functions)."""
    return [
        load_sample(size, seed=s, params=params, device=device,
                    n_angles=n_angles, noise_range=noise_range)
        for s in seeds
    ]


def split_seeds(train_seeds, test_seeds):
    return list(train_seeds), list(test_seeds)


def build_train_test_data(train_seeds, test_seeds, params, device,
                          n_angles=180, noise_range=(0.0, 0.1)):
    """Convenience: build both train and test datasets."""
    size = params.size
    train_data = build_dataset(size, seeds=train_seeds, params=params,
                               device=device, n_angles=n_angles,
                               noise_range=noise_range)
    test_data = build_dataset(size, seeds=test_seeds, params=params,
                              device=device, n_angles=n_angles,
                              noise_range=noise_range)
    return train_data, test_data
