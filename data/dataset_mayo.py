"""
data/dataset_mayo.py

Équivalent de data/dataset.py::build_train_test_data, mais utilisant de vraies
coupes CT Mayo Clinic (chargées via mayo_dataset.MayoSliceDataset) à la place
des phantoms TGV synthétiques générés par seed.

Format de sortie identique à l'original : train_data / test_data sont des
listes de tuples (initial_state, clean, functions), donc main.py n'a besoin
que de changer l'import + l'appel à cette fonction, rien d'autre.
"""

from Algo_setuptorch import get_setup, build_algo_functions
from data.mayo_dataset import MayoSliceDataset


def build_train_test_data_mayo(
    train_patients,
    test_patients,
    params,
    device,
    n_angles=180,
    cache_dir="./mayo_cache_512",
    noise_level=0.0,
    max_train_slices=None,
    max_test_slices=None,
    verbose=True,
):
    """
    train_patients, test_patients : listes d'IDs patients (ex ["L004", "L006"])
        -- ne JAMAIS mettre le même patient dans les deux listes (fuite de données,
        des coupes voisines d'un même patient sont très corrélées).
    params : instance de Algo_setuptorch.Params (params.size doit correspondre
        à la résolution du cache, ex 512).
    max_train_slices / max_test_slices : limite le nombre de coupes utilisées
        (chaque coupe déclenche un get_setup() complet avec power-iteration sur
        norm_A/norm_B (~30-60 itérations) + un RayTransform à 1000x1000 -- coûteux
        à 512x512. Utile pour un premier run rapide avant de tout lancer sur
        l'ensemble des coupes disponibles.
    """
    train_ds = MayoSliceDataset(train_patients, cache_dir=cache_dir)
    test_ds = MayoSliceDataset(test_patients, cache_dir=cache_dir)

    if verbose:
        print(f"[mayo] train: {len(train_patients)} patients, "
              f"{len(train_ds)} coupes disponibles")
        print(f"[mayo] test:  {len(test_patients)} patients, "
              f"{len(test_ds)} coupes disponibles")

    def build_split(ds, max_slices, tag):
        n = len(ds) if max_slices is None else min(max_slices, len(ds))
        data = []
        for i in range(n):
            img = ds[i].squeeze(0).numpy()  # (size, size), normalisé [0,1]
            setup = get_setup(
                size=params.size,
                n_angles=n_angles,
                seed=i,                     # utilisé seulement pour le bruit
                noise_level=noise_level,
                device=device,
                phantom_array=img,
            )
            functions = build_algo_functions(setup, params)
            data.append((setup["initial_state"], setup["phantom"], functions))
            if verbose and (i + 1) % max(1, n // 10) == 0:
                print(f"[mayo] {tag}: {i + 1}/{n} instances construites")
        return data

    train_data = build_split(train_ds, max_train_slices, "train")
    test_data = build_split(test_ds, max_test_slices, "test")
    return train_data, test_data