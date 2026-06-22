"""Reference solution for the TGV² tomography problem.

Run once as a script to compute a high-accuracy reference image u* (PDHG run
far) and save it to disk:

    python reference.py

Then load it anywhere without recomputing:

    from reference import load_reference, l2_distance_history
    u_ref = load_reference()
"""

import os
import torch

REF_PATH = os.path.join("references", "u_ref_0_2.pt")

# Problem configuration
SIZE      = 128
N_ANGLES  = 180
NOISE     = 0.2
SEED      = 1000
REF_ITERS = 5000    

def load_reference(file="u_ref.pt", device=None):
    """Load the saved reference image u* as a torch tensor [1,1,H,W]."""
    path = os.path.join("references", file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python reference.py` once to create it.")
    u_ref = torch.load(path, map_location=device or "cpu").float()
    return u_ref


def l2_distance_history(x_hist, u_ref, relative=True):
    """Relative L2 distance ‖u_t − u*‖ / ‖u*‖ at every iteration.

    x_hist : list of iterates, each a list of blocks (image = block 0) or a
             single image tensor.
    """
    import numpy as np
    u_ref = u_ref[:, 0:1, :, :].float()
    denom = u_ref.norm().clamp(min=1e-12) if relative else torch.tensor(1.0)
    out = []
    for x in x_hist:
        u = (x[0] if isinstance(x, (list, tuple)) else x)[:, 0:1, :, :].float()
        out.append(((u.to(u_ref.device) - u_ref).norm() / denom).item())
    return np.asarray(out)


if __name__ == "__main__":
    from NN_tomo.Algo_setuptorch import Params, get_setup
    from NN_tomo.pdhg_tomography_tgv import run_pdhg
    from NN_tomo.PSNR import psnr_history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = Params(size=SIZE)
    setup = get_setup(SIZE, n_angles=N_ANGLES, seed=SEED,
                      noise_level=NOISE, device=device)

    print(f"running PDHG for {REF_ITERS} iterations ...")
    u_ref = run_pdhg(setup, params, niter=REF_ITERS, verbose=True).float()

    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    torch.save(u_ref.cpu(), REF_PATH)

    psnr = psnr_history([u_ref], setup["phantom"])[0]
    print(f"saved reference to {REF_PATH}  |  PSNR vs phantom = {psnr:.2f} dB")