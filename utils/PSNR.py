import numpy as np


def psnr_history(x_list, clean, max_val=1.0):
    """
    PSNR of each iterate against the clean image [1, 1, H, W].

    Each element of x_list is either an image tensor or a list of blocks
    whose first element is the image.
    """
    hist = []
    for x in x_list:
        image = x[0] if isinstance(x, list) else x
        mse = ((image - clean) ** 2).mean().item()
        hist.append(float(20.0 * np.log10(max_val / np.sqrt(mse))) if mse > 1e-12 else 100.0)
    return hist
