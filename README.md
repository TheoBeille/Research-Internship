# TGV² tomography — forward–backward splitting with learned deviations

Reconstruct a CT image `u` from a noisy sinogram `y = K u_true + noise` with the
TGV² model

```
min_{u,w}  ½‖K u − y‖²  +  α₁‖∇u − w‖₁  +  α₂‖E w‖₁
```

The optimality conditions are written as a monotone inclusion
`0 ∈ A(x) + C(x)` with `x = (u, w, p, q)` and solved by the forward–backward
splitting with history and deviations of Sadeghi, Banert and Giselsson (2024).
A small network chooses the deviations; a safeguard rescales them so that the
convergence guarantee is kept.

## Layout

```
Algo_setuptorch.py          problem setup: operators (ODL), data, Params,
                            resolvent, safeguard budget, KKT residual, objective
algorithm/fbs_step.py       one iteration, safeguard, unrolled loop
algorithm/unrolled_model.py UnrolledFBS: the unrolled loop with the network
algorithm/run.py            evaluation runs: zero / random / learned deviations
models/deviation_net.py     DeviationNet: the CNN that predicts the deviations
training/train.py           training loop (AdamW, cosine schedule)
data/mayo_dataset.py        Mayo CT slices: DICOM -> .npy cache -> instances
mayo_download.py            download the Mayo dataset
utils/                      PSNR, figures, PDHG reference problem
main.py                     baseline -> training -> comparison
make_paper_figures.ipynb    figures of the paper (run after main.py)
```

## Run

```bash
pip install -r requirements.txt
conda install -c astra-toolbox astra-toolbox   # GPU ray transform

python mayo_download.py          # download 5 patients
python data/mayo_dataset.py      # build the .npy cache
python main.py                   # baseline, training, figures in plots/
```

## Notes on the method

- **Resolvent.** `(M + γA)⁻¹` is explicit thanks to the preconditioning metric
  `M = γ [[I/s, −Bᵀ], [−B, I/s]]`, with `B(u,w) = (∇u − w, E w)` and
  `s = 0.95/‖B‖`. It costs one PDHG-like step, with no inner loop.
- **γ.** It cancels in the resolvent step and only enters through the θ
  coefficients, as the product `γ·β̄`. The condition is `γ·β̄ < 4 − 2λ₀`.
- **Safeguard.** Budget and deviation norms are both measured in the M-norm.
  The network output is normalised and rescaled to use a fraction `α²ζ` of the
  budget.
- **Memory.** Each unrolled iteration is wrapped in `torch.utils.checkpoint`,
  so activations are recomputed during the backward pass instead of stored
  (about 1.6 GB for 20 iterations at 512×512).
- **Loss.** KKT residual `‖(A + C)(x_T)‖` at the last unrolled iteration.
