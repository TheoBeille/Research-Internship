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
utils/                      PSNR, figures, PDHG, helpers of the notebooks
main.py                     training (writes the checkpoint)
1_baselines.ipynb           figures without any trained model (zero / random)
2_learned.ipynb             learned vs. zero vs. random, safeguard ablation
3_pdhg_comparison.ipynb     learned vs. PDHG, time comparison table
```

## Run

```bash
pip install -r requirements.txt
conda install -c astra-toolbox astra-toolbox   # GPU ray transform

python mayo_download.py          # download 5 patients
python data/mayo_dataset.py      # build the .npy cache
python main.py                   # training
# then run the notebooks 1, 2, 3: figures and tables go to plots/
```

## Notes on the method

- **Resolvent.** `(M + γA)⁻¹` is explicit thanks to the preconditioning metric
  `M = γ [[I/τ, −Bᵀ], [−B, I/σ]]`, with `B(u,w) = (∇u − w, E w)` and
  `τσ‖B‖² = 0.95²`. It costs one PDHG-like step (primal step `τ`, dual step
  `σ`), with no inner loop.
- **Primal step.** `τ` is the gradient step on the data term (`‖K‖ = 1`). The
  default is `τ = 1.5` (`Params(primal_step=...)`). The balanced choice
  `τ = σ ≈ 0.3` is about five times slower. A larger `τ` leaves less room to
  the deviations: the safeguard budget is proportional to `4 − γβ̄ − 2λ₀`.
- **γ and β̄.** γ cancels in the resolvent step and only enters through the θ
  coefficients, as the product `γβ̄`. `β̄` is set to the cocoercivity constant
  of `C` in the metric `M` (computed by power iteration, 5% margin), which
  gives `γβ̄ ≈ 1.05 τ`. The condition is `γβ̄ < 4 − 2λ₀`.
- **Zero deviations.** Without deviations the iterates are those of relaxed
  PDHG with relaxation `λ₀` (exactly when the forward–backward map is affine,
  and to three digits in our runs): the schedule `λₙ` only matters once
  deviations are used.
- **Safeguard.** Budget and deviation norms are both measured in the M-norm.
  The network output is normalised and rescaled to use at most a fraction
  `α²ζ` of the budget; a small output gives a small deviation.
- **Memory.** Each unrolled iteration is wrapped in `torch.utils.checkpoint`,
  so activations are recomputed during the backward pass instead of stored
  (about 1.6 GB for 20 iterations at 512×512).
- **Loss.** TGV² objective at the last unrolled iterate (`LOSS` in `main.py`;
  the KKT residual `‖(A + C)(x_T)‖` and the error to the ground truth are
  also available). The number of unrolled iterations is drawn between `T`
  and `2T` at every step.
