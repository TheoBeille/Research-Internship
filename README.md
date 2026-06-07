# TGV² Tomography — Forward–Backward Splitting with Learned Deviations

This project takes a **second-order Total Generalized Variation (TGV²)** image
reconstruction problem, rewrites it as a **monotone inclusion**, solves it with
a **Forward–Backward splitting** algorithm *with history and deviations*



## 1. The mathematical problem

We reconstruct an image `u` from a (noisy) sinogram `y = A·u_true + noise`,
where `A` is the parallel-beam ray transform. The variational problem is the
TGV² model

```
min_{u,w}  ½‖A u − y‖²  +  α₁‖∇u − w‖₁  +  α₂‖E w‖₁
```

with `∇` the forward gradient, `E` the symmetrised gradient, and anisotropic
(componentwise) ℓ¹ norms. Introducing the dual variables `p` (for `∇u − w`) and
`q` (for `E w`), the optimality conditions form a **monotone inclusion**

```
0 ∈ A_op(x) + C_op(x),   x = (u, w, p, q)
```

where:

- **`C`** is the single-valued, cocoercive **forward (data) operator**
  `C(x) = (Aᵀ(A u − y), 0, 0, 0)`. All the measured data enters here.
- **`A_op`** is the maximally monotone part handled implicitly by the
  **resolvent** `R_A = (I + γ A_op)⁻¹`, which is the TGV prox/projection step.

One iteration of the algorithm (history term + deviations `u_n, v_n`) lives in
`algorithm/fbs_step.py::one_step`. With `u_n = v_n = 0` it is exactly the
provably-convergent base algorithm; the network only chooses `u_n, v_n` within a
safeguarded budget so convergence is preserved.

---

## 2. Repository layout

```
Algo_setuptorch.py        # CORE. Builds the problem in ODL, wraps it in torch:
                          #   - RayTransform A (normalised to ‖A‖=1)
                          #   - TGV operators (grad, E and adjoints)
                          #   - phantom, data y, back-projection init
                          #   - Params (γ, λ, α, ζ, ...)
                          #   - build_algo_functions: resolvent R_A (FISTA),
                          #     C, KKT residual, deviation budget δ
pdhg_tomography_tgv.py    # Independent PDHG (Chambolle–Pock) solver for the SAME
                          #   objective → used to compute the reference u*
reference.py              # Precompute / load the high-accuracy reference u*
data/dataset.py           # Build tomographic samples (seeds → instances)
algorithm/
  fbs_step.py             # one_step: a single FBS iteration with deviations
  unrolled_model.py       # UnrolledFBS: unrolls T steps, calls the net, safeguards
  normalization.py        # block-norm helpers used by the safeguarding
models/deviation_net.py   # DeviationNet: CNN that outputs the raw deviations
training/
  train.py                # training loop (AdamW + cosine schedule)
  loss.py                 # trajectory loss helper
run.py                    # run_zero (acceptance test), run_learned, run_random
plots.py                  # convergence / training figures (+ O(1/t), O(1/t²) lines)
PSNR.py                   # PSNR metric vs the phantom
main.py                   # end-to-end script: setup → run_zero → train → figures
main.ipynb                # step-by-step visual notebook (what you actually run)
references/               # saved u* tensors (u_ref_0_02.pt, u_ref_0_2.pt)
checkpoints/, *.pt        # saved model weights
plots/                    # output figures (PDF)
```

---

## 3. Installation

```bash
pip install -r requirements.txt
# ASTRA is best installed via conda (not reliably on PyPI):
conda install -c astra-toolbox astra-toolbox
```

Core requirements: `numpy`, `torch`, `matplotlib`, `odl`, `scikit-image`.
The ray transform backend is selected automatically with the fallback
`astra_cuda → astra_cpu → skimage`, so it runs on CPU without a GPU (just
slower).

---

## 4. How to run

### Step 1 — compute the reference solution `u*` (once)

The convergence/distance plots compare against a high-accuracy minimiser of the
*same* objective, obtained by running PDHG far:

```bash
python reference.py
```

This saves `references/u_ref_*.pt`. **The `SEED` and `NOISE` in `reference.py`
must match the ones you use in `main.py` / `main.ipynb`**, otherwise `y` differs
and the distance-to-reference is meaningless. (Current defaults:
`SEED = 1000`, `NOISE = 0.2`, `REF_ITERS = 5000`.)

### Step 2 — run the full pipeline

```bash
python main.py            # setup → run_zero (acceptance test) → train → figures
# or, for the visual, cell-by-cell version:
jupyter notebook main.ipynb
```

`main.py` does, in order:

1. Build train/test instances (`build_train_test_data`).
2. **`run_zero` (T = 100)** — the acceptance test. Prints the KKT residual
   dropping and the reconstruction PSNR. This must converge on its own.
3. Train `UnrolledFBS` (T = 10 unrolled steps, 70 epochs, lr = 1e-3).
4. Save a checkpoint and produce `run_learned` vs `run_zero` convergence plots.

The KKT residual `kkt_residual_norm(x)` (in `Algo_setuptorch.py`) is the single
source of truth for "has it converged".

---

## 5. Switching denoising ↔ tomography

The switch is a **single line** in `Algo_setuptorch.py::get_setup`:

```python
A = _make_ray_transform(U, n_angles)   # TOMOGRAPHY (current)
# A = odl.IdentityOperator(U)          # DENOISING
```

Everything else adapts automatically because the code is written in terms of `A`:

- **Data**: `y = A(phantom) + noise`. With `A = I` this is `phantom + noise`
  (denoising); with the ray transform it is a noisy sinogram.
- **Forward operator**: `C(x) = (Aᵀ(A u − y), 0, 0, 0)`. With `A = I` this
  collapses to `u − y`, the usual denoising data gradient.
- **Initial point**: `init = Aᵀy` — the back-projection for tomography, simply
  `y` for denoising. (Note: `run_zero` starts from `x = 0` regardless; the data
  is carried inside `C`, so the start does not change the limit.)
- **Normalisation**: `A` is rescaled to `‖A‖ = 1` by power iteration, so
  `β̄ = ‖A‖² = 1` in both cases — the step-size regime is identical.

So to do denoising: comment the `_make_ray_transform` line, uncomment the
`IdentityOperator` line, recompute the reference with `A = I`, and rerun.

---

## 6. Key parameters and numerical conditions

Set in `Params` and finalised in `build_algo_functions`:

- `lam0 = 0.9`, `lam(n) = lam0·(1+n)^0.3` — relaxation/averaging sequence.
- `beta_bar = max(1.0, ‖A‖²) = 1` after normalisation.
- **Step size** `γ` is *not* a free guess: it is set to
  `γ = gamma_safety · 2/(lam0·beta_bar) = 0.85 · 2/0.9 ≈ 1.89`,
  i.e. just inside the cocoercivity bound `γ < 2/(lam0·β̄)` of the forward step.
- `alpha1 = alpha2 = 0.1` — TGV ℓ∞-ball radii (the duals are clamped to these).
- `zeta = 0.9` — fraction of the deviation budget `δ` the network may use; the
  safeguard rescales the raw network output so `‖deviation‖² ≤ ζ·δ`.

### The resolvent `R_A`

`R_A` has no closed form for TGV², so it is solved **inner-iteratively with
FISTA** (`resolvent_A`, strongly-convex variant, `max_iter = 100`,
`tol = 1e-7`). Its Lipschitz constant is `L = 1 + (γ‖B‖)²` with `B` the TGV
operator; the inner step is `step_safety/L`. This is numerically stable and does
**not** require `γ‖B‖ < 1` (FISTA handles any `γ`), which is why a large outer
`γ ≈ 1.89` is fine.

---

## 7. Outputs

- `checkpoints/`, `checkpoint_tomo_*.pt` — model weights.
- `references/u_ref_*.pt` — reference minimisers (named by noise level).
- `plots/*.pdf` — convergence and training figures.

> Cross-check: `pdhg_tomography_tgv.py` solves the *same* objective by a
> completely independent method (PDHG). `run_zero` and PDHG must converge to the
> same image — comparing PSNR and the reconstruction is the cleanest validation
> that the inclusion is set up correctly.