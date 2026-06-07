# TGV² tomography — Forward–Backward with learned deviations (ODL)

Same project layout as the denoising code (`NN_claude`), adapted to
**tomography** (parallel-beam Radon transform, ODL + ASTRA). The TGV² problem is
written as a monotone inclusion `0 ∈ A_op(x) + C_op(x)` and solved by
Algorithm 1 (Forward–Backward with history and deviations); the deviations are
then learned by a network to accelerate convergence.

## Layout (flat, like your version)

```
Algo_setuptorch.py      # ODL setup: RayTransform, phantom, C, resolvent, KKT, Params
data/dataset.py         # tomographic instances (init = back-projection, clean = phantom)
algorithm/              # fbs_step (one iteration), unrolled_model, normalization
models/deviation_net.py # network producing the deviations
training/{train,loss}.py
run.py                  # run_zero (baseline) and run_learned
plots.py, PSNR.py       # figures and metric
main.py                 # end-to-end training
main.ipynb              # visualization notebook
legacy/                 # old code + first package (archived)
```

## What changes vs denoising (and what was wrong in the first tomo attempt)

1. `A = RayTransform(U, geometry)` instead of `IdentityOperator`.
2. data term `C(x) = (Aᵀ(A u − y), 0, 0, 0)` instead of `u − noisy`.
3. the data `y` lives in the sinogram space; the algorithm starts from the
   back-projection `u₀ = Aᵀy`.

Two numerical conditions, set automatically in `build_algo_functions`:

- **Contractive resolvent**: `γ‖B‖ < 1` (B = TGV operator). We set
  `γ = 0.7/‖B‖`.
- **Cocoercivity of the data term**: `γ < 2/(λ₀β̄)` with `β̄ ≥ ‖A‖²`. We
  **normalize the Radon transform to `‖A‖ = 1`** (power iteration), so
  `β̄ = 1` — the same well-conditioned regime as denoising.

`‖A‖` and `‖B‖` are estimated at build time. The TGV weights `α1 = α2 = 5e-4`
give a faithful reconstruction after normalization.

## Run

```bash
# full training (and baseline + figures)
python main.py

# step-by-step visualization
jupyter notebook main.ipynb
```

The notebook shows: phantom, sinogram, back-projection; `‖A‖`, `‖B‖`, `γ‖B‖`;
the convergence of `run_zero` (acceptance test); the reconstruction and PSNR;
and the baseline vs learned comparison. The convergence plots also overlay
`O(1/t)` and `O(1/t²)` reference rates.

## Dependencies

`odl`, `astra-toolbox` (conda recommended), `torch`, `numpy`, `matplotlib`. See
`requirements.txt`. The ODL backend falls back automatically
`astra_cuda → astra_cpu → skimage`.

> Convergence proof (ODL-independent): `convergence_proof.pdf` — the same
> `run_zero`/`one_step` logic in NumPy drops the KKT residual by several orders
> of magnitude (reconstruction ~3 %).
