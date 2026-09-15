# Learning to Accelerate TGV² Tomographic Reconstruction

**A convolutional network learns the deviations of a forward–backward splitting scheme, reaching in 10 iterations an accuracy that the standard solver needs hundreds of iterations to match — without ever losing the convergence guarantee.**

<p align="center">
  <img src="docs/figures/reconstructions_10it.png" width="100%">
  <br>
  <em>After 10 iterations. Left to right: ground truth, back-projection init, zero-deviation baseline (25.6 dB), <b>learned (29.9 dB)</b>, PDHG (15.6 dB).</em>
</p>

Research internship at **KTH Royal Institute of Technology**, Department of Mathematics
(Jan–Jun 2026), supervised by Prof. Ozan Öktem. → **[Full report (PDF)](docs/report.pdf)**

---

## Results

Parallel-beam CT, 128×128, synthetic TGV phantoms, noise level drawn in [0, 0.1].
All methods minimise the **same** TGV² objective.

| After 10 iterations | PSNR ↑ | Objective gap `f(xₙ) − f*` ↓ |
| --- | --- | --- |
| **Learned deviations** | **29.9 dB** | **~10⁻¹** |
| Zero deviations (plain FBS) | 25.6 dB | ~10⁰ |
| PDHG (Chambolle–Pock) | 15.6 dB | ~10² |

Three things make the claim hold up:

- **The gain comes from learning, not from perturbation.** A random-direction control, given the exact same deviation budget, is *slower* than the baseline.
- **Convergence is never traded away.** A hard safeguard rescales the network output so the Lyapunov condition of the underlying theorem always holds. Run 100× past the training horizon, the learned iteration still converges; the same iteration with the safeguard removed diverges.
- **It transfers.** The network is fully convolutional: trained only on 128×128, it accelerates 512×512 problems with no retraining.

<p align="center">
  <img src="docs/figures/psnr_vs_iterations.png" width="49%">
  <img src="docs/figures/kkt_residual.png" width="49%">
  <br>
  <em>Left: reconstruction PSNR on held-out test seeds. Right: KKT residual, log–log, against the O(1/t) and O(1/t²) references.</em>
</p>

---

## Why this problem

Second-order Total Generalised Variation (TGV²) is a better image model than Total Variation: it preserves edges without turning smooth gradients into flat plateaus (the *staircasing* artefact), which matters for soft tissue in medical images. The reason it is not used in practice is cost — the resulting problem is non-smooth and coupled, first-order solvers converge at O(1/t), and a CT slice may need hundreds of iterations. **The question here is how much accuracy is reachable in the ~10 iterations a real acquisition pipeline can afford.**

---

## How it works

**1. Recast as a monotone inclusion.** Dualising the two ℓ¹ terms turns the TGV² problem into

```
0 ∈ A x + C x,     x = (u, w, p, q)
```

with `C(x) = (Kᵀ(K u − y), 0, 0, 0)` the cocoercive data gradient (proved `1/‖K‖²`-cocoercive) and `A` the sum of a skew-symmetric coupling and the normal cones of the dual constraints (proved maximally monotone). This is exactly the structure required by the forward–backward scheme with history and deviations of Sadeghi, Banert & Giselsson.

**2. The deviations are free.** That scheme injects a pair `(Δ¹ₙ, Δ²ₙ)` at each step. Its theorem guarantees convergence for *any* sequence of deviations satisfying a safeguarding inequality. So the deviations are a steering direction that can be chosen as aggressively as one likes — the perfect thing to learn.

**3. Learn them by unrolling.** T iterations are treated as the layers of a weight-tied network. A 17k-parameter CNN (21 input channels: the iterates, the previous deviations, the data-fidelity gradient → two Conv–InstanceNorm–LeakyReLU blocks → 6 output channels) predicts the raw deviation; a normalisation-and-safeguard layer rescales it to the largest provably admissible magnitude. **The network picks the direction, the safeguard picks the size.** Training back-propagates a loss on the final iterate through all T steps, with the horizon randomised so the rule does not overfit one depth.

**4. The resolvent.** `(I + γA)⁻¹` has no closed form for TGV². Solving it by naive fixed-point iteration requires `γ‖B‖ < 1`, which collapses the outer step size and kills tomography. Reformulating the resolvent's own saddle problem and solving it with FISTA removes that condition entirely (`L = 1 + (γ‖B‖)²` already absorbs the factor), which is what allows the outer `γ ≈ 1.8`.

---

## Quickstart

```bash
pip install -r requirements.txt
# ASTRA is not reliably installable from PyPI:
conda install -c astra-toolbox astra-toolbox
```

The ray-transform backend falls back `astra_cuda → astra_cpu → skimage`, so it runs on CPU (slower) without a GPU.

```bash
# 1. Precompute the high-accuracy reference u* (PDHG, 5000 iterations). Once.
python reference.py

# 2. Full pipeline: build data → acceptance test → train → figures
python main.py
```

> ⚠️ `SEED` and `NOISE` in `reference.py` must match those in `main.py`, otherwise `y` differs and every distance-to-reference number is meaningless. Defaults: `SEED = 1000`, `NOISE = 0.2`.

`main.py` runs, in order: build train/test instances → **`run_zero` (T=100), the acceptance test** → train `UnrolledFBS` (T=10, 70 epochs, AdamW + cosine) → save checkpoint → convergence figures.

The two notebooks reproduce the two training targets studied in the report:
`running_kkt.ipynb` (loss on the monotone-inclusion residual) and `running_TGVobjectif.ipynb` (loss on the TGV² objective). Both work; the report compares them.

**Denoising instead of tomography** is a one-line switch in `Algo_setuptorch.py::get_setup` — replace the ray transform by `odl.IdentityOperator(U)`. Everything else is written in terms of `K` and adapts, including the `‖K‖ = 1` normalisation.

---

## Repository layout

```
Algo_setuptorch.py          # CORE. Problem setup in ODL, wrapped in torch:
                            #   ray transform (normalised to ‖K‖=1), TGV operators,
                            #   phantom + data + init, Params (γ, λ, α, ζ),
                            #   build_algo_functions: FISTA resolvent, C,
                            #   KKT residual, deviation budget δ
reference.py                # High-accuracy reference u* via PDHG
main.py                     # End-to-end: setup → run_zero → train → figures

algorithm/
  fbs_step.py               # one_step: a single FBS iteration with deviations
  unrolled_model.py         # UnrolledFBS: unrolls T steps, calls net, safeguards
  normalization.py          # block-norm helpers used by the safeguard
  run.py                    # run_zero (acceptance test), run_learned, run_random
models/deviation_net.py     # DeviationNet: the 17k-parameter CNN
training/                   # train.py (AdamW + cosine), loss.py
data/                       # dataset.py: seeds → tomographic instances
utils/                      # PSNR, plotting, independent PDHG solver
references/                 # saved u* tensors
trained_models/             # checkpoints
plots/                      # output figures (PDF)

running_kkt.ipynb           # training target: inclusion residual
running_TGVobjectif.ipynb   # training target: TGV² objective
```

**Correctness check.** `utils/` contains an independent PDHG solver for the *same* objective. With deviations switched off, the scheme must converge to the same image as PDHG — it does (36.8 dB vs 36.6 dB, visually indistinguishable). That agreement is the cleanest evidence the inclusion is set up correctly.

---

## Scope and limitations

Stated plainly, because they matter for reading the numbers:

- Training and evaluation use **synthetic TGV phantoms** with artificial noise, not clinical scans. Validating on a real dose-realistic dataset (e.g. Mayo) is the obvious next step.
- The gain is concentrated in the **early regime**. Asymptotically the learned scheme settles onto the baseline rate — by design, since the safeguard ties it to the baseline's guarantees.
- The budget fraction `ζₙ` and the parameter sequences `(γₙ, λₙ)` are fixed by hand. Letting the network allocate the budget across iterations is unexplored.
- Each outer iteration solves the resolvent with ~100 inner FISTA steps, which dominates wall-clock cost. Learning a cheap approximate resolvent would cut the forward/back-projection count — the quantity that actually matters in CT.

---

## References

1. H. Sadeghi, S. Banert, P. Giselsson. *Incorporating history and deviations in forward–backward splitting.* Numerical Algorithms 96 (2024). — the convergence theorem used here.
2. S. Banert, J. Rudzusika, O. Öktem, J. Adler. *Accelerated forward-backward optimization using deep learning.* arXiv:2105.05210 (2021). — the learning strategy this work adapts.
3. K. Bredies, K. Kunisch, T. Pock. *Total generalized variation.* SIAM J. Imaging Sci. 3(3), 2010.
4. A. Chambolle, T. Pock. *A first-order primal-dual algorithm...* JMIV 40(1), 2011. — the PDHG baseline.
5. J. Adler, O. Öktem et al. *ODL: Operator Discretization Library.*

The monotone-inclusion formulation for TGV² tomography, its implementation, the FISTA resolvent, the deviation network and the experiments were built during this internship; the convergence theorem of [1] is the only component reused as is.

## License

MIT
