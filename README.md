# Phase-space geometric measure of magic: reproducibility code (release v7)

Code accompanying

> S. Dutta, Tushar, H. Ma and W. Zhang, *A Phase-Space Geometric Measure of Magic in Qubit Systems*, arXiv:2603.20792 (version 7, October 2026).

This release regenerates every numerical result and every figure in version 7 of the paper. It also contains the exact certificates for the 1,820-pair deficit grid. Each claim is checked automatically, and the notebook ends with a pass/fail ledger.

## Contents

| Path | What it does |
|---|---|
| `verification_v7.ipynb` | Recomputes every numerical claim in v7. Each closed-form result and explicit certificate is checked against linear programming over the complete stabilizer polytope (floating point, HiGHS). Section 11 runs the two-qubit frame census over all 2048 Pauli-covariant frames. Section 12.3 re-checks the rational certificates in exact arithmetic. Section 12.4 checks the additions in v7. Section 13 writes all figures as PDF. This copy has no outputs. |
| `verification_v7_executed.ipynb` | The same notebook, saved with the outputs of a complete run (`RUN_N4 = RUN_SLOW = True`). |
| `make_figures.py` | Standalone script that writes all figure PDFs to `figures/` in about a minute. Its code is generated from the notebook. |
| `figures/` | The figure PDFs used in the paper. |
| `lp_1820_grid/` | Exact primal and dual certificates for the 1,820 pairs ρ = (a,b,0)/8, σ = (c,d,e)/8 of Observation 5.2. `check_certificates.py` verifies them in exact rational arithmetic in seconds, using no LP and no floating point. `lp_1820.py` regenerates them in about 2 minutes. See `lp_1820_grid/README.md`. |

The Lean 4 formalization of the two-factor equatorial identity and the self-tensor inequality is not part of this repository. It is the file `QubitWignerDistanceTensorRules.lean` at revision `59c09d64d1050c396df54b58b4f410f09f0c25d0` of the trureturing repository (Ref. [31] of the paper).

## Running

```bash
pip install -r requirements.txt
python make_figures.py                      # figures only (~1 minute)
python lp_1820_grid/check_certificates.py   # exact check of the 1,820 certificates (seconds)
jupyter notebook verification_v7.ipynb      # full verification
```

Flags at the top of the notebook:

* `RUN_N4` (default `True`) runs the four-qubit linear programs over all 36,720 stabilizer states.
* `RUN_SLOW` (default `True` in this release) runs the per-frame maxima of the frame census and the multistart searches.
* `MAKE_FIGURES` (default `True`) writes the figure PDFs.

A complete run takes about 20 minutes (143 checks). With `RUN_N4 = RUN_SLOW = False` it takes a few minutes.

## What is exact and what is numerical

Values the paper calls exact are proved analytically or certified individually in exact arithmetic. Everything else comes from floating-point linear programs (HiGHS via SciPy) and is labelled as sampled or numerical. This includes the frame census at eight parameter values, the noise sweep at seven noise levels, and the multistart searches.

## Conventions

* **Phase-point operators.** Wootters frame, A₍q,p₎ = ½(I + (−1)ᵖ X + (−1)^(q+p) Y + (−1)^q Z), with tensor products for n qubits and W_ρ(α) = Tr(ρ A_α)/2ⁿ.
* **The measures.** C(ρ) is the ℓ₁ distance from W_ρ to the convex hull of the stabilizer Wigner functions. Γ is the robustness of magic.
* **Witnesses.** A Hermitian H certifies C(ρ) ≥ 2⁻ⁿ[Tr(Hρ) − max_σ Tr(Hσ)] only when it is normalized so that max_α |Tr(H A_α)| ≤ 2ⁿ (Theorem 3.3). Every witness check enforces this normalization.
* **Stabilizer states.** Generated as the orbit of |0…0⟩ under H, S and CNOT, with counts checked: 6, 60, 1080 and 36720.

## Changes from v6

* **New checks** (Section 12.4 of the notebook):
  * Proposition 7.1, the tensor rate log‖W_ρ‖₁, checked through aᵏ − 1 ≤ C(ρ^⊗k) ≤ aᵏ + 1 for k ≤ 4;
  * the group-averaged commuting witnesses of Remark 8.2;
  * the quantum-net counterexample of Remark 4.16.
* **Grid certificates.** The 1,820-pair certificates are now in `lp_1820_grid/`.
* **Removed files.** The v4 notebook and `scripts/` are removed. The notebook supersedes them, and they remain in the git history.

## Citation

If you use this code, please cite the paper (arXiv:2603.20792).
