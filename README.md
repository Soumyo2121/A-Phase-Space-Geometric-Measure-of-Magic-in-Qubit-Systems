# Phase-space geometric measure of magic — reproducibility code

Code accompanying

> S. Dutta and Tushar, *A Phase-Space Geometric Measure of Magic in Qubit Systems*, arXiv:2603.20792.

Everything numerical in the paper can be regenerated from this repository.
Each claim is checked automatically, and the notebook ends with a pass/fail ledger.

## Contents

| File | What it does |
|---|---|
| `verification_v4.ipynb` | Recomputes every numerical claim in the paper and checks every closed-form result against linear programming. Saved with the outputs of a full run. |
| `scripts/frame_census.py` | Standalone version of the two-qubit frame census (Remark 4.12): the 2048 Pauli-covariant frames, the ratio classes, and the Mermin–Peres criterion for M₂ = 2. |
| `scripts/fig_dichotomy_regen.py` | Generates Fig. 1 (`fig_dichotomy.pdf`). |
| `scripts/kappa_noise_fig.py` | Generates Fig. 6 (`fig_noise_nb.pdf`). |

## Running

```bash
pip install -r requirements.txt
jupyter notebook verification_v4.ipynb
```

Two flags at the top of the notebook control the cost:

* `RUN_N4` — the four-qubit linear programs over all 36,720 stabilizer states (adds roughly 10–20 minutes).
* `RUN_SLOW` — multistart searches and the per-frame maxima in the frame census (adds roughly 15 minutes).

With both set to `False` the notebook finishes in about two minutes.

## Conventions

Wootters phase-point operators
A₍q,p₎ = ½(I + (−1)ᵖ X + (−1)^(q+p) Y + (−1)^q Z), tensor products for n qubits, and
W_ρ(α) = Tr(ρ A_α)/2ⁿ.
C(ρ) is the ℓ₁ distance from W_ρ to the convex hull of the stabilizer states' Wigner functions, and Γ is the robustness of magic; both are computed by linear programming (SciPy/HiGHS).
Stabilizer states are generated as the orbit of |0…0⟩ under H, S and CNOT, and their counts are checked (6, 60, 1080, 36720).

## Citation

If you use this code, please cite the paper (arXiv:2603.20792).
