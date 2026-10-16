# LP outputs for the 1,820-pair deficit grid

**The grid.**
- ρ = (a, b, 0)/8 with a, b ≥ 0, a² + b² ≤ 64 and a + b > 8. These are 13 equatorial states.
- σ = (c, d, e)/8 with c, d, e > 0, c² + d² + e² ≤ 64 and c + d + e > 8. These are 140 states, all with s(σ) = +1.

Every pair is non-free.

**The setup.** The frame is the Wootters product frame. Single-qubit phase points are ordered (q, p) = (0,0), (0,1), (1,0), (1,1), and the two-qubit index is 4·i₁ + i₂, with the first qubit as the row. The single-qubit values C(ρ) and C(σ) come from the closed form ½ max(0, ‖r‖₁ − 1).

## How the values were obtained

1. **Floating-point solve.** C(ρ⊗σ) was solved with HiGHS dual simplex (SciPy) over all 60 two-qubit stabilizer states.
2. **Rounding to rationals.** The optimal mixture and the dual witness were rounded to rationals.
3. **Exact re-check.** Each pair was re-checked in exact arithmetic:
   - **Mixture:** the weights are non-negative and sum to 1.
   - **Witness:** every entry satisfies |S_α| ≤ 1.
   - **Matching values:** ‖W_target − Σᵢ wᵢ Wᵢ‖₁ equals ⟨S, W_target⟩ − maxᵢ ⟨S, Wᵢ⟩, where the max runs over all 60 stabilizer states.

All 1,820 pairs certify, so every C(ρ⊗σ) in the files is exact. `check_certificates.py` repeats the check independently: it uses no LP and no floating point, and it rebuilds the stabilizer Wigner vectors from the stabilizer groups.

## Files

| File | Contents |
|---|---|
| `lp_1820_results.csv` | One row per pair, described below. |
| `lp_1820_certificates.json` | The same certificates in machine-readable form. |
| `stabilizer_states.csv` | For each of the 60 stabilizer states: its index, its signed stabilizer group, and its exact Wigner vector. These indices are the ones used in `active_mixture`. |
| `lp_1820.py` | Regenerates everything. Takes about 2 minutes. |
| `check_certificates.py` | Exact verification only. Takes seconds. |

Each row of `lp_1820_results.csv` contains:
- the integers a, b, c, d and e;
- the exact values C(ρ), C(σ), C(ρ⊗σ), D and D/C(ρ), plus float columns;
- the free-support value maxᵢ ⟨S, Wᵢ⟩;
- `active_mixture`, as stabilizer index:weight pairs;
- the 16-entry witness S.

## Summary

- **Every pair has a strict deficit.** The minimum is D = 17/512. It is attained by 168 pairs, for example ρ = (2, 7, 0)/8 and σ = (4, 4, 1)/8, so the strict branch holds on the whole grid. This is finite exact evidence, and universal strictness remains open.
- **The factored form fails.** For 82 of the 140 σ, D/C(ρ) changes with ρ.
- **Witnesses.** The optimal witnesses have free support 0 for 1,516 pairs and ½ for the other 304.
- **Entangled stabilizers.** The returned optimal mixture uses an entangled stabilizer in 273 pairs. Optimal mixtures are not unique, so this does not show that entangled stabilizers are needed.
- **Denominators.** The denominators of C(ρ⊗σ) divide 512.
