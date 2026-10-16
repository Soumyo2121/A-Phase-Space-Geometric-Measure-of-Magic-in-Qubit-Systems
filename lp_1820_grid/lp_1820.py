#!/usr/bin/env python3
"""
lp_1820.py -- full 60-stabilizer LP for the deficit grid proposed by Ma and Zhang.

Grid: rho = (a,b,0)/8 with a,b >= 0, a^2+b^2 <= 64, a+b > 8          (13 equatorial states)
      sigma = (c,d,e)/8 with c,d,e > 0, c^2+d^2+e^2 <= 64, c+d+e > 8 (140 states, s(sigma) = +1)
      -> 1,820 pairs, all non-free.

For each pair: C(rho), C(sigma) (closed form, Prop. 4.3), C(rho x sigma) by HiGHS dual simplex in the
Wootters frame, the deficit D = (1+C(rho))(1+C(sigma)) - 1 - C(rho x sigma), an optimal stabilizer mixture
and an optimal dual witness S. Each floating-point solution is converted to rationals and re-checked in
exact arithmetic (fractions.Fraction):
  primal: weights >= 0, sum 1, || W_target - sum_i w_i W_i ||_1 = value
  dual:   |S_a| <= 1, <S, W_target> - max_i <S, W_i> = value     (max over all 60 stabilizer states)
A pair is 'certified' when both hold with the same rational value.

Outputs: lp_1820_results.csv, lp_1820_certificates.json, stabilizer_states.csv
Requires numpy, scipy (>= 1.9).
"""
import itertools, json, csv, sys
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # files live next to this script

# ---------------- Wootters frame and the 60 two-qubit stabilizer states ----------------
I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1]).astype(complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]                       # phase-point order (q,p)
A1 = [0.5 * (I2 + (-1)**p * X + (-1)**(q + p) * Y + (-1)**q * Z) for q, p in PTS]
A2 = [np.kron(a, b) for a in A1 for b in A1]                  # index 4*i1 + i2

def stabilizer_states2():
    Hd = (X + Z) / np.sqrt(2); Sg = np.diag([1, 1j])
    gates = [np.kron(Hd, I2), np.kron(I2, Hd), np.kron(Sg, I2), np.kron(I2, Sg),
             np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex),
             np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], complex)]
    def key(v):
        i = np.flatnonzero(np.abs(v) > 1e-9)[0]; v = v * abs(v[i]) / v[i]
        return tuple(np.round(np.r_[v.real, v.imag], 7))
    v0 = np.zeros(4, complex); v0[0] = 1; seen = {key(v0): v0}; front = [v0]
    while front:
        nxt = []
        for v in front:
            for g in gates:
                w = g @ v; k = key(w)
                if k not in seen: seen[k] = w; nxt.append(w)
        front = nxt
    return list(seen.values())

PSI = stabilizer_states2(); assert len(PSI) == 60
LABELS = [a + b for a in 'IXYZ' for b in 'IXYZ'][1:]
def stab_group(v):
    out = []
    for l in LABELS:
        e = np.vdot(v, np.kron(PAULI[l[0]], PAULI[l[1]]) @ v).real
        if abs(abs(e) - 1) < 1e-9: out.append(('+' if e > 0 else '-') + l)
    return out
WSTAB = [[F(round(np.vdot(v, A @ v).real / 4 * 16), 16) for A in A2] for v in PSI]   # exact: multiples of 1/16
for v, w in zip(PSI, WSTAB):
    assert np.allclose([float(x) for x in w], [np.vdot(v, A @ v).real / 4 for A in A2], atol=1e-12)
WS = np.array([[float(x) for x in w] for w in WSTAB])          # 60 x 16

def w1(r):  # exact single-qubit Wigner vector, r rational
    rx, ry, rz = r
    return [F(1, 4) * (1 + (-1)**p * rx + (-1)**(q + p) * ry + (-1)**q * rz) for q, p in PTS]
kron = lambda u, v: [x * y for x in u for y in v]
C1 = lambda r: max(F(0), (sum(abs(x) for x in r) - 1) / 2)    # Prop. 4.3, exact

# ---------------- LP and exact certification ----------------
N, d = WS.shape
c_obj = np.r_[np.zeros(N), np.ones(d)]
A_ub = np.block([[-WS.T, -np.eye(d)], [WS.T, -np.eye(d)]])
A_eq = np.r_[np.ones(N), np.zeros(d)][None]

def solve(wt):
    w = np.array([float(x) for x in wt])
    r = linprog(c_obj, A_ub=A_ub, b_ub=np.r_[-w, w], A_eq=A_eq, b_eq=[1], bounds=(0, None), method='highs-ds')
    assert r.status == 0, r.message
    lam = r.x[:N]; m = r.ineqlin.marginals; S = m[:d] - m[d:]
    if not np.isclose(S @ w - (WS @ S).max(), r.fun, atol=1e-7): S = -S
    return r.fun, lam, S

def rat(x, D):
    return F(x).limit_denominator(D)

def certify(wt, val, lam, S):
    for D in (2**10, 2**14, 2**18, 10**6):
        ws = [rat(x, D) if x > 1e-10 else F(0) for x in lam]
        tot = sum(ws)
        if tot == 0: continue
        if tot != 1:                                        # renormalize the largest weight to make the sum exact
            j = max(range(N), key=lambda i: ws[i]); ws[j] += 1 - tot
            if ws[j] < 0: continue
        mix = [sum(ws[i] * WSTAB[i][a] for i in range(N) if ws[i]) for a in range(d)]
        primal = sum(abs(x - y) for x, y in zip(wt, mix))
        Sq = [max(F(-1), min(F(1), rat(x, D))) for x in S]
        dual = sum(s * t for s, t in zip(Sq, wt)) - max(sum(s * x for s, x in zip(Sq, w)) for w in WSTAB)
        if primal == dual:
            return primal, ws, Sq
    return None

rho_grid = [(a, b) for a in range(9) for b in range(9) if a*a + b*b <= 64 and a + b > 8]
sig_grid = [(c, dd, e) for c in range(1, 9) for dd in range(1, 9) for e in range(1, 9)
            if c*c + dd*dd + e*e <= 64 and c + dd + e > 8]
assert (len(rho_grid), len(sig_grid)) == (13, 140)

rows, certs, ncert = [], [], 0
for k, ((a, b), (c, dd, e)) in enumerate(itertools.product(rho_grid, sig_grid)):
    r = (F(a, 8), F(b, 8), F(0)); s = (F(c, 8), F(dd, 8), F(e, 8))
    wt = kron(w1(r), w1(s)); Cr, Cs = C1(r), C1(s)
    val, lam, S = solve(wt)
    cert = certify(wt, val, lam, S)
    Cj = cert[0] if cert else None
    Dfl = float((1 + Cr) * (1 + Cs) - 1) - val
    Dq = (1 + Cr) * (1 + Cs) - 1 - Cj if cert else None
    ncert += cert is not None
    active = [(i, w) for i, w in enumerate(cert[1]) if w] if cert else [(i, x) for i, x in enumerate(lam) if x > 1e-10]
    rows.append({'id': k, 'a': a, 'b': b, 'c': c, 'd': dd, 'e': e,
                 'C_rho': str(Cr), 'C_sigma': str(Cs),
                 'C_joint': str(Cj) if cert else '', 'C_joint_float': f'{val:.12f}',
                 'D': str(Dq) if cert else '', 'D_float': f'{Dfl:.12f}',
                 'D_over_C_rho': str(Dq / Cr) if cert else '', 'D_over_C_rho_float': f'{Dfl / float(Cr):.12f}',
                 'exact_certified': cert is not None,
                 'free_support': str(max(sum(s_ * x for s_, x in zip(cert[2], w)) for w in WSTAB)) if cert else '',
                 'active_mixture': ';'.join(f'{i}:{w}' for i, w in active),
                 'witness_S': ' '.join(str(x) for x in cert[2]) if cert else ' '.join(f'{x:.6f}' for x in S)})
    certs.append({'id': k, 'r_rho': [a, b, 0], 'r_sigma': [c, dd, e], 'denominator': 8,
                  'C_joint': str(Cj) if cert else None, 'certified': cert is not None,
                  'mixture': {str(i): str(w) for i, w in active} if cert else None,
                  'witness_S': [str(x) for x in cert[2]] if cert else None})
    if k % 200 == 0: print(k, file=sys.stderr)

with open(os.path.join(HERE, 'lp_1820_results.csv'), 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
json.dump({'frame': 'Wootters product frame, phase points (q,p)=00,01,10,11; two-qubit index 4*i1+i2',
           'stabilizer_index': 'see stabilizer_states.csv', 'pairs': certs},
          open(os.path.join(HERE, 'lp_1820_certificates.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'stabilizer_states.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['index', 'stabilizer_group', 'wigner_vector'])
    for i, (v, w) in enumerate(zip(PSI, WSTAB)):
        wr.writerow([i, ' '.join(stab_group(v)), ' '.join(str(x) for x in w)])
print(f'{len(rows)} pairs, {ncert} exactly certified')
