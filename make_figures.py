#!/usr/bin/env python3
"""
make_figures.py -- regenerate every figure of
"A Phase-Space Geometric Measure of Magic in Qubit Systems" (revised version) as vector PDFs.

    python make_figures.py

writes figures/fig_dichotomy.pdf, fig_circuits.pdf, fig_logical_bloch.pdf, fig_wigner_ry.pdf,
fig_wigner_rx.pdf, fig_kappa_nb.pdf, fig_noise_nb.pdf and fig_pca_nb.pdf (about one minute),
and checks each figure's numbers against the paper. Upload the PDFs to Overleaf.

Generated from verification_v7.ipynb (Section 13); the code is identical to the notebook's.
Requires numpy, scipy (>= 1.9, HiGHS) and matplotlib.
"""

# ---- flags and ledger
RUN_N4       = False   # four-qubit LPs (Sections 1.1, 10)
RUN_SLOW     = True    # frame-census maxima and multistart searches (Sections 4, 11)
MAKE_FIGURES = True    # write figures/fig_*.pdf (Section 13)

import matplotlib
matplotlib.use("Agg")
import numpy as np, itertools, time, collections, os, sys
from scipy.optimize import linprog, minimize
from scipy import sparse
np.set_printoptions(precision=6, suppress=True, linewidth=110)
T0 = time.time()

LEDGER = []
def check(name, ok, detail=""):
    # Record a check in the ledger and print it.
    ok = bool(ok)
    LEDGER.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  —  {detail}" if detail else ""))

def close(a, b, tol=1e-7):
    return abs(a - b) <= tol

# ---- machinery (Section 1 of the notebook)
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.diag([1, -1]).astype(complex)
Hd = (X + Z) / np.sqrt(2)
Sg = np.diag([1, 1j]).astype(complex)
P1 = [I2, X, Y, Z]
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}

def kron_all(ops):
    out = np.array([[1]], dtype=complex)
    for o in ops:
        out = np.kron(out, o)
    return out

def pauli(label):
    return kron_all([PAULI[c] for c in label])

def A1q(q, p, a=(1, 1, 1)):
    # Single-qubit phase-point operator; a=(a_x,a_y,a_z) selects one of the 8 valid origins.
    return 0.5 * (I2 + a[0] * (-1)**p * X + a[1] * (-1)**(q + p) * Y + a[2] * (-1)**q * Z)

PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
A1 = [A1q(q, p) for q, p in PTS]                        # order (q,p) = 00, 01, 10, 11

def phase_points(n, single=None):
    single = A1 if single is None else single
    return [kron_all(t) for t in itertools.product(single, repeat=n)]

def wigner_matrix(Aops):
    # Matrix M with W = Re(M @ vec(rho)) / dim.
    return np.array([A.T.reshape(-1) for A in Aops])

def proj(v):
    v = np.asarray(v, dtype=complex); v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())

def bloch(r):
    return 0.5 * (I2 + r[0] * X + r[1] * Y + r[2] * Z)

def stabilizer_states(n):
    # All pure n-qubit stabilizer states, as the orbit of |0..0> under H, S and CNOT.
    dim = 2**n
    gates = []
    for i in range(n):
        for g in (Hd, Sg):
            ops = [I2] * n; ops[i] = g; gates.append(kron_all(ops))
    for c, t in itertools.permutations(range(n), 2):
        M = np.zeros((dim, dim), dtype=complex)
        for b in range(dim):
            bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[c]: bits[t] ^= 1
            M[int(''.join(map(str, bits)), 2), b] = 1
        gates.append(M)
    def key(v):
        i = np.flatnonzero(np.abs(v) > 1e-9)[0]
        v = v * (abs(v[i]) / v[i])
        return tuple(np.round(np.concatenate([v.real, v.imag]), 7))
    start = np.zeros(dim, dtype=complex); start[0] = 1
    seen = {key(start): start}; frontier = [start]
    while frontier:
        nxt = []
        for v in frontier:
            for g in gates:
                w = g @ v; k = key(w)
                if k not in seen:
                    seen[k] = w; nxt.append(w)
        frontier = nxt
    return np.array(list(seen.values()))

class Magic:
    # C and Gamma for n qubits in a given frame (default: Wootters product frame).
    def __init__(self, n, Aops=None, states=None):
        self.n = n; self.dim = 2**n
        self.Aops = phase_points(n) if Aops is None else Aops
        self.M = wigner_matrix(self.Aops)
        self.psi = stabilizer_states(n) if states is None else states
        R = np.einsum('si,sj->sij', self.psi, self.psi.conj()).reshape(len(self.psi), -1)
        self.Ws = (R @ self.M.T).real / self.dim                 # (N_stab, 4^n)
        self.N, self.d = self.Ws.shape
        Wt = sparse.csr_matrix(self.Ws.T); Id = sparse.identity(self.d, format='csr')
        self.cC = np.r_[np.zeros(self.N), np.ones(self.d)]
        self.AubC = sparse.vstack([sparse.hstack([-Wt, -Id]), sparse.hstack([Wt, -Id])]).tocsr()
        self.AeqC = sparse.csr_matrix(np.r_[np.ones(self.N), np.zeros(self.d)][None, :])
        self.AeqG = sparse.hstack([Wt, -Wt]).tocsr()
    def W(self, rho):
        return (self.M @ np.asarray(rho).reshape(-1)).real / self.dim
    def C(self, rho, witness=False):
        w0 = self.W(rho)
        r = linprog(self.cC, A_ub=self.AubC, b_ub=np.r_[-w0, w0], A_eq=self.AeqC, b_eq=[1],
                    bounds=(0, None), method='highs')
        assert r.status == 0, r.message
        if not witness:
            return r.fun
        m = r.ineqlin.marginals; y = m[:self.d] - m[self.d:]
        for cand in (y, -y):
            if close(cand @ w0 - (self.Ws @ cand).max(), r.fun, 1e-6):
                return r.fun, cand
        return r.fun, None
    def Gamma(self, rho):
        w0 = self.W(rho)
        r = linprog(np.ones(2 * self.N), A_eq=self.AeqG, b_eq=w0, bounds=(0, None), method='highs')
        assert r.status == 0, r.message
        return r.fun
    def M_n(self):
        return np.abs(self.Ws).sum(1).max()
    def stab_max(self, H):
        # max over pure stabilizer states of Tr(H sigma)
        return np.real(np.einsum('si,ij,sj->s', self.psi.conj(), H, self.psi)).max()
    def certified(self, H, rho):
        # Best bound certified by the direction H (Theorem 3.3): rescale H so that
        # max_alpha |Tr(H A_alpha)| = 2^n. Returns (bound, max_alpha |Tr(H A_alpha)|).
        t = np.array([np.trace(H @ A).real for A in self.Aops]); mu = np.abs(t).max()
        return (np.trace(H @ rho).real - self.stab_max(H)) / mu, mu
    def C_restricted(self, rho, labels):
        # Dual LP for C with the witness operator restricted to span{Pauli(l) : l in labels}.
        B = np.array([[np.trace(pauli(l) @ A).real for l in labels] for A in self.Aops]) / self.dim
        w = self.W(rho)
        A_ub = np.vstack([np.c_[self.Ws @ B, -np.ones(self.N)], np.c_[B, np.zeros(len(B))],
                          np.c_[-B, np.zeros(len(B))]])
        r = linprog(np.r_[-(B.T @ w), 1.0], A_ub, np.r_[np.zeros(self.N), np.ones(2 * len(B))],
                    bounds=(None, None), method='highs')
        return -r.fun

def fam(name, th, n=2):
    z = np.zeros(2**n, dtype=complex); o = np.zeros(2**n, dtype=complex); z[0] = 1; o[-1] = 1
    if name == 'Ry':  return proj(np.cos(th/2) * z + np.sin(th/2) * o)
    if name == 'Rx':  return proj(np.cos(th/2) * z - 1j * np.sin(th/2) * o)
    if name == 'BRz': return proj(z + np.exp(1j * th) * o)

def logical_ops(n):
    return (kron_all([X]*n), kron_all([Y] + [X]*(n-1)), kron_all([Z] + [I2]*(n-1)))

C1closed = lambda r: max(0.0, (np.abs(r).sum() - 1) / 2)          # Proposition 4.3
s_of = lambda r: 0 if abs(np.prod(r)) < 1e-12 else int(np.sign(np.prod(r)))

expected_counts = {1: 6, 2: 60, 3: 1080, 4: 36720}
MAG = {}
for n in (1, 2, 3):
    t = time.time(); MAG[n] = Magic(n)
    check(f"number of {n}-qubit stabilizer states = {expected_counts[n]}",
          MAG[n].N == expected_counts[n], f"found {MAG[n].N} ({time.time()-t:.1f}s)")
if RUN_N4:
    t = time.time(); MAG[4] = Magic(4)
    check("number of 4-qubit stabilizer states = 36720", MAG[4].N == 36720,
          f"found {MAG[4].N} ({time.time()-t:.1f}s)")
M1, M2, M3 = MAG[1], MAG[2], MAG[3]
W1, W2 = M1.W, M2.W
rng = np.random.default_rng(2603)
def rand_pure(dim):
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim); return proj(v)
def rand_mixed(dim, rank=None):
    rank = dim if rank is None else rank
    G = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank)); r = G @ G.conj().T
    return r / np.trace(r).real
def rand_bloch(pure=False):
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    return v if pure else v * rng.uniform(0, 1) ** (1/3)

# ----------------------------------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984', '#0b0b0b', '#52514e'
COL  = {'Ry': BLUE, 'Rx': ORANGE, 'BRz': AQUA}
NAME = {'Ry': r'$R_y$', 'Rx': r'$R_x$', 'BRz': r'Bell$+R_z$'}
plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
    'legend.frameon': False, 'lines.linewidth': 2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42})
FIGNUM = {}

def save(fig, name):
    path = os.path.join(FIGDIR, name + '.pdf')
    fig.savefig(path)
    print(f"   wrote {path}")
    return path

# ----------------------------------------------------------------------
def fig_dichotomy():
    rg_ = np.random.default_rng(2026)
    pts = {'both': [], 'one': [], 'two': []}
    while sum(len(v) for v in pts.values()) < 300:
        a, b = rg_.normal(size=3), rg_.normal(size=3)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        ca, cb = C1closed(a), C1closed(b)
        if ca < 1e-3 or cb < 1e-3:
            continue
        D = (1 + ca) * (1 + cb) - 1 - M2.C(np.kron(bloch(a), bloch(b)))
        key = ['both', 'one', 'two'][int(s_of(a) > 0) + int(s_of(b) > 0)]
        pts[key].append((ca * cb, D))
    sigmas = [np.array([.5, .5, .5]), np.array([.7, .4, .2]), np.array([.85, .3, .2])]
    ray = {}
    for sg in sigmas:
        xs, ys = [], []
        for rad in np.linspace(0.72, 1.0, 29):
            r = np.array([rad, rad, 0]) / np.sqrt(2); cr = C1closed(r)
            if cr < 1e-6:
                continue
            D = (1 + cr) * (1 + C1closed(sg)) - 1 - M2.C(np.kron(bloch(r), bloch(sg)))
            xs.append(cr); ys.append(D / cr)
        ray[tuple(sg)] = (xs, ys)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'wspace': 0.32})
    lab = {'both': r'$s(\rho),s(\sigma)\leq0$', 'one': r'one sign product $>0$', 'two': r'both $>0$'}
    for key, c, mk in [('two', AQUA, '^'), ('one', ORANGE, 's'), ('both', BLUE, 'o')]:
        arr = np.array(pts[key])
        ax.scatter(arr[:, 0], arr[:, 1], s=14, marker=mk, color=c, edgecolor='white', linewidth=0.5,
                   label=f'{lab[key]} ({len(arr)})', zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xlabel(r'$C(\rho)\,C(\sigma)$')
    ax.set_ylabel(r'deficit $(1{+}C_\rho)(1{+}C_\sigma)-1-C(\rho\otimes\sigma)$')
    ax.set_title('(a) Haar-random pure pairs', fontsize=9, loc='left', color=INK)
    ax.legend(fontsize=7, loc='upper left')
    for sg, c in zip(sigmas, [BLUE, ORANGE, AQUA]):
        xs, ys = ray[tuple(sg)]
        bx.plot(xs, ys, color=c, marker='o', ms=3)
        bx.annotate(r'$\vec r_\sigma=(%.2g,%.2g,%.2g)$' % tuple(sg), (xs[-1], ys[-1]), xytext=(4, 0),
                    textcoords='offset points', fontsize=7, color=INK, va='center')
    bx.set_xlabel(r'$C(\rho)$, $\rho$ on the ray $\vec r\propto(1,1,0)$')
    bx.set_ylabel(r'deficit$/C(\rho)$')
    bx.set_title(r'(b) deficit$/C(\rho)$ is not constant', fontsize=9, loc='left', color=INK)
    bx.set_xlim(0, 0.29)
    save(fig, 'fig_dichotomy')

    zero = {k: sum(abs(d) < 1e-8 for _, d in v) for k, v in pts.items()}
    FIGNUM['fig1'] = {k: (len(v), zero[k]) for k, v in pts.items()}
    check("Fig 1(a): zero deficit on every s<=0 pair and on no other pair",
          zero['both'] == len(pts['both']) and zero['one'] == zero['two'] == 0,
          f"s<=0: {zero['both']}/{len(pts['both'])}; others: {zero['one'] + zero['two']}/{len(pts['one']) + len(pts['two'])}")
    xs, ys = ray[(.7, .4, .2)]
    check("Fig 1(b): D/C(rho) for r_sigma=(.7,.4,.2) runs from 0.575 down to 0.5121",
          close(ys[0], 0.575, 1e-6) and close(ys[-1], 0.5121320, 1e-6), f"{ys[0]:.4f} -> {ys[-1]:.4f}")
    return fig

if MAKE_FIGURES:
    fig = fig_dichotomy()

# ----------------------------------------------------------------------
def fig_circuits():
    # 3 data units per inch, so aspect is 1:1 and circles stay round
    W_IN, H_IN = 3.45, 2.75
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3 * W_IN); ax.set_ylim(0, 3 * H_IN)
    ax.axis('off'); ax.set_aspect('equal')
    x0, x1 = 0.75, 4.55              # wire extent
    gap = 0.9                        # vertical distance between the two wires of a row

    def wire(y): ax.plot([x0, x1], [y, y], color=INK, lw=0.8, zorder=1)
    def gate(x, y, label, c, w=1.25):
        tint = 0.86 * np.ones(3) + 0.14 * np.array(matplotlib.colors.to_rgb(c))   # opaque light tint
        ax.add_patch(Rectangle((x - w / 2, y - 0.3), w, 0.6, facecolor=tint, edgecolor=c, lw=1.0, zorder=3))
        ax.text(x, y, label, ha='center', va='center', fontsize=8.5, color=INK, zorder=4)
    def cnot(x, yc, yt):
        ax.plot([x, x], [yt - 0.17, yc], color=INK, lw=0.8, zorder=2)
        ax.add_patch(Circle((x, yc), 0.07, color=INK, zorder=4))
        ax.add_patch(Circle((x, yt), 0.17, facecolor='white', edgecolor=INK, lw=0.8, zorder=3))
        ax.plot([x - 0.17, x + 0.17], [yt, yt], color=INK, lw=0.8, zorder=4)
        ax.plot([x, x], [yt - 0.17, yt + 0.17], color=INK, lw=0.8, zorder=4)

    rows = [
        ('Ry', [('g', 1.65, r'$R_y(\theta)$'), ('cx', 3.2)],
         r'$\cos\frac{\theta}{2}|00\rangle+\sin\frac{\theta}{2}|11\rangle$',
         r'$\kappa=1,\ \ C=|\sin\theta|+|\cos\theta|-1$'),
        ('Rx', [('g', 1.65, r'$R_x(\theta)$'), ('cx', 3.2)],
         r'$\cos\frac{\theta}{2}|00\rangle-i\sin\frac{\theta}{2}|11\rangle$',
         r'$\kappa=2,\ \ C=\frac{1}{2}(|\sin\theta|+|\cos\theta|-1)$'),
        ('BRz', [('g', 1.3, r'$H$', 0.55), ('cx', 2.35), ('g', 3.55, r'$R_z(\theta)$')],
         r'$\frac{1}{\sqrt{2}}\,(|00\rangle+e^{i\theta}|11\rangle)$',
         r'$\kappa=1,\ \ C=|\sin\theta|+|\cos\theta|-1$'),
    ]
    ytop = 3 * H_IN - 0.9
    for k, (fam_, ops, state, kc) in enumerate(rows):
        y1 = ytop - 2.95 * k; y2 = y1 - gap; c = COL[fam_]
        ax.text(0.05, y1 + 0.62, NAME[fam_], ha='left', va='center', fontsize=10, color=c)
        ax.text(3 * W_IN - 0.05, y1 + 0.62, kc, ha='right', va='center', fontsize=8, color=INK)
        for y in (y1, y2):
            wire(y); ax.text(x0 - 0.08, y, r'$|0\rangle$', ha='right', va='center', fontsize=8.5, color=INK)
        for op in ops:
            if op[0] == 'g':
                gate(op[1], y1, op[2], c, *(op[3:] if len(op) > 3 else ()))
            else:
                cnot(op[1], y1, y2)
        ax.annotate('', xy=(x1 + 0.5, (y1 + y2) / 2), xytext=(x1 + 0.1, (y1 + y2) / 2),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
        ax.text(x1 + 0.6, (y1 + y2) / 2, state, ha='left', va='center', fontsize=8.5, color=INK)
        if k < 2:
            ax.plot([0.05, 3 * W_IN - 0.05], [y2 - 0.62, y2 - 0.62], color='#e6e5e1', lw=0.6)
    save(fig, 'fig_circuits')
    # the three circuits prepare the stated states (global phase aside)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    th = 0.9; e0 = np.array([1, 0, 0, 0], dtype=complex)
    Ry_ = lambda t: np.cos(t/2) * I2 - 1j * np.sin(t/2) * Y
    Rx_ = lambda t: np.cos(t/2) * I2 - 1j * np.sin(t/2) * X
    Rz_ = lambda t: np.cos(t/2) * I2 - 1j * np.sin(t/2) * Z
    outs = {'Ry': CX @ np.kron(Ry_(th), I2) @ e0, 'Rx': CX @ np.kron(Rx_(th), I2) @ e0,
            'BRz': np.kron(Rz_(th), I2) @ CX @ np.kron(Hd, I2) @ e0}
    check("Fig 2: the three circuits prepare the Ry, Rx and Bell+Rz states",
          all(np.allclose(proj(v), fam(k, th)) for k, v in outs.items()))
    return fig

if MAKE_FIGURES:
    fig = fig_circuits()

# ----------------------------------------------------------------------
def fig_logical_bloch():
    fig = plt.figure(figsize=(3.6, 3.6)); ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False   # draw in call order: sphere, octahedron, then the three curves on top
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]; u = u + np.pi / 40   # keep grid lines off the drawn curves
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color='#ebeae6', lw=0.25,
                      rstride=2, cstride=2, zorder=0)
    V = {'0': (0, 0, 1), '1': (0, 0, -1), '+': (1, 0, 0), '-': (-1, 0, 0), '+i': (0, 1, 0), '-i': (0, -1, 0)}
    for a, b in [('0', '+'), ('0', '-'), ('0', '+i'), ('0', '-i'), ('1', '+'), ('1', '-'), ('1', '+i'),
                 ('1', '-i'), ('+', '+i'), ('+i', '-'), ('-', '-i'), ('-i', '+')]:
        ax.plot(*np.array([V[a], V[b]]).T, color=GREY, lw=0.7)
    tt = np.linspace(0.01, np.pi - 0.01, 200)
    # logical Bloch vectors: Ry (sin t, 0, cos t); Rx (0, -sin t, cos t); Bell+Rz (cos t, sin t, 0)
    ax.plot(np.sin(tt), 0 * tt, np.cos(tt), color=BLUE, lw=2, label=r'$R_y$, XZ meridian ($\kappa=1$)')
    ax.plot(0 * tt, -np.sin(tt), np.cos(tt), color=ORANGE, lw=2, label=r'$R_x$, YZ meridian ($\kappa=2$)')
    ph = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(ph), np.sin(ph), 0 * ph, color=AQUA, lw=2, label=r'Bell$+R_z$, equator ($\kappa=1$)')
    for k_, p in V.items():
        ax.scatter(*p, color=INK, s=10)
        ax.text(*(1.3 * np.array(p)), r'$|%s_L\rangle$' % k_, fontsize=7, color=INK, ha='center', va='center')
    ax.set_box_aspect((1, 1, 1), zoom=1.25); ax.set_axis_off(); ax.view_init(16, -58)
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 0.06))
    save(fig, 'fig_logical_bloch')
    XL2, YL2, ZL2 = logical_ops(2); t = 0.8
    lb = lambda r: np.array([np.trace(L @ r).real for L in (XL2, YL2, ZL2)])
    check("Fig 3: the drawn logical Bloch curves match <X_L>, <Y_L>, <Z_L> of the three families",
          np.allclose(lb(fam('Ry', t)), [np.sin(t), 0, np.cos(t)]) and
          np.allclose(lb(fam('Rx', t)), [0, -np.sin(t), np.cos(t)]) and
          np.allclose(lb(fam('BRz', t)), [np.cos(t), np.sin(t), 0]))
    return fig

if MAKE_FIGURES:
    fig = fig_logical_bloch()

# ----------------------------------------------------------------------
WCMAP = LinearSegmentedColormap.from_list('wig', ['#9c2a6a', '#e7b8d2', '#ffffff', '#d9d8d3', '#52514e'])

def fig_wigner(kind, th=np.pi/3, vmax=0.30):
    Wm = M2.W(fam(kind, th)).reshape(4, 4)
    fig, ax = plt.subplots(figsize=(1.8, 1.95))   # drawn at print size: 0.48\columnwidth
    ax.imshow(Wm, cmap=WCMAP, norm=TwoSlopeNorm(vmin=-0.125, vcenter=0, vmax=vmax))   # -1/8 is the two-qubit minimum
    ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    for i in range(4):
        for j in range(4):
            v = Wm[i, j]
            ax.text(j, i, f'{v:.3f}'.replace('-', '\u2212'), ha='center', va='center', fontsize=5.6,
                    color='white' if (v > 0.22 or v < -0.075) else INK)
            if v < -1e-12:
                ax.add_patch(Rectangle((j - .46, i - .46), .92, .92, fill=False, edgecolor=INK, lw=1.1))
    ticks = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
    ax.set_xticks(range(4), ticks, fontsize=5.5); ax.set_yticks(range(4), ticks, fontsize=5.5)
    ax.tick_params(length=0); ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_xlabel(r'qubit 2: $(q_2,p_2)$', fontsize=6.5, labelpad=2); ax.set_ylabel(r'qubit 1: $(q_1,p_1)$', fontsize=6.5, labelpad=2)
    ax.tick_params(pad=1.5)
    nneg = int((Wm < -1e-12).sum())
    ax.set_title(NAME[kind] + r', $\theta=\pi/3$: %d negative entries' % nneg, fontsize=7,
                 color=INK, y=-0.15)
    save(fig, 'fig_wigner_' + kind.lower())
    return fig, Wm

if MAKE_FIGURES:
    fig_ry, Wry = fig_wigner('Ry')
    fig_rx, Wrx = fig_wigner('Rx')
    check("Fig 4: 4 negative entries for Ry and 2 for Rx at theta = pi/3; values match the 8W matrices of Eq. (31)",
          (Wry < -1e-12).sum() == 4 and (Wrx < -1e-12).sum() == 2 and
          np.allclose(8 * Wry, [[1.5, 1.5, 3**.5/2, -3**.5/2], [1.5, 1.5, -3**.5/2, 3**.5/2],
                                [3**.5/2, -3**.5/2, .5, .5], [-3**.5/2, 3**.5/2, .5, .5]]))

# ----------------------------------------------------------------------
def fig_kappa():
    th = np.linspace(0.01, np.pi - 0.01, 400)
    base = np.abs(np.sin(th)) + np.abs(np.cos(th)) - 1
    thd = np.linspace(0.1, np.pi - 0.1, 13); thd = thd[np.abs(thd - np.pi / 2) > 0.05]
    lp = {k: [(M2.C(fam(k, t)), M2.Gamma(fam(k, t))) for t in thd] for k in ('Ry', 'Rx')}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.3})
    ax.plot(th, base, color=BLUE)
    ax.plot(th, base / 2, color=ORANGE)
    ax.plot(th, base, color=GREY, ls=(0, (4, 3)), lw=1.4)
    for k, c in (('Ry', BLUE), ('Rx', ORANGE)):
        ax.scatter(thd, [v[0] for v in lp[k]], s=16, color=c, edgecolor='white', lw=0.6, zorder=4)
    xt = ([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.set_xticks(*xt); ax.set_xlabel(r'$\theta$')
    ax.set_title(r'(a) $C$ and $\Gamma-1$ (dots: LP)', fontsize=9, loc='left')
    ax.annotate(r'$C(\rho^{Ry}_\theta)=\Gamma-1$ (both)', (np.pi/4, 0.414), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=7, color=INK)
    ax.annotate(r'$C(\rho^{Rx}_\theta)$', (np.pi/4, 0.207), xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=7, color=INK)
    ax.set_ylim(0, 0.47)
    keep = np.abs(th - np.pi / 2) > 0.02
    for k, c, val in (('Ry', BLUE, 1), ('Rx', ORANGE, 2)):
        bx.plot(th[keep], np.full(keep.sum(), val), color=c)
        bx.scatter(thd, [(g - 1) / cc for cc, g in lp[k]], s=16, color=c, edgecolor='white', lw=0.6, zorder=4)
        bx.annotate(NAME[k] + (r': $\kappa=%d$' % val), (np.pi, val), xytext=(-2, 6), textcoords='offset points',
                    ha='right', fontsize=8, color=INK)
    bx.axvline(np.pi / 2, color=MUTED, lw=0.8, ls=':')
    bx.text(np.pi / 2, 0.15, r' $\theta=\pi/2$: stabilizer', fontsize=7, color=MUTED)
    bx.set_ylim(0, 2.5); bx.set_xticks(*xt)
    bx.set_xlabel(r'$\theta$'); bx.set_ylabel(r'$\kappa$')
    bx.set_title(r'(b) $\kappa=(\Gamma-1)/C$', fontsize=9, loc='left')
    save(fig, 'fig_kappa_nb')
    dev = max(max(abs(cc - (abs(np.sin(t)) + abs(np.cos(t)) - 1) / (2 if k == 'Rx' else 1)),
                  abs(g - (abs(np.sin(t)) + abs(np.cos(t))))) for k in lp for t, (cc, g) in zip(thd, lp[k]))
    check("Fig 5: LP dots agree with the closed forms", dev < 1e-8, f"max deviation {dev:.1e}")
    return fig

if MAKE_FIGURES:
    fig = fig_kappa()

# ----------------------------------------------------------------------
def fig_noise():
    t0 = np.pi / 3; sc = np.sin(t0) + np.cos(t0); pc = (sc - 1) / (sc - 0.5)
    ps = np.linspace(0, pc * 0.985, 40)
    noise = {k: [] for k in ('Ry', 'Rx')}
    for k in noise:
        for p in ps:
            r = (1 - p) * fam(k, t0) + p * np.eye(4) / 4
            noise[k].append((M2.C(r), M2.Gamma(r) - 1))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'wspace': 0.3})
    for k in noise:
        ax.plot(ps, [v[0] for v in noise[k]], color=COL[k], label=r'$C$, ' + NAME[k])
    ax.plot(ps, [v[1] for v in noise['Ry']], color=GREY, ls=(0, (4, 3)), lw=1.4, label=r'$\Gamma-1$ (both)')
    ax.set_xlabel(r'noise rate $p$'); ax.set_title(r'(a) $C$ and $\Gamma-1$, $\theta=\pi/3$', fontsize=9, loc='left')
    ax.legend(fontsize=7)
    for k in noise:
        kap = [g / c for c, g in noise[k]]
        bx.plot(ps, kap, color=COL[k], marker='o', ms=2.5)
        bx.annotate(NAME[k], (ps[-1], kap[-1]), xytext=(-2, 6), textcoords='offset points', ha='right',
                    fontsize=8, color=INK)
    for ax_ in (ax, bx):
        ax_.axvline(pc, color=MUTED, lw=0.8, ls=':')
    bx.text(pc, 0.15, r'$p_c$ ', ha='right', fontsize=8, color=MUTED)
    bx.set_ylim(0, 2.5); bx.set_xlabel(r'noise rate $p$'); bx.set_ylabel(r'$\kappa$')
    bx.set_title(r'(b) $\kappa$ under depolarizing noise', fontsize=9, loc='left')
    save(fig, 'fig_noise_nb')
    kx = np.array([g / c for c, g in noise['Rx']]); ky = np.array([g / c for c, g in noise['Ry']])
    p15 = float(ps[np.argmax(ky > 1.5 - 1e-6)])
    FIGNUM['fig6'] = {'p_c': pc, 'p_where_kappa_Ry_reaches_1.5': p15}
    check("Fig 6: p_c = 0.4226, kappa(Rx) = 2 throughout, kappa(Ry) rises from 1 to 3/2 (reached by p ≈ 0.16)",
          close(pc, 0.4226, 1e-4) and np.allclose(kx, 2, atol=1e-6) and close(ky[0], 1, 1e-6)
          and close(ky[-1], 1.5, 1e-6) and 0.14 < p15 < 0.17, f"p_c = {pc:.4f}, kappa(Ry) = 3/2 from p = {p15:.3f}")
    return fig

if MAKE_FIGURES:
    fig = fig_noise()

# ----------------------------------------------------------------------
def fig_pca():
    mean = M2.Ws.mean(0)
    _, _, Vt = np.linalg.svd(M2.Ws - mean, full_matrices=False); B = Vt[:3].T
    pca = lambda w: (w - mean) @ B
    fig = plt.figure(figsize=(4.2, 3.6)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*pca(M2.Ws).T, s=10, color=GREY, depthshade=False, label='stabilizer states')
    tt = np.linspace(0.01, np.pi - 0.01, 200)
    for k in ('Ry', 'Rx', 'BRz'):
        ts = np.linspace(0, 2 * np.pi, 400) if k == 'BRz' else tt
        O = np.array([pca(M2.W(fam(k, t))) for t in ts])
        ax.plot(*O.T, color=COL[k], lw=1.8, label=NAME[k])
        ax.scatter(*pca(M2.W(fam(k, np.pi / 4))), marker='*', s=80, color=COL[k], edgecolor=INK, lw=0.5, zorder=5)
    ax.set_xlabel('PC1', labelpad=-3); ax.set_ylabel('PC2', labelpad=-3); ax.set_zlabel('PC3', labelpad=-5)
    ax.set_box_aspect((1, 1, 0.9), zoom=0.85)
    ax.tick_params(labelsize=6, pad=-2); ax.legend(fontsize=7, loc='upper left')
    save(fig, 'fig_pca_nb')
    tq = np.linspace(0.05, np.pi / 2 - 0.05, 40)
    check("Fig 7: the stars mark the most magic point theta = pi/4 of each family (C maximal on (0, pi/2))",
          all(abs(tq[np.argmax([M2.C(fam(k, t)) for t in tq])] - np.pi / 4) < 0.06 for k in ('Ry', 'Rx', 'BRz')))
    return fig

if MAKE_FIGURES:
    fig = fig_pca()

# ----------------------------------------------------------------------
if MAKE_FIGURES:
    for f in sorted(os.listdir(FIGDIR)):
        if f.endswith('.pdf'):
            print(f"   {f:24s} {os.path.getsize(os.path.join(FIGDIR, f)) / 1024:6.1f} kB")
    expected_figs = {'fig_dichotomy', 'fig_circuits', 'fig_logical_bloch', 'fig_wigner_ry', 'fig_wigner_rx',
                     'fig_kappa_nb', 'fig_noise_nb', 'fig_pca_nb'}
    check("all eight figure PDFs written (seven figures; Figure 4 has two panels)",
          expected_figs <= {f[:-4] for f in os.listdir(FIGDIR) if f.endswith('.pdf')})

# ---- summary
npass = sum(ok for _, ok, _ in LEDGER)
print(f"\n{npass} / {len(LEDGER)} checks passed ({time.time() - T0:.0f}s)")
sys.exit(0 if npass == len(LEDGER) else 1)
