"""
Regenerate fig_dichotomy in TETRAHEDRAL-SIGN variables.

Why this figure must change: the old fig_dichotomy shaded the plane by the
hemisphere of <Z>_sigma and claimed superadditivity fails for <Z> > 0. The v4
correction (obs:dichotomy, rem:pauli_covariance) replaced that with the
Pauli-invariant criterion s(rho)=sgn(r_x r_y r_z): equality holds iff
s(rho)<=0 AND s(sigma)<=0. The figure should therefore classify pairs by the
(s(rho), s(sigma)) bucket, not by hemisphere.

Output: fig_dichotomy.pdf (and .png). Match your other figures' style as needed.

Deps: numpy, scipy, matplotlib.  Runtime: seconds.
"""
import numpy as np, itertools
from scipy.optimize import linprog
import matplotlib.pyplot as plt

np.random.seed(2026)
TOL = 1e-9

I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)

A1=[0.5*(I2+((-1)**p)*X+((-1)**(q+p))*Y+((-1)**q)*Z) for q in(0,1) for p in(0,1)]
def bloch(v): v=np.asarray(v,float); return 0.5*(I2+v[0]*X+v[1]*Y+v[2]*Z)

# --- 1- and 2-qubit stabilizer states + LP for C ---
stab1=[bloch(e) for e in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
P1=[I2,X,Y,Z]; paulis2=[np.kron(a,b) for a in P1 for b in P1]
gens=[s*paulis2[i] for i in range(1,16) for s in(1,-1)]
states2,seen=[],set()
for g1,g2 in itertools.product(gens,repeat=2):
    if np.allclose(g1@g2,g2@g1) and not np.allclose(g1,g2) and not np.allclose(g1,-g2):
        r=(np.eye(4)+g1)@(np.eye(4)+g2)/4
        if abs(np.trace(r)-1)>1e-9: continue
        k=tuple(np.round(r,9).flatten().view(float))
        if k not in seen: seen.add(k); states2.append(r)
assert len(states2)==60
A2=[np.kron(a,b) for a in A1 for b in A1]

def make_C(Aops,Ws):
    nS,dim=Ws.shape
    cvec=np.r_[np.zeros(nS),np.ones(dim)]
    A_ub=np.block([[-Ws.T,-np.eye(dim)],[Ws.T,-np.eye(dim)]])
    A_eq=np.r_[np.ones(nS),np.zeros(dim)][None,:]
    Aarr=np.array(Aops); norm=Aarr.shape[1]
    def C(rho):
        w0=np.einsum('ij,aji->a',rho,Aarr).real/norm
        res=linprog(cvec,A_ub=A_ub,b_ub=np.r_[-w0,w0],A_eq=A_eq,b_eq=[1],
                    bounds=[(0,None)]*(nS+dim),method='highs')
        assert res.status==0
        return res.fun
    return C
W1s=np.array([[np.trace(s@A).real/2 for A in A1] for s in stab1])
C1=make_C(A1,W1s)
W2s=np.array([[np.trace(s@A).real/4 for A in A2] for s in states2])
C2=make_C(A2,W2s)

def sgn_s(v):
    p=np.prod(v)
    return 0 if abs(p)<1e-9 else (1 if p>0 else -1)

# --- sample Haar-random single-qubit pure pairs, compute deficit + buckets ---
def rand_pure_bloch():
    psi=np.random.randn(2)+1j*np.random.randn(2); psi/=np.linalg.norm(psi)
    r=np.outer(psi,psi.conj())
    return np.array([np.trace(r@P).real for P in (X,Y,Z)])

N=400
buckets={(-1,-1):[], (-1,1):[], (1,-1):[], (1,1):[]}
for _ in range(N):
    u,v=rand_pure_bloch(),rand_pure_bloch()
    cu,cv=C1(bloch(u)),C1(bloch(v))
    if cu<1e-3 or cv<1e-3: continue
    deficit=(1+cu)*(1+cv)-1-C2(np.kron(bloch(u),bloch(v)))
    su,sv=sgn_s(u),sgn_s(v)
    key=(1 if su>0 else -1, 1 if sv>0 else -1)
    buckets[key].append((cu*cv, max(deficit,0)))

# --- plot: deficit vs C(rho)C(sigma), colored by (s_rho,s_sigma) bucket ---
fig,ax=plt.subplots(1,1,figsize=(5.2,4.0))
styles={(-1,-1):('#2ca02c','o',r'$s(\rho)\leq 0,\ s(\sigma)\leq 0$ (equality)'),
        (-1, 1):('#d62728','^',r'$s(\sigma)>0$'),
        ( 1,-1):('#ff7f0e','v',r'$s(\rho)>0$'),
        ( 1, 1):('#9467bd','s',r'$s(\rho)>0,\ s(\sigma)>0$')}
for key,(color,mk,lab) in styles.items():
    pts=buckets[key]
    if not pts: continue
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.scatter(xs,ys,s=18,c=color,marker=mk,alpha=0.7,edgecolors='none',label=lab)
ax.axhline(0,color='k',lw=0.6,ls='--',alpha=0.5)
ax.set_xlabel(r'$C(\rho)\,C(\sigma)$')
ax.set_ylabel(r'deficit $=(1{+}C(\rho))(1{+}C(\sigma))-1-C(\rho\otimes\sigma)$')
ax.set_title('Tetrahedral dichotomy: deficit by sign-product bucket')
ax.legend(fontsize=7,loc='upper left',framealpha=0.9)
plt.tight_layout()
plt.savefig('fig_dichotomy.pdf'); plt.savefig('fig_dichotomy.png',dpi=200)
print("wrote fig_dichotomy.pdf / .png")
# sanity: the (-,-) bucket should sit on y=0; others above
for key in buckets:
    ys=[p[1] for p in buckets[key]]
    if ys: print(f"  bucket {key}: n={len(ys)}, max deficit={max(ys):.4f}")
