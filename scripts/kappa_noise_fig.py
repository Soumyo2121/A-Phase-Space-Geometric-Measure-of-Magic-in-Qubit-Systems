"""
Regenerate fig_noise CORRECTLY: kappa(Lambda_p(rho)) vs p for Rx and Ry.
Finding from kappa_rigidity_v2: Rx is rigid (kappa=2 flat); Ry is NOT
(kappa rises). This plots both honestly on a dense p-grid.
Gamma is computed over the FULL 60-state stabilizer set (the v1 bug was
restricting it to the 6 logical states).
"""
import numpy as np, itertools
from scipy.optimize import linprog
import matplotlib.pyplot as plt
TOL=1e-6
I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],dtype=complex); Y=np.array([[0,-1j],[1j,0]],dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)
A1=[0.5*(I2+((-1)**p)*X+((-1)**(q+p))*Y+((-1)**q)*Z) for q in(0,1) for p in(0,1)]
A2=[np.kron(a,b) for a in A1 for b in A1]
P1=[I2,X,Y,Z]; paulis2=[np.kron(a,b) for a in P1 for b in P1]
gens=[s*paulis2[i] for i in range(1,16) for s in(1,-1)]
stab2,seen=[],set()
for g1,g2 in itertools.product(gens,repeat=2):
    if np.allclose(g1@g2,g2@g1) and not np.allclose(g1,g2) and not np.allclose(g1,-g2):
        r=(np.eye(4)+g1)@(np.eye(4)+g2)/4
        if abs(np.trace(r)-1)>1e-9: continue
        k=tuple(np.round(r,9).flatten().view(float))
        if k not in seen: seen.add(k); stab2.append(r)
assert len(stab2)==60
W2s=np.array([[np.trace(s@a).real/4 for a in A2] for s in stab2])
Mfull=np.array([np.concatenate([s.real.flatten(),s.imag.flatten()]) for s in stab2]).T

def C2(rho):
    nS,dim=W2s.shape
    w0=np.array([np.trace(rho@a).real/4 for a in A2])
    cvec=np.r_[np.zeros(nS),np.ones(dim)]
    A_ub=np.block([[-W2s.T,-np.eye(dim)],[W2s.T,-np.eye(dim)]])
    A_eq=np.r_[np.ones(nS),np.zeros(dim)][None,:]
    r=linprog(cvec,A_ub=A_ub,b_ub=np.r_[-w0,w0],A_eq=A_eq,b_eq=[1],
              bounds=[(0,None)]*(nS+dim),method='highs'); assert r.status==0
    return r.fun
def Gamma(rho):
    b=np.concatenate([rho.real.flatten(),rho.imag.flatten()]); nS=len(stab2)
    r=linprog(np.ones(2*nS),A_eq=np.hstack([Mfull,-Mfull]),b_eq=b,
              bounds=[(0,None)]*2*nS,method='highs')
    return r.fun if r.status==0 else np.nan
k00=np.array([1,0,0,0],dtype=complex); k11=np.array([0,0,0,1],dtype=complex)
def proj(v): v=v/np.linalg.norm(v); return np.outer(v,v.conj())
def depol(rho,p): return (1-p)*rho+p*np.eye(4)/4
def ry(th): return proj(np.cos(th/2)*k00+np.sin(th/2)*k11)
def rx(th): return proj(np.cos(th/2)*k00-1j*np.sin(th/2)*k11)

th=np.pi/3
sc=abs(np.sin(th))+abs(np.cos(th))
pstar=(sc-1)/(sc-0.5)          # p_c: C -> 0 here (LP-verified; the old 1-1/sc was wrong)
ps=np.linspace(0,pstar*0.999,40)
def curve(mk):
    ks=[]
    for p in ps:
        rho=depol(mk(th),p); c=C2(rho)
        ks.append((Gamma(rho)-1)/c if c>TOL else np.nan)
    return np.array(ks)
kRx,kRy=curve(rx),curve(ry)

print(f"p_c = {pstar:.4f}\np/p_c   kappa_Rx   kappa_Ry")
for i in range(0,40,5):
    print(f"{ps[i]/pstar:6.3f}   {kRx[i]:8.4f}   {kRy[i]:8.4f}")

fig,ax=plt.subplots(figsize=(5,3.6))
ax.plot(ps/pstar,kRx,'-',color='#d62728',lw=2,label=r'$R_x$ (rigid, $\kappa=2$)')
ax.plot(ps/pstar,kRy,'-',color='#2ca02c',lw=2,label=r'$R_y$ ($\kappa$ rises with $p$)')
ax.axhline(2,ls=':',color='#d62728',alpha=0.4); ax.axhline(1,ls=':',color='#2ca02c',alpha=0.4)
ax.set_xlabel(r'depolarizing strength $p/p_c$')
ax.set_ylabel(r'$\kappa(\Lambda_p(\rho))$')
ax.set_title(r'$\kappa$ under depolarizing noise ($\theta=\pi/3$)')
ax.legend(fontsize=8,loc='center left'); ax.set_ylim(0.8,2.2)
plt.tight_layout(); plt.savefig('fig_noise_nb.pdf'); plt.savefig('fig_noise_nb.png',dpi=200)
print("\nwrote fig_noise_nb.pdf/.png")
print(f"Rx flat at 2? max dev = {np.nanmax(np.abs(kRx-2)):.2e}")
print(f"Ry range: {np.nanmin(kRy):.3f} -> {np.nanmax(kRy):.3f}")
