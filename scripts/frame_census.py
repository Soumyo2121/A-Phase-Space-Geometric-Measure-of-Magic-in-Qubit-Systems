#!/usr/bin/env python3
"""
Two-qubit frame census for arXiv:2603.20792, rem:frame_freedom.
Replaces the old 6144-frame enumeration (which double-counted and missed frames).

Claims checked:
  [1] Every sign pattern A(0) = (1/4)(I + sum_P eps_P P), eps in {+-1}^15, is a valid
      Pauli-covariant frame (Hermitian, unit trace, Tr A(a)A(b) = 4 delta).  -> 32768
  [2] Translation relabels origins in orbits of 16                          -> 2048 frames
  [3] Frames from 5 commuting triples + one eigenstate each                 -> 304 frames (4864 origins)
  [4] (C(rho_Rx), C(rho_Ry)) at theta = pi/3: values and class sizes, all / triple-built
  [5] max C over the 64 face (x) face products per frame (sqrt(3)/2 or less?)
  [6] multistart optimisation of the separable max in sampled frames of each class
Expected (reference run): [4] 512 x 4 overall; 80/80/80/64 triple-built.
                          [5] sqrt3/2 in 128 of 2048.  [6] ~0.666 / ~0.739 / 0.866.
Also [4b]: M_2 = 2 in every frame.
Runtime: [4] ~10 s, [5] ~5 min, [6] ~4 min.   Set RUN_SEPOPT = False to skip [6].
"""
import numpy as np, itertools, collections, time
from scipy.optimize import linprog, minimize
RUN_FACEMAX, RUN_SEPOPT = True, True

I2=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex); Z=np.diag([1,-1]).astype(complex)
P1=[I2,X,Y,Z]
mats=[np.kron(P1[i],P1[j]) for i in range(4) for j in range(4) if (i,j)!=(0,0)]
D=[np.kron(a,b) for a in P1 for b in P1]
chi=np.array([[1 if np.allclose(d@M,M@d) else -1 for M in mats] for d in D])   # 16 x 15

# 60 two-qubit stabilizer states
gens=[s*M for M in mats for s in (1,-1)]; stab,seen=[],set()
for g1,g2 in itertools.product(gens,repeat=2):
    if np.allclose(g1@g2,g2@g1) and not np.allclose(g1,g2) and not np.allclose(g1,-g2):
        r=(np.eye(4)+g1)@(np.eye(4)+g2)/4
        if abs(np.trace(r)-1)>1e-9: continue
        k=tuple(np.round(r,9).flatten().view(float))
        if k not in seen: seen.add(k); stab.append(r)
assert len(stab)==60
pvec=lambda r: np.array([np.trace(r@M).real for M in mats])
Sv=np.array([pvec(s) for s in stab])
# W_rho(a) = Tr(rho A(a))/4 = (1 + sum_P eps_P chi_a(P) <P>)/16
Wmat=lambda e: (1+(chi*e)@Sv.T)/16
Wvec=lambda e,pv: (1+(chi*e)@pv)/16
def C(e,pv,Ws=None):
    Ws=Wmat(e) if Ws is None else Ws; w0=Wvec(e,pv)
    c=np.r_[np.zeros(60),np.ones(16)]
    A=np.block([[-Ws,-np.eye(16)],[Ws,-np.eye(16)]])
    r=linprog(c,A_ub=A,b_ub=np.r_[-w0,w0],A_eq=np.r_[np.ones(60),np.zeros(16)][None],
              b_eq=[1],bounds=[(0,None)]*76,method='highs')
    assert r.status==0; return r.fun

# [1] validity of all sign patterns (explicit operator check)
t0=time.time(); allp=list(itertools.product([1,-1],repeat=15)); nvalid=0
for e in allp:
    A0=0.25*(np.eye(4)+sum(s*M for s,M in zip(e,mats)))
    Ao=np.array([d@A0@d.conj().T for d in D])
    G=np.einsum('aij,bji->ab',Ao,Ao).real
    nvalid+= np.allclose(Ao.sum(0),4*np.eye(4)) and np.allclose(G,4*np.eye(16))
print(f"[1] valid origin operators: {nvalid} / {len(allp)}   [{time.time()-t0:.0f}s]")

# [2] translation orbits
rep,reps={},[]
for e in allp:
    if e in rep: continue
    orb=[tuple(np.array(e)*chi[a]) for a in range(16)]; r=min(orb)
    for x in orb: rep[x]=r
    reps.append(r)
print(f"[2] distinct frames up to translation: {len(reps)}")

# [3] triple-partition (MUB-built) frames
def plab(a,b):
    M=mats[a]@mats[b]
    for l,Q in enumerate(mats):
        for ph in (1,-1,1j,-1j):
            if np.allclose(M,ph*Q): return l,ph
triples=sorted({tuple(sorted((a,b,plab(a,b)[0]))) for a,b in itertools.combinations(range(15),2)
                if np.allclose(mats[a]@mats[b],mats[b]@mats[a])})
parts=[]
def srch(used,ch):
    if len(ch)==5: parts.append(ch); return
    f=min(set(range(15))-used)
    for t in triples:
        if f in t and not set(t)&used: srch(used|set(t),ch+[t])
srch(set(),[])
mub=set()
for part in parts:
    for sg in itertools.product([1,-1],repeat=10):
        e=np.zeros(15,int)
        for k,(a,b,c) in enumerate(part):
            s1,s2=sg[2*k],sg[2*k+1]; ph=plab(a,b)[1]
            e[a],e[b],e[c]=s1,s2,int(np.real(ph))*s1*s2
        mub.add(tuple(e))
mubreps={rep[e] for e in mub}
print(f"[3] partitions: {len(parts)}; triple-built origins: {len(mub)} (6*4^5=6144 before dedup); "
      f"frames: {len(mubreps)}; product frame included: {tuple([1]*15) in mub}")

# [4] Rx / Ry classes
proj=lambda v: np.outer(v,v.conj())/np.vdot(v,v).real
k00,k11=np.eye(4)[0],np.eye(4)[3]; TH=np.pi/3
rx=pvec(proj(np.cos(TH/2)*k00-1j*np.sin(TH/2)*k11))
ry=pvec(proj(np.cos(TH/2)*k00+np.sin(TH/2)*k11))
res={}
for r in reps:
    e=np.array(r); Ws=Wmat(e); res[r]=(C(e,rx,Ws),C(e,ry,Ws))
def tab(keys,name):
    cnt=collections.Counter((round(res[k][0],6),round(res[k][1],6)) for k in keys)
    print(f"[4] {name} ({len(keys)} frames):")
    for (a,b),n in sorted(cnt.items()): print(f"      C_Rx={a:.6f}  C_Ry={b:.6f}  ratio={a/b:.3f}  n={n}")
tab(reps,"all frames"); tab([r for r in reps if r in mubreps],"triple-built frames")

# [4b] M_2 = max over stabilizer states of ||W_s||_1, per frame (reference: 2.0 in all 2048)
M2=collections.Counter(round(np.abs(Wmat(np.array(r))).sum(0).max(),6) for r in reps)
print(f"[4b] M_2 across all frames: {sorted(M2.items())}")

# [4c] analytic criterion: ||W_s||_1 = 2  iff  lambda_S * c_P1 c_P2 c_P3 = -1  (S = stabilizer group of s)
trip={}
for a,b in itertools.combinations(range(15),2):
    if np.allclose(mats[a]@mats[b],mats[b]@mats[a]):
        c,ph=plab(a,b); trip[tuple(sorted((a,b,c)))]=int(np.real(ph))
ok=True
for r in reps:
    e=np.array(r); nrm=np.abs(Wmat(e)).sum(0)
    for j in range(60):
        sup=tuple(sorted(np.nonzero(np.abs(Sv[j])>0.5)[0]))
        ok&=bool(np.isclose(nrm[j], 2 if trip[sup]*np.prod(e[list(sup)])==-1 else 1))
mp=[t for t in trip if set(t)<=set(range(15))]
print(f"[4c] criterion holds for all 2048 frames x 60 states: {ok}")

# [5] face (x) face maxima
bl=lambda v: 0.5*(I2+v[0]*X+v[1]*Y+v[2]*Z)
faces=[np.array(s)/np.sqrt(3) for s in itertools.product([1,-1],repeat=3)]
fp=[pvec(np.kron(bl(f),bl(g))) for f in faces for g in faces]
if RUN_FACEMAX:
    t0=time.time(); mx={}
    for r in reps:
        e=np.array(r); Ws=Wmat(e); mx[r]=max(C(e,p,Ws) for p in fp)
    for name,keys in (("all",reps),("triple-built",[r for r in reps if r in mubreps])):
        print(f"[5] max over face pairs, {name}:",
              sorted(collections.Counter(round(mx[r],6) for r in keys).items()),
              f" (sqrt3/2 = {np.sqrt(3)/2:.6f})   [{time.time()-t0:.0f}s]")

# [6] separable max by multistart Nelder-Mead (a LOWER bound on the true max)
if RUN_SEPOPT and RUN_FACEMAX:
    rng=np.random.default_rng(1)
    sv=lambda t,p: np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    tp=lambda v:(np.arccos(v[2]),np.arctan2(v[1],v[0]))
    def sepmax(e):
        Ws=Wmat(e); f=lambda x:-C(e,pvec(np.kron(bl(sv(x[0],x[1])),bl(sv(x[2],x[3])))),Ws)
        st=sorted((np.r_[tp(a),tp(c)] for a in faces for c in faces),key=f)[:6]
        st+= [rng.uniform(0,1,4)*[np.pi,2*np.pi,np.pi,2*np.pi] for _ in range(6)]
        return max(-minimize(f,x0,method='Nelder-Mead',
                   options={'xatol':1e-7,'fatol':1e-10,'maxiter':1500}).fun for x0 in st)
    groups=collections.defaultdict(list)
    for r in reps: groups[(round(mx[r],6), r in mubreps)].append(r)
    for k in sorted(groups):
        smp=[groups[k][i] for i in rng.choice(len(groups[k]),min(4,len(groups[k])),replace=False)]
        print(f"[6] face-max {k[0]:.6f}, triple-built={k[1]!s:5} (n={len(groups[k])}): "
              f"optimised separable max {np.round([sepmax(np.array(r)) for r in smp],6)}")
