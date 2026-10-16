#!/usr/bin/env python3
"""
check_certificates.py -- verify every certificate in lp_1820_certificates.json in exact rational arithmetic.
No LP solver and no floating point: stabilizer Wigner vectors are rebuilt from the stabilizer groups listed in
stabilizer_states.csv, using Tr(P A_(q,p)) = 1, (-1)^p, (-1)^(q+p), (-1)^q for P = I, X, Y, Z.
For each pair it checks: weights >= 0 summing to 1; |S_a| <= 1; and
  || W_target - sum_i w_i W_i ||_1  ==  <S, W_target> - max_{all 60 stabilizers} <S, W_i>  ==  stated C_joint.
"""
import csv, json, sys
from fractions import Fraction as F
import os
HERE = os.path.dirname(os.path.abspath(__file__))   # files live next to this script
PTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
tr1 = {'I': lambda q, p: 1, 'X': lambda q, p: (-1)**p, 'Y': lambda q, p: (-1)**(q + p), 'Z': lambda q, p: (-1)**q}
def wig_from_group(gens):
    group = [(1, 'II')] + [(1 if g[0] == '+' else -1, g[1:]) for g in gens]
    return [F(1, 16) * sum(sg * tr1[P[0]](*a) * tr1[P[1]](*b) for sg, P in group) for a in PTS for b in PTS]
WST = [wig_from_group(row['stabilizer_group'].split()) for row in csv.DictReader(open(os.path.join(HERE, 'stabilizer_states.csv')))]
assert len(WST) == 60 and all(sum(w) == 1 for w in WST)
def w1(r):
    rx, ry, rz = r
    return [F(1, 4) * (1 + (-1)**p * rx + (-1)**(q + p) * ry + (-1)**q * rz) for q, p in PTS]
data = json.load(open(os.path.join(HERE, 'lp_1820_certificates.json')))
bad = 0
for c in data['pairs']:
    r = [F(x, c['denominator']) for x in c['r_rho']]; s = [F(x, c['denominator']) for x in c['r_sigma']]
    tgt = [x * y for x in w1(r) for y in w1(s)]
    wts = {int(i): F(w) for i, w in c['mixture'].items()}; S = [F(x) for x in c['witness_S']]
    val = F(c['C_joint'])
    mix = [sum(w * WST[i][a] for i, w in wts.items()) for a in range(16)]
    primal = sum(abs(x - y) for x, y in zip(tgt, mix))
    dual = sum(si * ti for si, ti in zip(S, tgt)) - max(sum(si * wi for si, wi in zip(S, w)) for w in WST)
    ok = all(w >= 0 for w in wts.values()) and sum(wts.values()) == 1 and all(abs(x) <= 1 for x in S) \
         and primal == val and dual == val
    bad += not ok
print(f"{len(data['pairs']) - bad} / {len(data['pairs'])} certificates verified exactly")
sys.exit(1 if bad else 0)
