"""B37b analysis: aggregate by (family, r, arm) and classify each family."""
import json
import numpy as np
from collections import defaultdict

rows = json.load(open("results/b37b/rows.json"))
FAMS = list(dict.fromkeys(r["family"] for r in rows))
ARMS = list(dict.fromkeys(r["arm"] for r in rows))
RS = sorted(set(r["r"] for r in rows))

def med(v):
    v = np.array(v, dtype=float)
    return float(np.median(v)) if len(v) else float("nan")

g = defaultdict(list)
for r in rows:
    g[(r["family"], r["r"], r["arm"])].append(r)

print("=" * 118)
print("TABLE 1 -- held-out rollout NMSE (median over 3 eval seeds; LR chosen on VALIDATION only)")
print("=" * 118)
hdr = f"{'family':21s} {'r':>2s} " + " ".join(f"{a.replace('_perturbed',''):>16s}" for a in ARMS)
print(hdr)
for f in FAMS:
    for r in RS:
        cells = []
        for a in ARMS:
            rs = g[(f, r, a)]
            cells.append(f"{med([x['test_nmse'] for x in rs]):16.2e}")
        print(f"{f:21s} {r:2d} " + " ".join(cells))

print()
print("=" * 118)
print("TABLE 2 -- Markov-parameter error  max_k<40 |C u(Cq)^k B - C* A^k B*| / (1+|C* A^k B*|)  (median)")
print("=" * 118)
print(hdr)
for f in FAMS:
    for r in RS:
        cells = [f"{med([x['markov'] for x in g[(f,r,a)]]):16.2e}" for a in ARMS]
        print(f"{f:21s} {r:2d} " + " ".join(cells))

print()
print("=" * 118)
print("TABLE 3 -- divergence fraction during training (loss > 1e12 or non-finite; 3 seeds)")
print("=" * 118)
print(hdr)
for f in FAMS:
    for r in RS:
        cells = [f"{np.mean([x['diverged'] for x in g[(f,r,a)]]):16.2f}" for a in ARMS]
        print(f"{f:21s} {r:2d} " + " ".join(cells))

print()
print("=" * 118)
print("TABLE 4 -- arm A DRIFT: exact init retained (best) vs params after 400 Adam steps (final), median NMSE")
print("=" * 118)
print(f"{'family':21s} {'r':>2s} {'init/best NMSE':>16s} {'final NMSE':>16s} {'drift ratio':>14s} {'mean |g_a|':>12s} {'mean |g_th|':>12s}")
for f in FAMS:
    for r in RS:
        rs = g[(f, r, "A_constructive")]
        b, fi = med([x['test_nmse'] for x in rs]), med([x['final_nmse'] for x in rs])
        print(f"{f:21s} {r:2d} {b:16.2e} {fi:16.2e} {fi/max(b,1e-300):14.2e} "
              f"{med([x['gnorm_a'] for x in rs]):12.2e} {med([x['gnorm_theta'] for x in rs]):12.2e}")

print()
print("=" * 118)
print("TABLE 5 -- conditioning / transient diagnostics (per family, median over seeds & arms A,B)")
print("=" * 118)
print(f"{'family':21s} {'r':>2s} {'rho(A)':>8s} {'cond(S)':>10s} {'cond(T)':>10s} {'cond(M) fin':>12s} {'max|z| fin':>11s} {'rho(M) fin':>11s}")
for f in FAMS:
    for r in RS:
        rs = [x for a in ARMS if a != "C_generic_stable" for x in g[(f, r, a)]]
        fin = [x for x in rs if np.isfinite(x['max_z'])]
        print(f"{f:21s} {r:2d} {med([x['rho_A'] for x in rs]):8.4f} {med([x['condS'] for x in rs]):10.2e} "
              f"{med([x['condT'] for x in rs]):10.2e} {med([x['condM'] for x in fin]):12.2e} "
              f"{med([x['max_z'] for x in fin]):11.2e} {med([x['rho'] for x in fin]):11.3f}")

print()
print("=" * 118)
print("TABLE 6 -- does failure correlate with the B37a transient statistic? (Spearman over ALL runs)")
print("=" * 118)
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5: return float("nan"), int(ok.sum())
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())

lognm = [np.log10(max(x['test_nmse'], 1e-300)) for x in rows]
for name, v in [("log10 init max|z| (B37a transient stat)", [np.log10(max(x['init_max_z'], 1e-300)) for x in rows]),
                ("log10 cond(T)", [np.log10(x['condT']) for x in rows]),
                ("log10 cond(S) of teacher", [np.log10(x['condS']) for x in rows]),
                ("rho(A) of teacher", [x['rho_A'] for x in rows])]:
    c, n = spearman(v, lognm)
    print(f"  rho_S(log NMSE, {name:42s}) = {c:+.3f}   (n={n})")
div = [float(x['diverged']) for x in rows]
for name, v in [("log10 init max|z|", [np.log10(max(x['init_max_z'], 1e-300)) for x in rows]),
                ("log10 cond(T)", [np.log10(x['condT']) for x in rows])]:
    c, n = spearman(v, div)
    print(f"  rho_S(diverged,  {name:42s}) = {c:+.3f}   (n={n})")

print()
print("=" * 118)
print("VERDICT per family  (learnable / initialization-sensitive / conditioning-limited)")
print("=" * 118)
GOOD, OK_ = 1e-3, 1e-1
for f in FAMS:
    a_ok, b_small, b_large, c_ok, divs = [], [], [], [], []
    for r in RS:
        a_ok.append(med([x['test_nmse'] for x in g[(f, r, 'A_constructive')]]))
        b_small.append(med([x['test_nmse'] for x in g[(f, r, 'B_perturbed_1e-06')]
                            + g[(f, r, 'B_perturbed_1e-04')]]))
        b_large.append(med([x['test_nmse'] for x in g[(f, r, 'B_perturbed_1e-02')]]))
        c_ok.append(med([x['test_nmse'] for x in g[(f, r, 'C_generic_stable')]]))
        divs += [x['diverged'] for a in ARMS for x in g[(f, r, a)]]
    A, Bs, Bl, C = max(a_ok), max(b_small), max(b_large), max(c_ok)
    if C < OK_ and Bs < OK_:
        v = "LEARNABLE"
    elif Bs < OK_ and (C >= OK_ or Bl >= OK_):
        v = "INITIALIZATION-SENSITIVE"
    else:
        v = "CONDITIONING-LIMITED"
    print(f"{f:21s} A={A:9.2e} B(1e-6,1e-4)={Bs:9.2e} B(1e-2)={Bl:9.2e} "
          f"C={C:9.2e} div={np.mean(divs):.2f}  ->  {v}")
