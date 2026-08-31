"""B37c section 8 -- B37b vs B37c head-to-head. B37b results are READ from its
frozen artifacts, never recomputed."""
import json
import numpy as np
from collections import defaultdict

C = json.load(open("results/b37c/rows.json"))
Bm = json.load(open("results/b37b/rows.json"))          # B37b 400-step sweep (frozen)
Bl = json.load(open("results/b37b/armC_long.json"))     # B37b 4000-step generic arm (frozen)
FAMS = list(dict.fromkeys(x["family"] for x in Bm)); RS = (4, 8)
m = lambda v: float(np.median(v)) if len(v) else float("nan")

c = defaultdict(list); [c[(x["family"], x["r"], x["arm"], x["steps"])].append(x) for x in C]
b = defaultdict(list); [b[(x["family"], x["r"], x["arm"])].append(x) for x in Bm]
bl = defaultdict(list); [bl[(x["family"], x["r"])].append(x) for x in Bl]

print("=" * 122)
print("TABLE 1 -- PRIMARY ARM: generic/random stable init, held-out NMSE (median, 3 seeds, LR on validation)")
print("=" * 122)
print(f"{'family':21s} {'r':>2s} | {'B37b 400':>10s} {'B37c 400':>10s} | {'B37b 4000':>10s} "
      f"{'B37c 4000':>10s} | {'improve':>9s} | {'B37b div':>8s} {'B37c div':>8s}")
for f in FAMS:
    for r in RS:
        b4 = m([x["test_nmse"] for x in b[(f, r, "C_generic_stable")]])
        c4 = m([x["test_nmse"] for x in c[(f, r, "C_generic_stable", 400)]])
        bL = m([x["test_nmse"] for x in bl[(f, r)]])
        cL = m([x["test_nmse"] for x in c[(f, r, "C_generic_stable", 4000)]])
        bd = np.mean([x["diverged"] for x in b[(f, r, "C_generic_stable")]] +
                     [x["diverged"] for x in bl[(f, r)]])
        cd = np.mean([x["diverged"] for s in (400, 4000)
                      for x in c[(f, r, "C_generic_stable", s)]])
        print(f"{f:21s} {r:2d} | {b4:10.2e} {c4:10.2e} | {bL:10.2e} {cL:10.2e} | "
              f"{bL/max(cL,1e-300):9.1f}x | {bd:8.2f} {cd:8.2f}")

print()
print("=" * 122)
print("TABLE 2 -- CHART FRAGILITY: exact/near-exact initialization (400 steps), held-out NMSE")
print("=" * 122)
ARMS = ["A_exact", "B_perturbed_1e-06", "B_perturbed_1e-04", "B_perturbed_1e-02"]
BARMS = ["A_constructive", "B_perturbed_1e-06", "B_perturbed_1e-04", "B_perturbed_1e-02"]
print(f"{'family':21s} {'r':>2s} | " + " ".join(f"{'B37b '+a.split('_')[-1]:>12s}" for a in BARMS)
      + " | " + " ".join(f"{'B37c '+a.split('_')[-1]:>12s}" for a in ARMS))
for f in FAMS:
    for r in RS:
        bb = " ".join(f"{m([x['test_nmse'] for x in b[(f,r,a)]]):12.2e}" for a in BARMS)
        cc = " ".join(f"{m([x['test_nmse'] for x in c[(f,r,a,400)]]):12.2e}" for a in ARMS)
        print(f"{f:21s} {r:2d} | {bb} | {cc}")

print()
print("=" * 122)
print("TABLE 3 -- divergence rate over ALL exact/near-exact arms (the B37b failure mode)")
print("=" * 122)
tb = np.mean([x["diverged"] for f in FAMS for r in RS for a in BARMS for x in b[(f, r, a)]])
tc = np.mean([x["diverged"] for f in FAMS for r in RS for a in ARMS for x in c[(f, r, a, 400)]])
print(f"  B37b global quotient chart : {tb:.3f}")
print(f"  B37c native ProductLocal   : {tc:.3f}")
for f in FAMS:
    xb = np.mean([x["diverged"] for r in RS for a in BARMS for x in b[(f, r, a)]])
    xc = np.mean([x["diverged"] for r in RS for a in ARMS for x in c[(f, r, a, 400)]])
    print(f"    {f:21s} B37b {xb:.2f}   B37c {xc:.2f}")

print()
print("=" * 122)
print("TABLE 4 -- NONNORMAL deep dive (all arms; B37c diagnostics)")
print("=" * 122)
print(f"{'r':>2s} {'arm':19s} {'steps':>5s} {'NMSE':>10s} {'markov':>10s} {'val best':>10s} "
      f"{'val final':>10s} {'rho(M)':>7s} {'Gam_H':>9s} {'max|z|':>9s} {'||b||':>8s} {'||C||':>9s} "
      f"{'hankel':>9s} {'|g_u|':>9s} {'div':>4s}")
for r in RS:
    for a, s in [("C_generic_stable", 400), ("C_generic_stable", 4000), ("A_exact", 400),
                 ("B_perturbed_1e-06", 400), ("B_perturbed_1e-04", 400), ("B_perturbed_1e-02", 400)]:
        rs = c[("nonnormal", r, a, s)]
        if not rs: continue
        print(f"{r:2d} {a:19s} {s:5d} {m([x['test_nmse'] for x in rs]):10.2e} "
              f"{m([x['markov'] for x in rs]):10.2e} {m([x['val_loss'] for x in rs]):10.2e} "
              f"{m([x['final_val'] for x in rs]):10.2e} {m([x['rho'] for x in rs]):7.3f} "
              f"{m([x['gamma_H'] for x in rs]):9.2e} {m([x['max_z'] for x in rs]):9.2e} "
              f"{m([x['b_norm'] for x in rs]):8.2e} {m([x['C_norm'] for x in rs]):9.2e} "
              f"{m([x['hankel'] for x in rs]):9.2e} {m([x['gnorm_u'] for x in rs]):9.2e} "
              f"{np.mean([x['diverged'] for x in rs]):4.2f}")

print()
print("=" * 122)
print("TABLE 5 -- B37c generic arm (4000 steps): transient / port diagnostics vs success")
print("=" * 122)
print(f"{'family':21s} {'r':>2s} {'NMSE':>10s} {'markov':>10s} {'rho(M)':>7s} {'Gam_H':>9s} "
      f"{'max|z|':>9s} {'||b||':>8s} {'||C||':>9s} {'hankel':>9s} {'|g_u|':>9s} {'|g_b|':>9s} {'|g_C|':>9s}")
for f in FAMS:
    for r in RS:
        rs = c[(f, r, "C_generic_stable", 4000)]
        print(f"{f:21s} {r:2d} {m([x['test_nmse'] for x in rs]):10.2e} "
              f"{m([x['markov'] for x in rs]):10.2e} {m([x['rho'] for x in rs]):7.3f} "
              f"{m([x['gamma_H'] for x in rs]):9.2e} {m([x['max_z'] for x in rs]):9.2e} "
              f"{m([x['b_norm'] for x in rs]):8.2e} {m([x['C_norm'] for x in rs]):9.2e} "
              f"{m([x['hankel'] for x in rs]):9.2e} {m([x['gnorm_u'] for x in rs]):9.2e} "
              f"{m([x['gnorm_b'] for x in rs]):9.2e} {m([x['gnorm_C'] for x in rs]):9.2e}")

print()
print("=" * 122)
print("TABLE 6 -- did any ordinary family get WORSE under B37c? (negative results preserved)")
print("=" * 122)
for f in FAMS:
    for r in RS:
        bL = m([x["test_nmse"] for x in bl[(f, r)]])
        cL = m([x["test_nmse"] for x in c[(f, r, "C_generic_stable", 4000)]])
        tag = "B37c WORSE" if cL > 3 * bL else ("B37c better" if cL * 3 < bL else "comparable")
        print(f"{f:21s} {r:2d}  B37b {bL:10.2e}   B37c {cL:10.2e}   -> {tag}")
