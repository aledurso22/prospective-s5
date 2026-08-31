import json
import numpy as np
from collections import defaultdict

S = json.load(open("results/b38a/sweep.json"))
O = json.load(open("results/b38a/online.json"))
I = json.load(open("results/b38a/identity.json"))
X = json.load(open("results/b38a/state_identity.json"))
FAMS = list(dict.fromkeys(x["family"] for x in S))
LS = sorted(set(x["L"] for x in S))
m = lambda v: float(np.median(v)) if len(v) else float("nan")
s = defaultdict(list); [s[(x["family"], x["L"], x["arm"])].append(x) for x in S]
o = defaultdict(list); [o[x["family"]].append(x) for x in O]

print("=" * 126)
print("TABLE 1 -- ARM A (matched TBPTT) vs ARM B (forward causal RTRL): identical model, data, schedule, Adam state")
print("=" * 126)
print(f"{'family':21s} " + " ".join(f"{'L='+str(L)+' A':>11s} {'L='+str(L)+' B':>11s}" for L in LS)
      + f" {'max traj dev':>13s}")
for f in FAMS:
    cells, td = [], []
    for L in LS:
        for a in ("A", "B"):
            cells.append(f"{m([x['test_nmse'] for x in s[(f,L,a)]]):11.2e}")
        td += [x.get("traj_dev", np.nan) for x in s[(f, L, 'A')]]
    print(f"{f:21s} " + " ".join(cells) + f" {np.nanmax(td):13.2e}")
alltd = [x.get("traj_dev", np.nan) for x in S]
print(f"\n  max validation-trajectory deviation over the whole sweep: {np.nanmax(alltd):.3e}")
print(f"  max per-chunk gradient rel err during training: u={max(r['u'] for r in I):.3e}  "
      f"b={max(r['b'] for r in I):.3e}  C_out={max(r['C'] for r in I):.3e}  "
      f"({sum(r['chunks'] for r in I)} chunks)")
print(f"  max parameter-trajectory deviation: {max(r['d_params'] for r in X):.3e};  "
      f"Adam m {max(r['d_m'] for r in X):.3e};  Adam v {max(r['d_v'] for r in X):.3e}")

print()
print("=" * 126)
print("TABLE 2 -- Markov-parameter error and divergence (median over 3 seeds)")
print("=" * 126)
print(f"{'family':21s} " + " ".join(f"{'L='+str(L)+' A':>11s} {'L='+str(L)+' B':>11s}" for L in LS)
      + f" {'div A':>6s} {'div B':>6s}")
for f in FAMS:
    cells = []
    for L in LS:
        for a in ("A", "B"):
            cells.append(f"{m([x['markov'] for x in s[(f,L,a)]]):11.2e}")
    da = np.mean([x["diverged"] for L in LS for x in s[(f, L, "A")]])
    db = np.mean([x["diverged"] for L in LS for x in s[(f, L, "B")]])
    print(f"{f:21s} " + " ".join(cells) + f" {da:6.2f} {db:6.2f}")

print()
print("=" * 126)
print("TABLE 3 -- ARM C every-token online (theta updated after EVERY token, trace carried)")
print("=" * 126)
print(f"{'family':21s} {'online NMSE':>12s} {'held-out NMSE':>14s} {'markov':>10s} "
      f"{'tokens to 90% drop':>19s} {'div':>5s} | {'best A/B batch NMSE':>20s}")
for f in FAMS:
    rs = o[f]
    ab = m([x["test_nmse"] for L in LS for a in ("A", "B") for x in s[(f, L, a)]])
    t9 = [x["t90"] for x in rs if x["t90"] > 0]
    print(f"{f:21s} {m([x['online_nmse'] for x in rs]):12.2e} "
          f"{m([x['heldout_nmse'] for x in rs]):14.2e} {m([x['markov'] for x in rs]):10.2e} "
          f"{(str(int(m(t9))) if t9 else 'not reached'):>19s} "
          f"{np.mean([x['diverged'] for x in rs]):5.2f} | {ab:20.2e}")
nfail = sum(1 for x in O if x["heldout_nmse"] > 0.1)
print(f"\n  Arm C: {len(O)-nfail}/{len(O)} runs reach held-out NMSE < 0.1; "
      f"divergence in {sum(x['diverged'] for x in O)}/{len(O)} runs")

print()
print("=" * 126)
print("TABLE 4 -- wall-clock, CPU-ONLY (median over the selected-LR runs, full 200-epoch train)")
print("=" * 126)
print(f"{'L':>5s} {'arm A (s)':>10s} {'arm B (s)':>10s} {'B/A':>6s}")
for L in LS:
    wa = m([x["wall"] for f in FAMS for x in s[(f, L, "A")]])
    wb = m([x["wall"] for f in FAMS for x in s[(f, L, "B")]])
    print(f"{L:5d} {wa:10.2f} {wb:10.2f} {wb/wa:6.2f}")
