"""B9.2 -- shared candidate-pool diagnostic. Diagnostic only: no S5
run, no training-algorithm change. Keeps |rho_j| as the per-lower-mode
selector (B9.1's conclusion); asks whether every lower mode's O(2N)
candidate scan can be replaced by a scan over a small SHARED pool P_K
of K upper-credit channels, with each lower mode still free to pick its
own best channel *inside* P_K (NOT the B9.1 Part-5 single-shared-
channel experiment, where every mode was forced to use the SAME
channel).

Two checkpoints per seed: step 0 (init) and step 600 (end of a short,
UNMODIFIED "online"-arm training run -- b5_train.py's own arm=='online'
update, reused verbatim/inlined here only because that script does not
expose intermediate params; not a new algorithm). Reduced from B5's
4-point checkpoint ladder [0,100,300,600] to 2 points to keep this
diagnostic's combinatorial-search cost down; still answers the
"stability across training" question with a clean before/after
comparison.

At each (seed, checkpoint): builds the full relevance matrix
R[j,m]=rho_{j,m} (2N x N, complex, via the B9.1-fixed, leak-free
StreamingRelevance) and the oracle-utility matrix U[j,m] (S=empty
definition from B9.1, via the leak-free per-row gamma_j sum) using
N_CAL_TRAJ=4 calibration trajectories drawn from the CURRENT params,
then evaluates candidate pools on N_TEST_TRAJ=4 held-out trajectories.

2N=12 at this toy's N=6 makes EXACT combinatorial search over pools
feasible for every K used here (max C(12,6)=924 subsets) -- no greedy
fallback was needed, but one is documented (not implemented) for the
case where 2N grows too large for brute force.

Run:  python -m credit_memory.b9_2_shared_pool
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
from collections import Counter

import numpy as np

from toyrig import ssm_rig as tcg
from credit_memory.hankel import build_F, build_c_t
from credit_memory.lagcorr import per_coordinate_contribution
from credit_memory.streaming import StreamingRelevance, run_windowed_calibration
from credit_memory.phase_b4c_streaming_rank1 import deploy_selected_channel
from credit_memory.b5_train import set_config, draw_task_batch, loss_of
from credit_memory.phase_b2bc_hankel_truncation import cos_np, relerr_np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "credit_memory")

N, T, BATCH, DELAY = 6, 60, 8, 20
SEEDS = list(range(8))
N_CAL_TRAJ, N_TEST_TRAJ = 4, 4
CHECKPOINTS = [0, 600]
LR = 1e-3
K_LIST = [1, 2, 4, 8, 2 * N]        # 2N=12=all candidates, sanity anchor
N_RANDOM_POOL_DRAWS = 20
EXACT_SEARCH_LIMIT = 5000           # C(2N,K) below this -> brute force


def run_online_training(seed, target_step):
    """Reuses b5_train.py's EXISTING arm=='online' update unmodified,
    purely to obtain a realistic post-training parameter snapshot."""
    tcg.L, tcg.N, tcg.T, tcg.DELAY, tcg.BATCH = 2, N, T, DELAY, BATCH
    rng = np.random.RandomState(1000 + seed)
    params = tcg.init_params(seed)
    flat = tcg.flatten(params)
    m_ = np.zeros_like(flat)
    v_ = np.zeros_like(flat)
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    for step in range(1, target_step + 1):
        x, y = draw_task_batch(rng)
        _, h, r = loss_of(params, x, y)
        q = tcg.spatial_q(params, h, r)
        Sa, Sb = tcg.sensitivities(params, h, x)
        G = tcg.assemble(params, h, x, r, q, Sa, Sb)
        g = tcg.flat_grads(G, params)
        m_ = b1_ * m_ + (1 - b1_) * g
        v_ = b2_ * v_ + (1 - b2_) * g ** 2
        flat = flat - LR * (m_ / (1 - b1_ ** step)) / (
            np.sqrt(v_ / (1 - b2_ ** step)) + eps)
        params = tcg.pack(params, flat)
    return params


def make_row(params, rng):
    x, y = draw_task_batch(rng)
    _, h, r = loss_of(params, x, y)
    q = tcg.spatial_q(params, h, r)
    Sa, Sb = tcg.sensitivities(params, h, x)
    lam = tcg.exact_lambda(params, q)
    G_bptt = tcg.assemble(params, h, x, r, lam, Sa, Sb, direct=True)["a"][0]
    G_online = tcg.assemble(params, h, x, r, q, Sa, Sb)["a"][0]
    return dict(a1=params["a"][1], B1=params["b"][1], q1=q[1], Sa0=Sa[0],
               G_bptt=G_bptt, G_online=G_online)


def exact_gamma_no_leak(f_diag, d, cal_rows, B1_col, m):
    gamma = np.zeros(2 * N, np.complex128)
    for row in cal_rows:
        c_row = build_c_t(row["q1"], B1_col)
        u_row = row["Sa0"][:, :, m]
        g_row, _ = per_coordinate_contribution(f_diag, d, c_row, u_row)
        gamma += g_row
    return gamma


def best_pool_exact(score_mat, K):
    """Exact combinatorial search maximizing sum_m max_{j in P} score[j,m].
    Falls back to greedy forward selection if C(2N,K) is too large (not
    exercised at this toy's 2N=12, documented for larger 2N)."""
    n2 = score_mat.shape[0]
    from math import comb
    if comb(n2, K) <= EXACT_SEARCH_LIMIT:
        best_val, best_P = -np.inf, None
        for P in itertools.combinations(range(n2), K):
            val = float(sum(score_mat[list(P), m].max() for m in range(N)))
            if val > best_val:
                best_val, best_P = val, P
        return set(best_P)
    # greedy fallback (not used here, documented for scale-up)
    P = set()
    remaining = set(range(n2))
    for _ in range(K):
        best_j, best_val = None, -np.inf
        for j in remaining:
            trial = P | {j}
            val = float(sum(score_mat[list(trial), m].max()
                            for m in range(N)))
            if val > best_val:
                best_val, best_j = val, j
        P.add(best_j)
        remaining.discard(best_j)
    return P


def pool_most_frequent(j_rho_unrestricted, rho_mat, K):
    counts = Counter(j_rho_unrestricted.values())
    ranked = sorted(range(2 * N),
                    key=lambda j: (-counts.get(j, 0),
                                  -np.sum(np.abs(rho_mat[j, :]))))
    return set(ranked[:K])


def pool_largest_lambda(f_diag, K):
    return set(np.argsort(-np.abs(f_diag))[:K].tolist())


def pool_uniform_coverage(f_diag, K):
    order = np.argsort(np.abs(f_diag))
    n2 = len(order)
    if K >= n2:
        return set(order.tolist())
    pos = sorted(set(int(round(p)) for p in np.linspace(0, n2 - 1, K)))
    i = 0
    while len(pos) < K and i < n2:
        if i not in pos:
            pos.append(i)
        i += 1
    return set(order[i] for i in sorted(pos)[:K])


def evaluate_pool(P, rho_mat, U_mat, f_diag, B1, test_rows,
                  j_rho_unrestricted, j_oracle_unrestricted):
    P = sorted(P)
    j_pool_by_mode = {m: P[int(np.argmax(np.abs(rho_mat[P, m])))]
                      for m in range(N)}
    coverage = sum(1 for m in range(N) if j_rho_unrestricted[m] in P)
    coss, relerrs = [], []
    for row in test_rows:
        G_hat = np.zeros(N, np.complex128)
        for m in range(N):
            G_hat[m] = deploy_selected_channel(
                f_diag, j_pool_by_mode[m], B1[:, m], row, m)
        coss.append(cos_np(G_hat, row["G_bptt"]))
        relerrs.append(relerr_np(G_hat, row["G_bptt"]))
    util_pool = float(sum(U_mat[j_pool_by_mode[m], m] for m in range(N)))
    util_rho = float(sum(U_mat[j_rho_unrestricted[m], m] for m in range(N)))
    util_oracle = float(sum(U_mat[j_oracle_unrestricted[m], m]
                            for m in range(N)))
    return dict(cos_median=float(np.median(coss)),
               relerr_median=float(np.median(relerrs)),
               coverage=coverage,
               regret_vs_rho=util_rho - util_pool,
               regret_vs_oracle=util_oracle - util_pool)


def effective_ranks(M, fracs=(0.90, 0.95, 0.99)):
    s = np.linalg.svd(M, compute_uv=False)
    sq = s ** 2
    total = sq.sum()
    if total <= 0:
        return {str(f): int(M.shape[1]) for f in fracs}, s.tolist()
    cum = np.cumsum(sq) / total
    return ({str(f): int(np.searchsorted(cum, f) + 1) for f in fracs},
           s.tolist())


def main() -> None:
    print("=" * 90)
    print(f"Phase B9.2: shared candidate-pool diagnostic, "
         f"{len(SEEDS)} seeds x {len(CHECKPOINTS)} checkpoints")
    print("=" * 90)

    per_ckpt_seed = {}     # (ckpt,seed) -> dict of everything
    rng_pool = np.random.RandomState(424242)

    for ckpt in CHECKPOINTS:
        for seed in SEEDS:
            set_config()
            params = (tcg.init_params(seed) if ckpt == 0
                      else run_online_training(seed, ckpt))
            a1, B1 = params["a"][1], params["b"][1]
            f_diag = build_F(a1)
            d = np.ones(2 * N, np.complex128)

            cal_rng = np.random.RandomState(777 + seed * 31 + ckpt)
            test_rng = np.random.RandomState(9000 + seed * 31 + ckpt)
            cal_rows = [make_row(params, cal_rng) for _ in range(N_CAL_TRAJ)]
            test_rows = [make_row(params, test_rng)
                        for _ in range(N_TEST_TRAJ)]

            rho_mat = np.zeros((2 * N, N), np.complex128)
            U_mat = np.zeros((2 * N, N))
            for m in range(N):
                est = run_windowed_calibration(f_diag, cal_rows, m)
                rho_mat[:, m] = est.rho
                gamma = exact_gamma_no_leak(f_diag, d, cal_rows, B1[:, m], m)
                G = gamma.sum()
                U_mat[:, m] = (2 * np.real(np.conj(G) * gamma)
                              - np.abs(gamma) ** 2)

            j_rho = {m: int(np.argmax(np.abs(rho_mat[:, m])))
                    for m in range(N)}
            j_oracle = {m: int(np.argmax(U_mat[:, m])) for m in range(N)}

            per_ckpt_seed[(ckpt, seed)] = dict(
                rho_mat=rho_mat, U_mat=U_mat, f_diag=f_diag, B1=B1,
                test_rows=test_rows, j_rho=j_rho, j_oracle=j_oracle)

    # ---------------- Part 1 ----------------
    print("\n--- Part 1: per-mode winners ---")
    n_unique_by_ckpt = {ckpt: [] for ckpt in CHECKPOINTS}
    winner_hist = Counter()
    for (ckpt, seed), d_ in per_ckpt_seed.items():
        uniq = len(set(d_["j_rho"].values()))
        n_unique_by_ckpt[ckpt].append(uniq)
        for j in d_["j_rho"].values():
            winner_hist[j] += 1
    for ckpt in CHECKPOINTS:
        print(f"checkpoint {ckpt}: median unique winning channels across "
             f"{N} modes = {np.median(n_unique_by_ckpt[ckpt]):.1f} "
             f"(of {2 * N} candidates, {N} modes)")
    print(f"winner frequency histogram (pooled over all seeds x "
         f"checkpoints, {N} picks each): {dict(sorted(winner_hist.items()))}")

    # stability across checkpoints (per seed)
    frac_changed = []
    jaccard_ckpt = []
    for seed in SEEDS:
        j0 = per_ckpt_seed[(0, seed)]["j_rho"]
        j1 = per_ckpt_seed[(600, seed)]["j_rho"]
        changed = sum(1 for m in range(N) if j0[m] != j1[m])
        frac_changed.append(changed / N)
        s0, s1 = set(j0.values()), set(j1.values())
        jaccard_ckpt.append(len(s0 & s1) / len(s0 | s1))
    print(f"stability across checkpoints (0 -> 600): median fraction of "
         f"modes whose winner changed = {np.median(frac_changed):.2f}; "
         f"median Jaccard overlap of winner sets = "
         f"{np.median(jaccard_ckpt):.2f}")

    # overlap across seeds, at each checkpoint
    for ckpt in CHECKPOINTS:
        sets = [set(per_ckpt_seed[(ckpt, s)]["j_rho"].values())
               for s in SEEDS]
        jacc = [len(a & b) / len(a | b)
               for a, b in itertools.combinations(sets, 2)]
        print(f"checkpoint {ckpt}: median pairwise cross-seed Jaccard "
             f"overlap of winner sets = {np.median(jacc):.2f} "
             f"({len(jacc)} seed pairs)")

    # ---------------- Part 2/3: pool construction + evaluation --------
    print("\n--- Part 2/3: shared candidate-pool evaluation ---")
    methods = ["oracle_exact", "rho_exact", "most_frequent",
              "largest_lambda", "uniform_coverage", "random"]
    pool_results = {meth: {K: [] for K in K_LIST} for meth in methods}

    for (ckpt, seed), d_ in per_ckpt_seed.items():
        rho_mat, U_mat = d_["rho_mat"], d_["U_mat"]
        f_diag, B1, test_rows = d_["f_diag"], d_["B1"], d_["test_rows"]
        j_rho, j_oracle = d_["j_rho"], d_["j_oracle"]
        abs_rho_mat = np.abs(rho_mat)

        for K in K_LIST:
            pools = dict(
                oracle_exact=best_pool_exact(U_mat, K),
                rho_exact=best_pool_exact(abs_rho_mat, K),
                most_frequent=pool_most_frequent(j_rho, rho_mat, K),
                largest_lambda=pool_largest_lambda(f_diag, K),
                uniform_coverage=pool_uniform_coverage(f_diag, K))
            random_results = []
            for _ in range(N_RANDOM_POOL_DRAWS):
                P = set(rng_pool.choice(2 * N, size=K,
                                        replace=False).tolist())
                random_results.append(evaluate_pool(
                    P, rho_mat, U_mat, f_diag, B1, test_rows,
                    j_rho, j_oracle))
            pool_results["random"][K].append(dict(
                cos_median=float(np.median(
                    [r["cos_median"] for r in random_results])),
                relerr_median=float(np.median(
                    [r["relerr_median"] for r in random_results])),
                coverage=float(np.mean(
                    [r["coverage"] for r in random_results])),
                regret_vs_rho=float(np.median(
                    [r["regret_vs_rho"] for r in random_results])),
                regret_vs_oracle=float(np.median(
                    [r["regret_vs_oracle"] for r in random_results]))))
            for meth, P in pools.items():
                pool_results[meth][K].append(evaluate_pool(
                    P, rho_mat, U_mat, f_diag, B1, test_rows,
                    j_rho, j_oracle))

    print(f"{'method':16s} " + " ".join(f"K={K:>2d}" for K in K_LIST)
         + "   (median held-out cos vs BPTT)")
    for meth in methods:
        line = [np.median([r["cos_median"] for r in pool_results[meth][K]])
               for K in K_LIST]
        print(f"{meth:16s} " + " ".join(f"{v:5.3f}" for v in line))

    print(f"\n{'method':16s} " + " ".join(f"K={K:>2d}" for K in K_LIST)
         + "   (median regret_vs_rho, oracle utility units)")
    for meth in methods:
        line = [np.median([r["regret_vs_rho"] for r in pool_results[meth][K]])
               for K in K_LIST]
        print(f"{meth:16s} " + " ".join(f"{v:6.2f}" for v in line))

    print(f"\n{'method':16s} " + " ".join(f"K={K:>2d}" for K in K_LIST)
         + "   (mean modes covered by pool, of N=6)")
    for meth in methods:
        line = [np.mean([r["coverage"] for r in pool_results[meth][K]])
               for K in K_LIST]
        print(f"{meth:16s} " + " ".join(f"{v:5.2f}" for v in line))

    # ---------------- Part 4: relevance-matrix structure --------------
    print("\n--- Part 4: relevance/utility matrix structure ---")
    rho_ranks = {"0.9": [], "0.95": [], "0.99": []}
    U_ranks = {"0.9": [], "0.95": [], "0.99": []}
    for (ckpt, seed), d_ in per_ckpt_seed.items():
        r_er, _ = effective_ranks(d_["rho_mat"])
        u_er, _ = effective_ranks(d_["U_mat"])
        for f in ("0.9", "0.95", "0.99"):
            rho_ranks[f].append(r_er[f])
            U_ranks[f].append(u_er[f])
    print(f"R[j,m]=rho effective rank (median over {len(per_ckpt_seed)} "
         f"(seed,ckpt)): 90%={np.median(rho_ranks['0.9']):.1f}  "
         f"95%={np.median(rho_ranks['0.95']):.1f}  "
         f"99%={np.median(rho_ranks['0.99']):.1f}  (full rank = {N})")
    print(f"U[j,m]=oracle utility effective rank (median): "
         f"90%={np.median(U_ranks['0.9']):.1f}  "
         f"95%={np.median(U_ranks['0.95']):.1f}  "
         f"99%={np.median(U_ranks['0.99']):.1f}  (full rank = {N})")

    git = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    doc = dict(
        git=git,
        config=dict(N=N, T=T, BATCH=BATCH, DELAY=DELAY, seeds=SEEDS,
                   checkpoints=CHECKPOINTS, n_cal_traj=N_CAL_TRAJ,
                   n_test_traj=N_TEST_TRAJ, k_list=K_LIST),
        part1=dict(
            median_unique_winners_by_ckpt={
                str(c): float(np.median(v))
                for c, v in n_unique_by_ckpt.items()},
            winner_histogram={str(k): v for k, v in winner_hist.items()},
            median_frac_winner_changed_0_to_600=float(np.median(frac_changed)),
            median_jaccard_stability_0_to_600=float(np.median(jaccard_ckpt))),
        part2_3_pool_cos_median={
            meth: {str(K): float(np.median(
                [r["cos_median"] for r in pool_results[meth][K]]))
                for K in K_LIST}
            for meth in methods},
        part2_3_pool_regret_vs_rho_median={
            meth: {str(K): float(np.median(
                [r["regret_vs_rho"] for r in pool_results[meth][K]]))
                for K in K_LIST}
            for meth in methods},
        part2_3_pool_regret_vs_oracle_median={
            meth: {str(K): float(np.median(
                [r["regret_vs_oracle"] for r in pool_results[meth][K]]))
                for K in K_LIST}
            for meth in methods},
        part2_3_pool_coverage_mean={
            meth: {str(K): float(np.mean(
                [r["coverage"] for r in pool_results[meth][K]]))
                for K in K_LIST}
            for meth in methods},
        part4_rho_effective_rank_median={
            f: float(np.median(v)) for f, v in rho_ranks.items()},
        part4_U_effective_rank_median={
            f: float(np.median(v)) for f, v in U_ranks.items()})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "b9_2_shared_pool_summary.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
