"""Phase B21 -- deep prefix IC-RTRL. Implements and verifies the exact
L>=3 temporal-prefix dynamic program: for a source originating at layer
i, the collected active-prefix module at layer k is

    C_{k<-i} subset Hom(T_i, T_k),   dim <= r_i * r_k  (generic routing)

maintained bottom-to-top, causally, one shared object per layer (never
a per-symbolic-path list).

PRIMARY implementation: q=1 Kronecker routing (B_l = M_l (x) Q_l),
derived directly while building this phase's own machinery and
STRONGER than the phase's own r_i*r_k prediction for this routing
class -- the coefficient side collapses entirely, leaving a SINGLE
SHARED r_k-dim vector per hop (not r_i*r_k):

    y_t^{(i+1)} = R_{i+1} @ y_{t-1}^{(i+1)} + Q_{i+1} @ v_t
    y_t^{(k)}   = R_k @ y_{t-1}^{(k)} + Q_k @ y_t^{(k-1)}      (k>i+1)

and copy q's full sensitivity at the top layer is a SCALAR (composed
from the M_k copy-mixing factors) times this one shared vector.

GENERIC (dense) routing is used only as an F3 comparison baseline
(Part F) via direct, uncompressed forward propagation -- a full
multi-hop Hom-space reduction for dense routing was not derived with
enough confidence to implement in this focused pass; reported as an
explicit scope limit, not silently assumed solved.

Run:  python -m credit_memory.b21_prefix_ic_rtrl
"""
from __future__ import annotations

import numpy as np

import credit_memory.b18_temporal_core as b18


# ---------------------------------------------------------------------------
# Kronecker-structured stack: B_l = M_l (x) Q_l for l>=1 (routing between
# recurrent layers); layer 0's input routing B_0 stays dense (external
# input is not itself multiplicity-structured).
# ---------------------------------------------------------------------------
def build_kron_stack(widths_rs, M_IN0, seed):
    rng = np.random.RandomState(seed)
    layers, Bs, Ms, Qs = [], [], [], []
    M_in = M_IN0
    n_prev = None
    for l_idx, (width, r) in enumerate(widths_rs):
        R = rng.randn(r, r) / np.sqrt(r) * 0.85
        layer = b18.Layer(width, r, R, trainable_R=True, routing="dense", M_IN=M_in)
        layers.append(layer)
        n_cur = layer.n
        if l_idx == 0:
            B = b18.init_routing(layer, M_in, rng, scale=1.0)
            Ms.append(None); Qs.append(None)
        else:
            r_prev = widths_rs[l_idx - 1][1]
            Q = rng.randn(r, r_prev) / np.sqrt(r_prev)
            M = rng.randn(n_cur, n_prev) / np.sqrt(n_prev)
            sv1 = np.linalg.svd(M, compute_uv=False)[0]
            M = M * (0.5 / (sv1 + 1e-9))
            B = np.kron(M, Q)
            Ms.append(M); Qs.append(Q)
        Bs.append(B)
        M_in = width
        n_prev = n_cur
    c = rng.randn(layers[-1].width) / np.sqrt(layers[-1].width)
    return dict(layers=layers, Bs=Bs, Ms=Ms, Qs=Qs, c=c)


def compose_gain(Ms, i, p0, L_):
    """Scalar per-(final top copy) routing gain from layer i's copy p0,
    composed through every subsequent M_k (k=i+1..L_-1)."""
    gain = np.zeros(Ms[i + 1].shape[1]); gain[p0] = 1.0
    for k in range(i + 1, L_):
        gain = Ms[k] @ gain
    return gain


# ---------------------------------------------------------------------------
# Part A/B/C: prefix IC-RTRL gradients, all trainable parameters,
# q=1 Kronecker stack. Returns the same grads dict shape as b18/b20 for
# direct comparison against bptt_gradients.
# ---------------------------------------------------------------------------
def prefix_ic_rtrl_gradients(params, x, y):
    layers, Bs, Ms, Qs, c = (params["layers"], params["Bs"], params["Ms"],
                             params["Qs"], params["c"])
    L_ = len(layers)
    r_list = [layer.r for layer in layers]
    T_, batch = x.shape[0], x.shape[1]

    grads = {"dR": [np.zeros_like(layer.R) for layer in layers],
            "dB": [np.zeros_like(B) for B in Bs],
            "dc": np.zeros_like(c)}

    h_prev = [np.zeros((batch, layer.width)) for layer in layers]

    # R_l for every layer: LOCAL cost is O(r_l^2) (per B19/B20), but its
    # cross-layer propagation is EXPLICITLY not this phase's compression
    # target (Part G's subject; the B20 correction already established
    # it does not compress). Implemented via the same explicit multi-hop
    # chain already verified exact in B20's naive_rtrl_gradients -- for
    # each (l_idx,j,k), a list of per-layer sensitivities l_idx..L_-1,
    # each updated via ITS OWN layer's dynamics, causally within the step.
    S_R_chain = {}
    for l_idx in range(L_):
        r_l = r_list[l_idx]
        for j in range(r_l):
            for k in range(r_l):
                S_R_chain[(l_idx, j, k)] = [np.zeros((batch, layers[l2].width))
                                            for l2 in range(l_idx, L_)]

    # top-layer's own routing B_{L-1}: O(r) eligibility per Part C fix 2
    # (sensitivity confined to the injection copy's own r_{L-1} block)
    r_top = r_list[-1]
    S_Btop_block = {}  # (i,m) -> (batch, r_top), confined to copy i//r_top

    # B_l for every non-top layer (l=0..L-2): needs prefix propagation
    # through layers l+1..L-1. l=0 uses external input as its drive
    # source; l>=1 uses the (Kronecker-structured) SAME-timestep output
    # of layer l-1 -- both follow the identical q=1 chain.
    v_state = {l: {(a, m): np.zeros((batch, r_list[l]))
                  for a in range(layers[l].width) for m in range(Bs[l].shape[1])}
              for l in range(L_ - 1)}
    y_chain = {l: {(a, m): {k: np.zeros((batch, r_list[k])) for k in range(l + 1, L_)}
                  for a in range(layers[l].width) for m in range(Bs[l].shape[1])}
              for l in range(L_ - 1)}

    loss_total = 0.0
    for t in range(T_):
        h_cur = [None] * L_
        inp = x[t]
        for l_idx, (layer, B) in enumerate(zip(layers, Bs)):
            A = b18.build_A(layer)
            sp = h_prev[l_idx] @ A.T + inp @ B.T
            h_cur[l_idx] = sp
            inp = sp
        yhat_t = h_cur[-1] @ c
        r_err_t = (yhat_t - y[t]) / (T_ * batch)
        loss_total += 0.5 * float(np.mean((yhat_t - y[t]) ** 2))
        grads["dc"] += np.einsum("bn,b->n", h_cur[-1], r_err_t)

        # ---- R_l for every layer: local forcing + explicit multi-hop
        # chain to the top (exact, deliberately uncompressed -- Part G) ----
        for l_idx in range(L_):
            r_l = r_list[l_idx]
            n_l = layers[l_idx].n
            hh = h_prev[l_idx].reshape(batch, n_l, r_l)
            for j in range(r_l):
                for k in range(r_l):
                    f = np.zeros((batch, n_l, r_l)); f[:, :, j] = hh[:, :, k]
                    local_forcing = f.reshape(batch, layers[l_idx].width)
                    old_chain = S_R_chain[(l_idx, j, k)]
                    new_chain = []
                    prev_updated = None
                    for idx2, l2 in enumerate(range(l_idx, L_)):
                        A2 = b18.build_A(layers[l2])
                        if l2 == l_idx:
                            new_val = old_chain[idx2] @ A2.T + local_forcing
                        else:
                            new_val = old_chain[idx2] @ A2.T + prev_updated @ Bs[l2].T
                        new_chain.append(new_val)
                        prev_updated = new_val
                    S_R_chain[(l_idx, j, k)] = new_chain
                    grads["dR"][l_idx][j, k] += float(np.sum(
                        r_err_t[:, None] * (new_chain[-1] * c[None, :])))

        # ---- top-layer's own routing B_{L-1}: O(r) local block ----
        inp_top = h_cur[-2] if L_ > 1 else x[t]
        for iN in range(layers[-1].width):
            q0 = iN // r_top
            for mM in range(inp_top.shape[1]):
                key = (iN, mM)
                old = S_Btop_block.get(key, np.zeros((batch, r_top)))
                fpos = iN % r_top
                fb = np.zeros((batch, r_top)); fb[:, fpos] = inp_top[:, mM]
                Rtop = layers[-1].R
                new = old @ Rtop.T + fb
                S_Btop_block[key] = new
                full = np.zeros((batch, layers[-1].width))
                full[:, q0 * r_top:(q0 + 1) * r_top] = new
                grads["dB"][L_ - 1][iN, mM] += float(np.sum(r_err_t[:, None] * (full * c[None, :])))

        # ---- B_l for every non-top layer: q=1 Kronecker prefix chain ----
        for l_idx in range(L_ - 1):
            drive_src = x[t] if l_idx == 0 else h_cur[l_idx - 1]
            for (a, mM), v in v_state[l_idx].items():
                p0, pos0 = a // r_list[l_idx], a % r_list[l_idx]
                drive = np.zeros((batch, r_list[l_idx])); drive[:, pos0] = drive_src[:, mM]
                v_state[l_idx][(a, mM)] = v @ layers[l_idx].R.T + drive
                prev_shared = v_state[l_idx][(a, mM)]
                for k in range(l_idx + 1, L_):
                    Rk = layers[k].R
                    Qk = Qs[k]
                    y_chain[l_idx][(a, mM)][k] = y_chain[l_idx][(a, mM)][k] @ Rk.T + prev_shared @ Qk.T
                    prev_shared = y_chain[l_idx][(a, mM)][k]
                if L_ > l_idx + 1:
                    gain = compose_gain(Ms, l_idx, p0, L_)
                    full = np.einsum("q,br->bqr", gain, y_chain[l_idx][(a, mM)][L_ - 1]).reshape(batch, -1)
                else:
                    full = v_state[l_idx][(a, mM)]
                grads["dB"][l_idx][a, mM] += float(np.sum(r_err_t[:, None] * (full * c[None, :])))

        h_prev = h_cur

    return grads, loss_total / T_
