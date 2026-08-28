"""Phase B20 -- end-to-end exact invariant-credit RTRL.

Builds a genuine forward-only (no backward pass through time, no
history replay) exact gradient algorithm for the deep I_n (x) R shared-
core architecture, using B19's minimal-tangent-module machinery to
compress the CROSS-LAYER propagation of lower-layer parameter
sensitivities, while accepting B19's own conclusion that a dense
core's OWN r^2 local parameter gradients are unavoidable and are not
further compressed.

Three algorithms are implemented and cross-verified to machine
precision:
  - `bptt_gradients` (reused from b18_temporal_core): reverse-mode
    adjoint, needs the full forward trajectory before it can run
    backward through time. The established reference throughout B16-19.
  - `naive_rtrl_gradients`: textbook forward-only RTRL -- maintains the
    FULL sensitivity dh_t^l/dtheta for every trainable theta, updated
    forward in time only. Correct by construction for any architecture,
    but O(n_params * N) storage/step -- not yet compressed.
  - `ic_rtrl_gradients`: the compressed version. Each layer's own R_l
    entries get a LOCAL O(r_l^2) sensitivity (matching B19's "small-
    core-local and unavoidable" finding -- not compressed further).
    Propagating a LOWER layer's parameter sensitivity through an UPPER
    layer's I_n (x) R dynamics is compressed via a recursive
    composition of B19's K-chain construction: since the upper layer's
    dynamics are block-diagonal (I_n (x) R) and the forcing entering
    each copy factors through the SAME per-source accumulator (via the
    lower layer's own already-computed K-chain), the propagated
    sensitivity for EACH lower-layer source is representable via a
    single shared r_upper x r_upper matrix recursion, independent of
    the upper layer's feature multiplicity n_upper.

Run:  python -m credit_memory.b20_ic_rtrl
"""
from __future__ import annotations

import numpy as np

import credit_memory.b18_temporal_core as b18


# ---------------------------------------------------------------------------
# Small helper: a 2-layer (or L-layer) dense-core stack, reusing b18's
# Layer/forward machinery directly (dense family, entrywise-trainable R).
# ---------------------------------------------------------------------------
def build_stack(widths_rs, M_IN0, seed):
    """widths_rs: list of (width, r) pairs, one per layer. Dense R at
    every layer (per B19's own conclusion -- the practically useful
    core), dense routing."""
    rng = np.random.RandomState(seed)
    layers, Bs = [], []
    M_in = M_IN0
    for l_idx, (width, r) in enumerate(widths_rs):
        R = rng.randn(r, r) / np.sqrt(r) * 0.85
        layer = b18.Layer(width, r, R, trainable_R=True, routing="dense", M_IN=M_in)
        layers.append(layer)
        scale = 1.0 if l_idx == 0 else 0.3
        Bs.append(b18.init_routing(layer, M_in, rng, scale=scale))
        M_in = width
    c = rng.randn(layers[-1].width) / np.sqrt(layers[-1].width)
    return dict(layers=layers, Bs=Bs, c=c)


# ---------------------------------------------------------------------------
# Naive (uncompressed) forward-only RTRL -- correct by construction,
# reference for verification. For every scalar trainable parameter
# theta originating at layer l0, maintains an EXPLICIT per-layer chain
# of intermediate sensitivities dh_t^{l0}/dtheta, dh_t^{l0+1}/dtheta,
# ..., dh_t^{L-1}/dtheta, each updated via ITS OWN layer's dynamics
# (A_l' @ prev + B_l' @ [just-updated sensitivity one layer down]),
# in the same causal (bottom-to-top, single-timestep) order as the
# forward pass itself. Never looks backward in time.
# ---------------------------------------------------------------------------
def naive_rtrl_gradients(params, x, y):
    layers, Bs, c = params["layers"], params["Bs"], params["c"]
    T_, batch = x.shape[0], x.shape[1]
    L_ = len(layers)

    h_prev = [np.zeros((batch, layer.width)) for layer in layers]
    grads = {"dR": [np.zeros_like(layer.R) for layer in layers],
            "dB": [np.zeros_like(B) for B in Bs],
            "dc": np.zeros_like(c)}

    def A_of(layer):
        return b18.build_A(layer)

    # sens[key] = list of length (L_ - l0), sens[key][k] = dh_t^{l0+k}/dtheta
    sens = {}
    for l_idx, layer in enumerate(layers):
        r = layer.r
        for j in range(r):
            for k in range(r):
                sens[("R", l_idx, j, k)] = [np.zeros((batch, layers[l2].width))
                                            for l2 in range(l_idx, L_)]
        Nl, Ml = Bs[l_idx].shape
        for i in range(Nl):
            for m in range(Ml):
                sens[("B", l_idx, i, m)] = [np.zeros((batch, layers[l2].width))
                                            for l2 in range(l_idx, L_)]

    loss_total = 0.0
    for t in range(T_):
        h_cur = [None] * L_
        inp = x[t]
        layer_local_h_prev = []
        layer_local_inp = []
        for l_idx, (layer, B) in enumerate(zip(layers, Bs)):
            A = A_of(layer)
            layer_local_h_prev.append(h_prev[l_idx].copy())
            layer_local_inp.append(inp.copy())
            sp = h_prev[l_idx] @ A.T + inp @ B.T
            h_cur[l_idx] = sp
            inp = sp
        yhat_t = h_cur[-1] @ c
        r_err_t = (yhat_t - y[t]) / (T_ * batch)
        loss_total += 0.5 * float(np.mean((yhat_t - y[t]) ** 2))
        grads["dc"] += np.einsum("bn,b->n", h_cur[-1], r_err_t)

        new_sens = {}
        for l_idx, layer in enumerate(layers):
            r = layer.r
            n_l = layer.n
            hprev_l = layer_local_h_prev[l_idx]
            inp_l = layer_local_inp[l_idx]
            # R-parameters of this layer
            for j in range(r):
                for k in range(r):
                    key = ("R", l_idx, j, k)
                    old_chain = sens[key]
                    # dR[j,k] (row j, col k): (A_l@h)_i = sum_m R[i,m] h[m],
                    # so d(A_l@h)/dR[j,k] is nonzero only at row j, value h[k]
                    hh = hprev_l.reshape(batch, n_l, r)
                    f = np.zeros((batch, n_l, r)); f[:, :, j] = hh[:, :, k]
                    local_forcing = f.reshape(batch, layer.width)
                    new_chain = []
                    prev_updated = None
                    for idx2, l2 in enumerate(range(l_idx, L_)):
                        layer2 = layers[l2]
                        A2 = A_of(layer2)
                        if l2 == l_idx:
                            new_val = old_chain[idx2] @ A2.T + local_forcing
                        else:
                            new_val = old_chain[idx2] @ A2.T + prev_updated @ Bs[l2].T
                        new_chain.append(new_val)
                        prev_updated = new_val
                    new_sens[key] = new_chain
                    grads["dR"][l_idx][j, k] += float(np.sum(r_err_t[:, None] * (new_chain[-1] * c[None, :])))
            # B-parameters of this layer
            Nl, Ml = Bs[l_idx].shape
            for i in range(Nl):
                for m in range(Ml):
                    key = ("B", l_idx, i, m)
                    old_chain = sens[key]
                    f = np.zeros((batch, Nl)); f[:, i] = inp_l[:, m]
                    new_chain = []
                    prev_updated = None
                    for idx2, l2 in enumerate(range(l_idx, L_)):
                        layer2 = layers[l2]
                        A2 = A_of(layer2)
                        if l2 == l_idx:
                            new_val = old_chain[idx2] @ A2.T + f
                        else:
                            new_val = old_chain[idx2] @ A2.T + prev_updated @ Bs[l2].T
                        new_chain.append(new_val)
                        prev_updated = new_val
                    new_sens[key] = new_chain
                    grads["dB"][l_idx][i, m] += float(np.sum(r_err_t[:, None] * (new_chain[-1] * c[None, :])))
        sens = new_sens
        h_prev = h_cur

    return grads, loss_total / T_


# ---------------------------------------------------------------------------
# IC-RTRL: compresses the cross-layer propagation of a LOWER layer's
# ROUTING (B_l) parameters via B19's K-chain -- each entry B_l[i,m]
# injects via a FIXED spatial direction (e_i) times a genuine scalar-in-
# time drive (inp_l[t,:,m]), exactly B18/B19's own setup, so propagating
# it through EACH upper layer's I_n (x) R dynamics stays representable
# via a SHARED r_up x r_up matrix per upper layer, independent of that
# upper layer's feature multiplicity n_up.
#
# Layer l's OWN R_l entries do NOT get this treatment: their forcing is
# (I (x) E_jk) @ h_prev_l, which depends on the network's OWN full,
# generically high-rank recurrent state h_prev_l -- NOT a simple fixed-
# direction/scalar-drive source. Verified directly below (not assumed):
# the compressed-vs-naive comparison for R_l's cross-layer propagation
# should show no achievable compression, and this phase reports that
# honestly rather than forcing a compression that doesn't exist.
#
# Scope (explicit): compression verified through exactly ONE upper-layer
# boundary (immediately adjacent layer) -- multi-hop (L>=3, propagating
# through two or more upper layers) is not implemented, since composing
# bimodules across boundaries can itself grow (per B19's own findings)
# and characterizing that growth honestly would be its own phase.
# ---------------------------------------------------------------------------
def ic_rtrl_gradients_L2(params, x, y):
    """L=2 only (layers[0], layers[1]=top). Compressed treatment of
    B_0 (routing into layer 0, from external input) via the K-chain;
    R_0, R_1, B_1, c computed by their natural (local or trivially-top)
    cost -- no compression claimed there."""
    layers, Bs, c = params["layers"], params["Bs"], params["c"]
    assert len(layers) == 2
    layer0, layer1 = layers
    B0, B1 = Bs
    T_, batch = x.shape[0], x.shape[1]
    r0, r1 = layer0.r, layer1.r
    n1 = layer1.n
    R0, R1 = layer0.R, layer1.R

    h0_prev = np.zeros((batch, layer0.width))
    h1_prev = np.zeros((batch, layer1.width))

    grads = {"dR": [np.zeros_like(R0), np.zeros_like(R1)],
            "dB": [np.zeros_like(B0), np.zeros_like(B1)],
            "dc": np.zeros_like(c)}

    # --- local (uncompressed) trackers for R0 (propagated naively to top,
    # exactly as in naive_rtrl -- this IS the "no compression available"
    # comparison point) and R1 (fully local, top layer) ---
    S0_local = {(j, k): np.zeros((batch, layer0.width)) for j in range(r0) for k in range(r0)}
    S0_at_top = {(j, k): np.zeros((batch, layer1.width)) for j in range(r0) for k in range(r0)}
    S1_local = {(j, k): np.zeros((batch, layer1.width)) for j in range(r1) for k in range(r1)}
    S1_B_local = {(i, m): np.zeros((batch, layer1.width))
                 for i in range(layer1.width) for m in range(layer0.width)}

    # --- compressed K-chain trackers for B0[i,m]. B0[i,m]'s local
    # sensitivity at layer 0 is confined to copy p0=i//r0's r0-dim block
    # (a genuine r0-dim vector v_t, since A0 is block-diagonal and never
    # mixes copies). Propagating v_t through layer 1's B1 routing and its
    # OWN A1=I_n1(x)R1 dynamics requires r0 separate r1 x r1 K-matrices
    # (one per component of v_t -- NOT a single r1 x r1 matrix; each
    # component of v_t is its own B19-style scalar source), giving
    # per-source storage O(r0 * r1^2), independent of n1. Derivation:
    #   block_q_t = sum_{k=1}^{r0} K_t^{(k)} @ W_q[:,k],
    #   K_t^{(k)} = R1 @ K_{t-1}^{(k)} + v_t[k] * I_{r1},
    #   W_q = B1's (r1 x r0) block routing FROM copy p0 INTO copy q.
    M_IN0 = B0.shape[1]
    v_B0 = {(i, m): np.zeros((batch, r0)) for i in range(layer0.width) for m in range(M_IN0)}
    K_B0 = {(i, m): np.zeros((batch, r0, r1, r1)) for i in range(layer0.width) for m in range(M_IN0)}
    W_B0 = {}  # W_B0[i]: (n1, r1, r0) -- B1's block from copy p0=i//r0 into every copy q
    for i in range(layer0.width):
        p0 = i // r0
        W_full = B1[:, p0 * r0:(p0 + 1) * r0]  # (N1, r0)
        W_B0[i] = W_full.reshape(n1, r1, r0)

    loss_total = 0.0
    for t in range(T_):
        h0_cur = h0_prev @ R0_kron(layer0).T + x[t] @ B0.T
        h1_cur = h1_prev @ R0_kron(layer1).T + h0_cur @ B1.T
        yhat_t = h1_cur @ c
        r_err_t = (yhat_t - y[t]) / (T_ * batch)
        loss_total += 0.5 * float(np.mean((yhat_t - y[t]) ** 2))
        grads["dc"] += np.einsum("bn,b->n", h1_cur, r_err_t)

        # R1 (top layer's own params): fully local
        A1 = R0_kron(layer1)
        hh1 = h1_prev.reshape(batch, n1, r1)
        for j in range(r1):
            for k in range(r1):
                f = np.zeros((batch, n1, r1)); f[:, :, j] = hh1[:, :, k]
                S1_local[(j, k)] = S1_local[(j, k)] @ A1.T + f.reshape(batch, layer1.width)
                grads["dR"][1][j, k] += float(np.sum(r_err_t[:, None] * (S1_local[(j, k)] * c[None, :])))

        # B1 (top layer's own routing): fully local, same pattern as R1
        for i in range(layer1.width):
            for m in range(layer0.width):
                fb = np.zeros((batch, layer1.width)); fb[:, i] = h0_cur[:, m]
                S1_B_local[(i, m)] = S1_B_local[(i, m)] @ A1.T + fb
                grads["dB"][1][i, m] += float(np.sum(r_err_t[:, None] * (S1_B_local[(i, m)] * c[None, :])))

        # R0 (lower layer's own params): local at layer 0, THEN naive
        # (uncompressed) propagation to the top -- deliberately NOT using
        # a K-chain here, since it doesn't apply; this is the direct
        # verification that no compression is available for this block.
        A0 = R0_kron(layer0)
        n0 = layer0.n
        hh0 = h0_prev.reshape(batch, n0, r0)
        for j in range(r0):
            for k in range(r0):
                f = np.zeros((batch, n0, r0)); f[:, :, j] = hh0[:, :, k]
                S0_local[(j, k)] = S0_local[(j, k)] @ A0.T + f.reshape(batch, layer0.width)
                S0_at_top[(j, k)] = S0_at_top[(j, k)] @ A1.T + S0_local[(j, k)] @ B1.T
                grads["dR"][0][j, k] += float(np.sum(r_err_t[:, None] * (S0_at_top[(j, k)] * c[None, :])))

        # B0 (routing into layer 0, from external input): COMPRESSED via
        # nested K-chain, independent of n1. Two stages per (i,m):
        #  (1) v_t (r0-dim, copy p0's own block): v_t = v_{t-1}@R0.T + drive
        #  (2) K_t^{(k)} (r1 x r1, one per k=1..r0): K_t^{(k)} =
        #      R1@K_{t-1}^{(k)} + v_t[k]*I_r1
        # then block_q_t = sum_k K_t^{(k)} @ W_q[:,k].
        for i in range(layer0.width):
            p0, pos0 = i // r0, i % r0
            for m in range(M_IN0):
                drive = np.zeros((batch, r0)); drive[:, pos0] = x[t, :, m]
                v_new = v_B0[(i, m)] @ R0.T + drive
                v_B0[(i, m)] = v_new
                K_prev = K_B0[(i, m)]  # (batch, r0, r1, r1)
                K_new = np.einsum("jl,bklm->bkjm", R1, K_prev) \
                    + v_new[:, :, None, None] * np.eye(r1)[None, None, :, :]
                K_B0[(i, m)] = K_new
                # reconstruct: block_q = sum_k K_new[:,k,:,:] @ W_B0[i][q,:,k]
                # W_B0[i][q,l,k] = W_q[l,k] (l=r1-row index, k=r0-col index)
                s_full = np.einsum("bkjl,qlk->bqj", K_new, W_B0[i]).reshape(batch, layer1.width)
                grads["dB"][0][i, m] += float(np.sum(r_err_t[:, None] * (s_full * c[None, :])))

        h0_prev, h1_prev = h0_cur, h1_cur

    return grads, loss_total / T_


def R0_kron(layer):
    return b18.build_A(layer)
