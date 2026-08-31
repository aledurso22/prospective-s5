"""B38a -- step-locked A/B PARAMETER and OPTIMIZER-STATE trajectory identity.

Both arms start from identical initialization and consume identical data in the
identical chunk order with identical Adam state. After every block of epochs we
compare the full parameter vector and both Adam moments (m, v)."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from credit_memory.b38a_train import (
    FAMILIES_ALL, L_VALUES, EPOCHS, EP_BLOCK, N_TRAIN, make_teacher_norm,
    data_for, build_epochs)
from credit_memory.b37c_productlocal_native import spec_from_blocks, generic_init, to_jax

def vec(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])

def dev(a, b):
    va, vb = vec(a), vec(b)
    return float(np.max(np.abs(va - vb)) / (1 + np.max(np.abs(vb))))

rows = []
print(f"{'family':21s} {'L':>4s} {'lr':>6s} {'max dev params':>15s} {'max dev Adam m':>15s} "
      f"{'max dev Adam v':>15s} {'blocks':>7s}")
for f in FAMILIES_ALL:
    for L in L_VALUES:
        t = make_teacher_norm(f, 8, 0)
        spec = spec_from_blocks(t["blocks"])
        (xtr, ytr), _, _ = data_for(t, f, 8, 0)
        p0 = to_jax(generic_init(spec, 0))
        z = jax.tree.map(jnp.zeros_like, p0)
        state = {a: [p0, z, z, jnp.array(0.0)] for a in ("A", "B")}
        runners = {a: build_epochs(spec, L, a, EP_BLOCK, N_TRAIN) for a in ("A", "B")}
        dp = dm = dv = 0.0
        nb = EPOCHS // EP_BLOCK
        for _ in range(nb):
            for a in ("A", "B"):
                state[a] = list(runners[a](*state[a], xtr, ytr, 3e-3))
            dp = max(dp, dev(state["A"][0], state["B"][0]))
            dm = max(dm, dev(state["A"][1], state["B"][1]))
            dv = max(dv, dev(state["A"][2], state["B"][2]))
        rows.append(dict(family=f, L=L, d_params=dp, d_m=dm, d_v=dv, blocks=nb))
        print(f"{f:21s} {L:4d} {3e-3:6.0e} {dp:15.3e} {dm:15.3e} {dv:15.3e} {nb:7d}")
a = np.array([[r["d_params"], r["d_m"], r["d_v"]] for r in rows])
print(f"\nWORST across all families/L:  params={a[:,0].max():.3e}  "
      f"Adam m={a[:,1].max():.3e}  Adam v={a[:,2].max():.3e}")
json.dump(rows, open("results/b38a/state_identity.json", "w"), indent=1)
