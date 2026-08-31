"""B38a section 2 -- the registered identity, verified DURING ACTUAL TRAINING.

    grad^{ProductLocal RTRL} == grad^{matched TBPTT}   for every chunk

Parameters are fixed across each differentiation interval; the incoming hidden
state is stop-gradiented at every chunk boundary; eligibility is reset per
chunk; one update per boundary. The RTRL gradient is what drives the update."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from credit_memory.b38a_train import (
    FAMILIES_ALL, make_teacher_norm, data_for, build_epochs_check, batch_z0,
    EPOCHS, EP_BLOCK, N_TRAIN, L_VALUES)
from credit_memory.b37c_productlocal_native import spec_from_blocks, generic_init, to_jax

rows = []
print(f"{'family':21s} {'L':>4s} {'chunks':>7s} {'worst rel err u':>16s} "
      f"{'worst rel err b':>16s} {'worst rel err C_out':>19s}")
for f in FAMILIES_ALL:
    for L in L_VALUES:
        t = make_teacher_norm(f, 8, 0)
        spec = spec_from_blocks(t["blocks"])
        (xtr, ytr), _, _ = data_for(t, f, 8, 0)
        p = to_jax(generic_init(spec, 0))
        m = jax.tree.map(jnp.zeros_like, p); v = jax.tree.map(jnp.zeros_like, p)
        st = jnp.array(0.0)
        run = build_epochs_check(spec, L, EP_BLOCK, N_TRAIN)
        worst = np.zeros(3)
        for _ in range(EPOCHS // EP_BLOCK):
            p, m, v, st, e = run(p, m, v, st, xtr, ytr, 3e-3)
            worst = np.maximum(worst, np.asarray(e))
        nch = EPOCHS * (256 // L)
        rows.append(dict(family=f, L=L, chunks=nch, u=float(worst[0]),
                         b=float(worst[1]), C=float(worst[2])))
        print(f"{f:21s} {L:4d} {nch:7d} {worst[0]:16.3e} {worst[1]:16.3e} {worst[2]:19.3e}")
w = np.array([[r["u"], r["b"], r["C"]] for r in rows])
print(f"\nWORST over {sum(r['chunks'] for r in rows)} training chunks: "
      f"u={w[:,0].max():.3e}  b={w[:,1].max():.3e}  C_out={w[:,2].max():.3e}")
json.dump(rows, open("results/b38a/identity.json", "w"), indent=1)
