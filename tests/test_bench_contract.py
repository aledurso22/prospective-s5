"""Fast contracts for S5 clipping and exact-credit run metadata.

Run:  python -m tests.test_bench_contract
"""
from __future__ import annotations

from train_bench import (clipping_instrumentation, credit_audit,
                         parse_args)


def main() -> None:
    default = parse_args(["--task", "copy", "--arm", "online"])
    assert default.clip == 0.0, "historical S5 default must remain unclipped"

    clipped = clipping_instrumentation([0.5, 1.0, 2.0, 4.0], 1.0)
    assert clipped["p_clip"] == 0.5
    assert clipped["chi"]["defined"]
    assert clipped["chi"]["p50"] == 1.5
    assert clipped["chi"]["max"] == 4.0

    unclipped = clipping_instrumentation([0.5, 2.0], 0.0)
    assert unclipped["p_clip"] == 0.0
    assert not unclipped["chi"]["defined"]
    assert unclipped["preclip_ratio"] is None

    for arm in ("routePC", "routePCreal", "routePCphase", "routePCadam"):
        audit = credit_audit(arm, 17)
        assert audit["exact_grad_calls"] == 0
        assert audit["exact_lambda_calls"] == 0
        assert audit["bptt_calls"] == 0
    assert credit_audit("baseline", 17)["bptt_calls"] == 17
    assert credit_audit("routeA", 17)["exact_grad_calls"] == 17
    assert credit_audit("tbptt", 17)["tbptt_calls"] == 17

    print("S5 benchmark clipping/audit contracts PASS")


if __name__ == "__main__":
    main()
