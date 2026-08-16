#!/usr/bin/env python3
"""Unit checks for the unified low-Fisher/block-imbalance actor guard."""

import importlib.util
import math
from pathlib import Path
import sys


TRAINER = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(__file__).with_name("train_shared_joint2b_block_damping.py")
)
sys.path.insert(0, str(TRAINER.parent))
SPEC = importlib.util.spec_from_file_location("joint2b_block_guard", TRAINER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def evaluate(fisher, ratio):
    return MODULE.block_imbalance_guard_damping(
        fisher,
        ratio,
        0.70,
        0.40,
        1.50,
        4.00,
        0.03,
        0.50,
    )


def main():
    # Either condition in isolation must leave the healthy path unchanged.
    assert evaluate(0.80, 8.0) == (0.03, 0.0, False)
    assert evaluate(0.20, 1.2) == (0.03, 0.0, False)

    # CaveFlyer's pre-collapse window activates smoothly, not at max ridge.
    damping, fraction, active = evaluate(0.652, 1.51)
    assert active
    assert 0.03 < damping < 0.15
    assert math.isclose(damping, 0.03 + fraction * 0.47)

    # A jointly severe low-Fisher/large-block-imbalance state reaches max.
    damping, fraction, active = evaluate(0.40, 4.0)
    assert active and fraction == 1.0 and damping == 0.50
    print("PASS: conjunctive activation and smooth actor damping")


if __name__ == "__main__":
    main()
