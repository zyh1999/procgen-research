# PAPER_MATCHED_DETERMINISTIC_GGN_V1 frozen launch package

This directory is generated from the exact original Paper RAT source
`cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`.
Only `Advantage_Update` is replaced by the deterministic critic-GGN joint-2B
FP64/Jacobi/direct solve, plus health telemetry. The config retains the Paper
actor/network/schedule fields and changes only the budget to 1M while adding
the frozen critic curvature coefficient `0.1` and solver controls.

Run `python3 audit_paper_matched_diff.py` and
`python3 test_paper_matched_deterministic_ggn_v1.py` before deployment.
