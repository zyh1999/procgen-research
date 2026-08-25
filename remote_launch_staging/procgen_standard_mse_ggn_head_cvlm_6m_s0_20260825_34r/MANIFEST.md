# Task34R frozen manifest

- Task: `PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-20260825-34R`
- Method: `DET_STANDARD_MSE_GGN_HEAD_CVLM_V1`
- Assignment: `52df68ca4c6def1d917778ab4faad2e7f0109c31`
- Campaign: `/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r`
- Matrix: BigFish, BossFight, CaveFlyer, CoinRun; seed 0; intended 6M.

Scientific identity: the Paper actor and sampled-critic direction remain on
policy-exclusive and shared parameters.  Only the 257 critic-exclusive value
head parameters use standard normalized-coordinate MSE GGN with `D=I`,
`W=I`, `K=J`.  The damping is selected by the frozen disjoint-next-minibatch
CVLM protocol.  Same-minibatch actual/predicted reduction is an identity check
only and never an LM acceptance input.

The exact byte hashes are recorded by `SHA256SUMS` after the implementation is
frozen.  Model and checkpoint files are never committed.
