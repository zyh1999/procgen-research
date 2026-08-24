# Frozen manifest

Task `PROCGEN-PAPER-SEPARATEB-DETGGN-6M-S0-20260824-07`; one method only:
`PAPER_MATCHED_SEPARATE_B_DET_GGN_V1`.

Four cells: BigFish, BossFight, CaveFlyer, CoinRun; seed0; intended 6,000,000;
last complete update 5,980,160. Roots are
`/scratch/h99859yz/procgen_paper_separateb_detggn_6m_s0_20260824_07/runs/PAPER_MATCHED_SEPARATE_B_DET_GGN_V1/<env>/seed0/6m`.

Paper seed0 reuse IDs are BigFish `1063880_0/1064035`, BossFight
`1063880_5/1064047`, CaveFlyer `1063880_10/1064067`, CoinRun
`1063880_15/1064074`. The monitor accepts only their exact progress rows and
never substitutes a 6M terminal row for an intermediate target.

Scheduler placement: CSF3 gpuH, one H200 per independently auditable sbatch,
at most four concurrent cells, 8 CPUs/GPU, no Jupyter, no requeue/retry.
