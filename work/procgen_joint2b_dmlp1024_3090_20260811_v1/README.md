# Strict sampled joint 2B DMLP1024 Procgen rerun

Formal matched comparison on four environments and seeds 0, 1, 2.

- shared actor-critic ResNet plus decision MLP `256 -> 1024 -> 256`
- Gaussian sampled critic score and paired-noise critic RHS
- strict stacked `2B x 2B` system with actor-critic cross blocks retained
- rollout `4096`, minibatch `512`, epochs `4`, transitions `6M`
- critic curvature coefficient `0.1`, critic objective coefficient `1.0`
- damping `0.5`, FP64 solve, SGD momentum `0`, Kaczmarz disabled
- Procgen rollout-level adaptive KL controller with lower `0.005`, upper `0.04`

GPU mapping: seeds 0 and 1 occupy all eight GPUs initially; seeds 2 continue
on GPUs 0-3 after the matched seed-0 environment finishes.
