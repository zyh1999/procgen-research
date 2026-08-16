# Procgen commit provenance correction

- 核对时间：2026-07-22 15:57 BST
- 用户指定 commit：`2b5affd64cbb3c624b4bc1f4767f449df231ffb2`
- 指定仓库：`https://github.com/agent-lab/trust-region`
- 桌面 clone：`/Users/user/Desktop/trust-region`
- 桌面 clone 当前 HEAD：`7a0698e1072f701735a87f660362357c3547ee63`
- 工作树：clean，跟踪 `origin/main`

## 结论

已完成的 PPO/Exact RAT runs 没有使用指定 commit `2b5affd...`。它们使用的是后续 HEAD/ZIP 版本及其 Procgen 专用补丁，因此现有曲线不能作为指定 commit 的正式结果。

`trust-region-main (3).zip` 不含 `.git` 或 commit metadata。其关键文件与桌面 clone 当前 HEAD 一致，但与目标 commit 不一致。

## 文件哈希

目标 commit `2b5affd...`：

- `train_shared.py`：`cbcd68118a2901fdcdf3bf2de55841d01b330e7a6cb38996ed8ba791eb2ab1e7`
- `configs/adv_resnet_shared.yaml`：`a849e0ced38bf0b3c59cbe49d92f9224cca58e156c44c45715c2c8e738e3fb84`
- `configs/ppo_resnet_shared.yaml`：`f1954fceec13b01af644140ae629bf8c3dec1275d44adc32a35268140915b12b`

实际 PPO snapshot：

- trainer：`1d20658b154022450b8598949f693b3c04a9bd34eb22ad2f002d59f9573b74d1`
- config：`fdf1538ef199a222ea2caafe9264c5db00319a6f1882d7d86b04506522601807`
- 对应桌面 clone 当前 HEAD 的关键文件，不对应目标 commit

实际 Exact RAT snapshot（4090 与 Bede 相同）：

- trainer：`f4cfcd3a5dd9ea84e9d7533a5f17c2d897db545a49d352850df89bdc69142369`
- config：`476b210d9da6e1dc973cf293d812a5c1e2f3c6f20654736a9687e397131da1ca`
- 这是后续 HEAD 基础上的真实 KL + Procgen KL threshold 分支，不对应目标 commit

## 影响正式比较的主要差异

从 `2b5affd...` 到实际使用的后续版本：

- SGD momentum：`1e-6` 变为 `0.1`
- Exact RAT 求解：两个独立 `torch.inverse` 变为一个联合 `torch.linalg.solve`
- KL adaptive LR：从每 minibatch 调整变为每 rollout/epoch 外调整
- easy horizon config：`3,000,000` 变为 `6,000,000`
- 后续版本还加入 Kaczmarz 开关、额外算法分支和日志/计时改动
- 实际 RAT 又恢复真实 rollout-behavior KL，并使用 Procgen 专用 upper `0.04`

这些差异会改变优化轨迹和预算，不能把现有结果重新标注为目标 commit。

## 当前操作边界

- 未删除或覆盖旧日志；旧结果保留为 `7a0698e` 派生版本证据。
- 未提交、取消或修改任何集群任务。
- 正确重跑前需要明确：使用目标 commit 原始 3M horizon，还是锁定目标 commit 代码但继续采用此前 formal 6M horizon。
