# Procgen 专用 KL 自适应学习率阈值变更记录

- 日期：2026-07-20
- 独立任务目录：`/Users/user/Documents/procgen`
- 核对代码：`/Users/user/Documents/Codex/2026-07-17/kan/latest_trust_region_20260720_161619/trust-region-main`
- 边界：`0.02 * 2` 只供显式选择 Procgen 阈值的配置使用；公共默认仍为 `0.01 * 2`。
- 集群边界：未提交、取消或修改任何 Bede/CSF3 作业。

## 代码分支

文件：`train_shared.py`

- 第 544 行仍由 `use_kl_adaptive_lr` 控制是否启用整个控制器。
- 第 546 行使用兼容旧配置的显式开关：
  `getattr(algo_config, 'use_procgen_kl_thresholds', False)`。
- 第 547 行：开关为 `True` 时，`kl_upper = 0.02 * 2`。
- 第 548–549 行：未配置或配置为 `False` 时，公共默认 `kl_upper = 0.01 * 2`。
- 第 551 行统一执行 `curr_kl > kl_upper`。
- 第 552 行保持降学习率为当前学习率除以 `1.5`，下限 `1e-4`。
- 第 553 行保持低阈值 `curr_kl < 0.01 / 2`。
- 第 554 行保持升学习率乘 `1.5`，上限 `algo_config.lr`。

修改后 `train_shared.py` SHA-256：

`c6b7c08c7d4da6cb5977170c312c1e0db9a2fb6c0c2bf059f61053e34d6367f9`

## 配置边界

全目录精确反向检查确认，只有以下三份 YAML 包含
`use_procgen_kl_thresholds: True`：

1. `configs/adv_resnet_shared.yaml:9`
2. `configs/adv_vit_shared.yaml:9`
3. `configs/test_adv_resnet.yaml:12`

三份配置均同时满足：

- `algo: adv`
- `use_kl_adaptive_lr: True`
- `use_procgen_kl_thresholds: True`
- 默认环境名采用 Procgen 的 `*-easy-*` 格式

其他配置没有该字段，因 `getattr(..., False)` 自动走公共 `0.01 * 2` 上阈值，不会受到 Procgen 调参覆盖。

配置 SHA-256：

- `adv_resnet_shared.yaml`：`1b7a8e62415729b2954070ad9da43cca1012b626ed766054b68cab5dd6fae921`
- `adv_vit_shared.yaml`：`7158c74a500ce906006940ca8263b91b69e6fd6f8ba3d9466461aa5d8d1feee6`
- `test_adv_resnet.yaml`：`a6cd3c7714a15db48297ad43108888021b191c6769ed6ba4957d18f8e3ce0bdc`

## 复核结果

通过：

1. `train_shared.py` 的 Python AST 和 `py_compile` 检查。
2. 代码断言：不存在全局直写的 `if curr_kl > 0.02 * 2`；Procgen 与公共上阈值均存在且由独立开关选择。
3. 三份目标 YAML 均可解析，且开关和 Procgen ADV 条件正确。
4. 对 `configs/*.yaml` 做字段反向扫描，开启新开关的集合恰好等于上述三份文件。
5. 降/升学习率因子、上下限和低 KL 阈值均未改变。

## 已知但不在本次范围的问题

`Advantage_Update` 在 `train_shared.py:280` 仍返回硬编码的
`pi_info['kl'] = 0.0`。本次仅复核 Procgen 专用阈值边界，未修改或绕过该行为。
