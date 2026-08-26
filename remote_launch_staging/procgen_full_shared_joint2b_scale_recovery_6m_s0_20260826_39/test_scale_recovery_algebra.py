#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import types

import torch

# Stubs make the pure solver importable without the remote Procgen runtime.
sys.modules.setdefault("procgen", types.SimpleNamespace(ProcgenEnv=object))
sys.modules.setdefault("utils", types.ModuleType("utils"))
for name in ("utils.logger", "utils.runners", "utils.utils", "vec_env"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["utils.runners"].Runner = object
for key in ("build_cnn", "build_resnet", "build_mlp", "SharedActorCritic",
            "count_vars", "safemean", "set_seed", "set_grads_from_flat"):
    setattr(sys.modules["utils.utils"], key, object)
for key in ("VecExtractDictObs", "VecMonitor", "VecNormalize"):
    setattr(sys.modules["vec_env"], key, object)

path = Path(__file__).with_name("train_full_shared_joint2b_scale_recovery_v1.py")
spec = importlib.util.spec_from_file_location("task39", path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

torch.manual_seed(39)
b, p = 16, 41
a = torch.randn(b, p, dtype=torch.float64)
c = torch.randn(b, p, dtype=torch.float64)
bpi = torch.randn(b, dtype=torch.float64)
bv = torch.randn(b, dtype=torch.float64)
d, info = module.solve_full_shared_joint2b_scale_recovery(a, c, bpi, bv)
assert info["h_bar"].shape == (2 * b, p)
assert torch.count_nonzero(info["gram"][:b, b:]) > 0
torch.testing.assert_close(d, info["h_bar"].t() @ torch.linalg.solve(info["system"], info["b_bar"]))
for ca, cc in ((5., 5.), (11., 1.), (1., 13.), (.125, 16.)):
    d2, i2 = module.solve_full_shared_joint2b_scale_recovery(ca*a, cc*c, ca*bpi, cc*bv)
    torch.testing.assert_close(d2, d, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(i2["gram"], info["gram"], rtol=1e-12, atol=1e-12)
assert abs(float(info["actor_normalized_mean_gram_diag"]) - 1.) < 1e-12
assert abs(float(info["critic_normalized_mean_gram_diag"]) - 1.) < 1e-12
assert int(info["cholesky_info"]) == 0
assert float(info["relative_residual"]) < 1e-12
try:
    module.solve_full_shared_joint2b_scale_recovery(torch.zeros_like(a), c, bpi, bv)
except FloatingPointError:
    pass
else:
    raise AssertionError("zero actor scale was not rejected")
print("TASK39_ALGEBRA_PASS")
