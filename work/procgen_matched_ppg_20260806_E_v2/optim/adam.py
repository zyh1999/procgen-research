import torch
from torch import Tensor

from torch.optim.optimizer import (
    _get_value,
    _use_grad_for_differentiable,
    Optimizer,
)

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        decoupled_weight_decay: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            differentiable=differentiable,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    # p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())
                    p_state["step"] = torch.tensor(step_val)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    # state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                    state["step"] = torch.tensor(0.0)

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            lr = group["lr"]
            weight_decay=group["weight_decay"]
            eps=group["eps"]
            maximize=group["maximize"]
            decoupled_weight_decay=group["decoupled_weight_decay"]

            # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
            # treating it as a scalar.
            # Note: ensure type declaration is under conditional check for isinstance
            # or else torchscript will get cranky about the DeviceDict type.
            beta1_dict = None

            for i, param in enumerate(params_with_grad):
                grad = grads[i] if not maximize else -grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                # update step
                step_t += 1

                if weight_decay != 0:
                    if decoupled_weight_decay:
                        # Perform stepweight decay
                        param.mul_(1 - lr * weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)

                if torch.is_complex(param):
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    exp_avg_sq = torch.view_as_real(exp_avg_sq)
                    param = torch.view_as_real(param)

                device = param.device

                if beta1_dict is not None:
                    dtype = param.dtype  # type: ignore[union-attr]

                    # cast to workaround https://github.com/pytorch/pytorch/issues/140601
                    key = (device, dtype)
                    if key not in beta1_dict:
                        beta1_dict[key] = beta1.to(device=device, dtype=dtype, non_blocking=True)  # type: ignore[union-attr]

                    device_beta1 = beta1_dict[key]
                else:
                    device_beta1 = beta1

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - device_beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = _get_value(step_t)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = bias_correction2**0.5

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss