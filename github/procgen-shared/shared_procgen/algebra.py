"""Exact per-layer FP32 contractions; reduced systems are solved in FP64.

Extracted unchanged from the frozen shared normal-batch implementation.
Conv scores exist only one layer at a time, never as a full-network B x P array.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _is_supported_parameter_layer(module):
    return isinstance(module, (nn.Linear, nn.Conv2d)) or (
        module.__class__.__name__ == "PopArt"
        and hasattr(module, "weight")
        and hasattr(module, "bias")
    )


class LayerwiseIO:
    """Capture one batched forward's exact parameter-layer inputs/outputs."""

    def __init__(self, model):
        self.model = model
        self.records = OrderedDict()
        self._handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if not name or not _is_supported_parameter_layer(module):
                continue
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                continue

            def save_io(current_module, args, output, current_name=name):
                if current_name in self.records:
                    raise RuntimeError(
                        "exact layerwise Gram requires one call per parameter "
                        f"layer; repeated module: {current_name}"
                    )
                if (
                    len(args) != 1
                    or not torch.is_tensor(args[0])
                    or not torch.is_tensor(output)
                ):
                    raise TypeError(
                        f"unsupported layer call signature: {current_name}"
                    )
                self.records[current_name] = (current_module, args[0], output)

            self._handles.append(module.register_forward_hook(save_io))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def factors(self, per_example_scalar, retain_graph=True):
        if per_example_scalar.ndim != 1:
            raise ValueError("per_example_scalar must have shape [batch]")
        outputs = [record[2] for record in self.records.values()]
        deltas = torch.autograd.grad(
            per_example_scalar.sum(),
            outputs,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        result = OrderedDict()
        for (name, (module, layer_input, _)), delta in zip(
            self.records.items(), deltas
        ):
            if layer_input.dtype != torch.float32:
                raise TypeError(
                    f"Task222 requires FP32 layer input at {name}, got "
                    f"{layer_input.dtype}"
                )
            if delta is not None and delta.dtype != torch.float32:
                raise TypeError(
                    f"Task222 requires FP32 output delta at {name}, got "
                    f"{delta.dtype}"
                )
            result[name] = (module, layer_input, delta)
        return result

    def assert_parameter_coverage(self):
        covered = set()
        for name, (module, _, _) in self.records.items():
            for local_name, parameter in module.named_parameters(recurse=False):
                if parameter.requires_grad:
                    covered.add(f"{name}.{local_name}")
        expected = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if covered != expected:
            missing = sorted(expected - covered)
            extra = sorted(covered - expected)
            raise RuntimeError(
                "exact layerwise parameter coverage mismatch; "
                f"missing={missing}, extra={extra}"
            )


def _flatten_batch(tensor):
    return tensor.reshape(tensor.shape[0], -1)


@torch.no_grad()
def exact_selected_kernel_projections_fp32(factors, parameter_directions):
    """Return ``H H.T`` and several ``H @ direction`` vectors in one pass.

    The convolution score rows already required by the selected Gram are
    reused for all parameter-space projections.  As in
    :func:`exact_projection_mixed`, every layer contraction is FP32 and each
    completed per-layer projection vector is converted once to FP64 before the
    cross-layer sum.  No selected ``q x P`` matrix survives the current layer.
    """
    if not parameter_directions:
        raise ValueError("parameter_directions must be nonempty")
    if not factors:
        raise RuntimeError("no factors supplied")
    sample = next(iter(factors.values()))[1]
    batch = int(sample.shape[0])
    direction_count = len(parameter_directions)
    kernel = torch.zeros(
        (batch, batch), device=sample.device, dtype=torch.float32
    )
    projections = torch.zeros(
        (batch, direction_count), device=sample.device, dtype=torch.float64
    )
    active = False

    for name, (module, layer_input, delta) in factors.items():
        if delta is None:
            continue
        active = True
        weights = []
        biases = []
        for direction in parameter_directions:
            weight = direction.get(f"{name}.weight")
            bias = direction.get(f"{name}.bias")
            if weight is None:
                raise RuntimeError(f"missing direction for {name}.weight")
            if (
                layer_input.dtype != torch.float32
                or delta.dtype != torch.float32
                or weight.dtype != torch.float32
                or (bias is not None and bias.dtype != torch.float32)
            ):
                raise TypeError("Task253 Gram/projection operands must remain FP32")
            weights.append(weight)
            biases.append(bias)

        has_bias = module.bias is not None and module.bias.requires_grad
        if has_bias and any(bias is None for bias in biases):
            raise RuntimeError(f"missing direction for {name}.bias")

        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise NotImplementedError("grouped convolution is not supported")
            patches = F.unfold(
                layer_input,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            ).transpose(1, 2)
            delta_flat = delta.flatten(2)
            if patches.shape[1] != delta_flat.shape[2]:
                raise RuntimeError("convolution input/output spatial mismatch")
            rows = torch.bmm(delta_flat, patches).flatten(1)
            kernel.addmm_(rows, rows.T)
            packed_weights = torch.stack(
                [weight.reshape(-1) for weight in weights], dim=1
            )
            current = rows @ packed_weights
            if has_bias:
                bias_rows = delta_flat.sum(dim=2)
                kernel.addmm_(bias_rows, bias_rows.T)
                packed_biases = torch.stack(
                    [bias.reshape(-1) for bias in biases], dim=1
                )
                current.addmm_(bias_rows, packed_biases)
        else:
            input_2d = _flatten_batch(layer_input)
            delta_2d = _flatten_batch(delta)
            delta_gram = delta_2d @ delta_2d.T
            kernel.add_((input_2d @ input_2d.T) * delta_gram)
            if has_bias:
                kernel.add_(delta_gram)
            packed_weights = torch.stack(weights, dim=0)
            directional = torch.einsum(
                "bi,moi->bmo", input_2d, packed_weights
            )
            if has_bias:
                directional = directional + torch.stack(
                    biases, dim=0
                ).unsqueeze(0)
            current = torch.einsum("bo,bmo->bm", delta_2d, directional)

        if current.dtype != torch.float32:
            raise TypeError("Task253 per-layer projections must remain FP32")
        projections.add_(current.to(torch.float64))

    if not active:
        raise RuntimeError("no active parameter layers in fused Gram/projection")
    return kernel, projections


@torch.no_grad()
def exact_selected_kernel_tail_statistics_vectorized_fp32(
    factors,
    selected_indices,
    tail_indices,
    tail_weight_sets,
):
    """Vectorized multi-RHS equivalent of selected Gram/tail statistics."""
    if selected_indices.ndim != 1 or tail_indices.ndim != 1:
        raise ValueError("selected/tail indices must be one-dimensional")
    if tail_weight_sets.ndim != 2:
        raise ValueError("tail_weight_sets must have shape [tail, rhs]")
    if tail_weight_sets.shape[0] != tail_indices.numel():
        raise ValueError("tail index/weight size mismatch")
    if tail_weight_sets.dtype != torch.float32:
        raise TypeError("tail weights must remain FP32")
    if not factors:
        raise RuntimeError("no factors supplied")

    sample = next(iter(factors.values()))[1]
    q = int(selected_indices.numel())
    rhs_count = int(tail_weight_sets.shape[1])
    kernel = torch.zeros((q, q), device=sample.device, dtype=torch.float32)
    cross = torch.zeros((q, rhs_count), device=sample.device, dtype=torch.float64)
    norm2 = torch.zeros((rhs_count,), device=sample.device, dtype=torch.float64)
    active = False

    for module, layer_input, delta in factors.values():
        if delta is None:
            continue
        active = True
        if layer_input.dtype != torch.float32 or delta.dtype != torch.float32:
            raise TypeError("vectorized ResNet factors must remain FP32")
        has_bias = module.bias is not None and module.bias.requires_grad

        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise NotImplementedError("grouped convolution is not supported")
            patches = F.unfold(
                layer_input,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            ).transpose(1, 2)
            delta_flat = delta.flatten(2)
            if patches.shape[1] != delta_flat.shape[2]:
                raise RuntimeError("convolution input/output spatial mismatch")
            selected_patches = patches.index_select(0, selected_indices)
            selected_delta = delta_flat.index_select(0, selected_indices)
            selected_rows = torch.bmm(
                selected_delta, selected_patches
            ).flatten(1)
            kernel.addmm_(selected_rows, selected_rows.T)

            tail_patches = patches.index_select(0, tail_indices)
            tail_delta = delta_flat.index_select(0, tail_indices)
            base_weights = torch.einsum(
                "bm,bol,bli->moi",
                tail_weight_sets,
                tail_delta,
                tail_patches,
            ).flatten(1)
            cross.add_((selected_rows @ base_weights.T).to(torch.float64))
            norm2.add_(base_weights.to(torch.float64).square().sum(dim=1))

            if has_bias:
                selected_bias_rows = selected_delta.sum(dim=2)
                kernel.addmm_(selected_bias_rows, selected_bias_rows.T)
                tail_bias_rows = tail_delta.sum(dim=2)
                base_biases = torch.einsum(
                    "bm,bo->mo", tail_weight_sets, tail_bias_rows
                )
                cross.add_(
                    (selected_bias_rows @ base_biases.T).to(torch.float64)
                )
                norm2.add_(
                    base_biases.to(torch.float64).square().sum(dim=1)
                )
        else:
            input_2d = _flatten_batch(layer_input)
            delta_2d = _flatten_batch(delta)
            selected_input = input_2d.index_select(0, selected_indices)
            selected_delta = delta_2d.index_select(0, selected_indices)
            delta_gram = selected_delta @ selected_delta.T
            kernel.add_((selected_input @ selected_input.T) * delta_gram)
            if has_bias:
                kernel.add_(delta_gram)

            tail_input = input_2d.index_select(0, tail_indices)
            tail_delta = delta_2d.index_select(0, tail_indices)
            base_weights = torch.einsum(
                "bm,bo,bi->moi", tail_weight_sets, tail_delta, tail_input
            )
            directional = torch.einsum(
                "bi,moi->bmo", selected_input, base_weights
            )
            base_biases = None
            if has_bias:
                base_biases = torch.einsum(
                    "bm,bo->mo", tail_weight_sets, tail_delta
                )
                directional = directional + base_biases.unsqueeze(0)
            cross.add_(
                torch.einsum("bo,bmo->bm", selected_delta, directional).to(
                    torch.float64
                )
            )
            norm2.add_(
                base_weights.to(torch.float64).square().sum(dim=(1, 2))
            )
            if base_biases is not None:
                norm2.add_(
                    base_biases.to(torch.float64).square().sum(dim=1)
                )

    if not active:
        raise RuntimeError("no active parameter layers in vectorized statistics")
    return kernel, cross, norm2


def exact_weighted_directions_vectorized_fp32(
    factors, weight_sets, denominator=1.0
):
    """Vectorized multi-RHS ``H.T @ weights`` with one unfold per layer."""
    if weight_sets.ndim != 2:
        raise ValueError("weight_sets must have shape [batch, directions]")
    if weight_sets.dtype != torch.float32:
        raise TypeError("reconstruction weights must remain FP32")
    if not factors:
        raise RuntimeError("no factors supplied")
    batch = next(iter(factors.values()))[1].shape[0]
    if weight_sets.shape[0] != batch:
        raise ValueError("factor/weight batch size mismatch")
    directions = [OrderedDict() for _ in range(weight_sets.shape[1])]
    divisor = float(denominator)
    for name, (module, layer_input, delta) in factors.items():
        if delta is None:
            continue
        if layer_input.dtype != torch.float32 or delta.dtype != torch.float32:
            raise TypeError("reconstruction factors must remain FP32")
        if isinstance(module, nn.Conv2d):
            patches = F.unfold(
                layer_input,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            ).transpose(1, 2)
            delta_flat = delta.flatten(2)
            packed_weights = torch.einsum(
                "bm,bol,blq->moq", weight_sets, delta_flat, patches
            )
            packed_biases = (
                torch.einsum("bm,bol->mo", weight_sets, delta_flat)
                if module.bias is not None and module.bias.requires_grad
                else None
            )
        else:
            input_2d = _flatten_batch(layer_input)
            delta_2d = _flatten_batch(delta)
            packed_weights = torch.einsum(
                "bm,bo,bi->moi", weight_sets, delta_2d, input_2d
            )
            packed_biases = (
                torch.einsum("bm,bo->mo", weight_sets, delta_2d)
                if module.bias is not None and module.bias.requires_grad
                else None
            )
        for index, direction in enumerate(directions):
            direction[f"{name}.weight"] = (
                packed_weights[index].reshape_as(module.weight) / divisor
            )
            if packed_biases is not None:
                direction[f"{name}.bias"] = packed_biases[index] / divisor
    return directions


def named_scale_fp32(direction, scale):
    scale_fp32 = torch.as_tensor(
        scale,
        device=next(iter(direction.values())).device,
        dtype=torch.float32,
    )
    return OrderedDict(
        (name, value * scale_fp32) for name, value in direction.items()
    )


def named_add_fp32(left, right):
    if set(left) != set(right):
        raise ValueError("named direction order mismatch")
    return OrderedDict((name, left[name] + right[name]) for name in left)


def named_dot_fp64(left, right):
    if set(left) != set(right):
        raise ValueError("named direction order mismatch")
    result = None
    for name in left:
        contribution = (
            left[name].to(torch.float64)
            * right[name].to(torch.float64)
        ).sum()
        result = contribution if result is None else result + contribution
    if result is None:
        raise RuntimeError("empty named direction")
    return result


def _conv_per_example_weight_grad_fp32(module, layer_input, delta):
    if module.groups != 1:
        raise NotImplementedError("grouped convolution is not supported")
    if layer_input.dtype != torch.float32 or delta.dtype != torch.float32:
        raise TypeError("Task222 convolution factors must remain FP32")
    patches = F.unfold(
        layer_input,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    ).transpose(1, 2)
    delta_flat = delta.flatten(2)
    if patches.shape[1] != delta_flat.shape[2]:
        raise RuntimeError("convolution input/output spatial mismatch")
    return torch.bmm(delta_flat, patches)


def exact_kernel_fp32(left, right=None):
    """Return exact ``H_left H_right.T`` using FP32 layer contractions."""
    self_kernel = right is None or right is left
    if right is None:
        right = left
    if tuple(left) != tuple(right):
        raise ValueError("factor layer order mismatch")
    kernel = None
    for name in left:
        module_l, input_l, delta_l = left[name]
        module_r, input_r, delta_r = right[name]
        if module_l is not module_r:
            raise ValueError(f"factor module mismatch at {name}")
        if delta_l is None or delta_r is None:
            continue
        if isinstance(module_l, nn.Conv2d):
            grad_l = _flatten_batch(
                _conv_per_example_weight_grad_fp32(
                    module_l, input_l, delta_l
                )
            )
            grad_r = grad_l if self_kernel else _flatten_batch(
                _conv_per_example_weight_grad_fp32(
                    module_r, input_r, delta_r
                )
            )
            contribution = grad_l @ grad_r.t()
            if module_l.bias is not None and module_l.bias.requires_grad:
                bias_l = delta_l.flatten(2).sum(dim=2)
                bias_r = bias_l if self_kernel else delta_r.flatten(2).sum(
                    dim=2
                )
                contribution = contribution + bias_l @ bias_r.t()
        else:
            input_l_2d = _flatten_batch(input_l)
            delta_l_2d = _flatten_batch(delta_l)
            input_r_2d = input_l_2d if self_kernel else _flatten_batch(input_r)
            delta_r_2d = delta_l_2d if self_kernel else _flatten_batch(delta_r)
            if any(
                value.dtype != torch.float32
                for value in (
                    input_l_2d, delta_l_2d, input_r_2d, delta_r_2d
                )
            ):
                raise TypeError("Task222 linear factors must remain FP32")
            delta_gram = delta_l_2d @ delta_r_2d.t()
            contribution = (input_l_2d @ input_r_2d.t()) * delta_gram
            if module_l.bias is not None and module_l.bias.requires_grad:
                contribution = contribution + delta_gram
        if contribution.dtype != torch.float32:
            raise TypeError("Task222 layer Gram contribution must be FP32")
        kernel = contribution if kernel is None else kernel + contribution
    if kernel is None:
        raise RuntimeError("no active parameter layers in exact kernel")
    if kernel.dtype != torch.float32:
        raise TypeError("Task222 accumulated Gram must be FP32")
    return kernel


def exact_projection_mixed(factors, parameter_direction):
    """Return ``H @ direction`` with FP32 layers and FP64 layer summation.

    Each layer's directional forward and elementwise contraction remain
    FP32.  The completed B-vector is cast once to FP64 before summation across
    layers, which protects the Eq.13 residual from cross-layer cancellation
    without moving the expensive layer work to FP64.
    """
    projection = None
    for name, (module, layer_input, delta) in factors.items():
        if delta is None:
            continue
        weight = parameter_direction.get(f"{name}.weight")
        bias = parameter_direction.get(f"{name}.bias")
        if weight is None:
            raise RuntimeError(f"missing direction for {name}.weight")
        if (
            layer_input.dtype != torch.float32
            or delta.dtype != torch.float32
            or weight.dtype != torch.float32
            or (bias is not None and bias.dtype != torch.float32)
        ):
            raise TypeError("Task222 projection operands must remain FP32")
        if isinstance(module, nn.Conv2d):
            directional_output = F.conv2d(
                layer_input,
                weight,
                bias,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
        else:
            directional_output = F.linear(layer_input, weight, bias)
        current = _flatten_batch(
            delta * directional_output
        ).sum(dim=1).to(torch.float64)
        projection = current if projection is None else projection + current
    if projection is None:
        raise RuntimeError("no active parameter layers in exact projection")
    if projection.dtype != torch.float64:
        raise TypeError("Task222 cross-layer projection sum must be FP64")
    return projection
