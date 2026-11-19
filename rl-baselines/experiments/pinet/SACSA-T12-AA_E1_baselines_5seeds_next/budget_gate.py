"""Positive Beta parameters with an algebraic negative tail.

Squareplus(x, 4) = (x + sqrt(x*x + 4))/2; inverse(y) = y - 1/y.
The rational negative branch avoids cancellation. Unlike clipped softplus it
has no dead raw-logit interval and decays polynomially on the negative side.
This changes optimization geometry, not the Beta family or A = H G P rule.
Reference: https://arxiv.org/abs/2112.11687
"""
import torch
from torch.nn import functional as F


def positive_gate(raw, minimum, maximum, logit_clip, parameterization):
    raw = torch.nan_to_num(raw, nan=0.0, posinf=logit_clip, neginf=-logit_clip)
    if parameterization == "softplus":
        value = F.softplus(raw.clamp(-logit_clip, logit_clip))
    elif parameterization == "squareplus":
        radius = torch.hypot(raw, torch.full_like(raw, 2.0))
        positive = 0.5 * raw + 0.5 * radius
        negative = (0.5 * radius - 0.5 * raw).clamp_min(1.0).reciprocal()
        value = torch.where(raw >= 0, positive, negative)
    else:
        raise ValueError(f"Unknown Beta gate parameterization: {parameterization}")
    return (value + minimum).clamp(minimum, maximum)


def inverse_positive_gate(target, minimum, logit_clip, parameterization):
    value = (target - minimum).clamp_min(1e-5)
    if parameterization == "squareplus":
        return value - value.reciprocal()
    if parameterization == "softplus":
        return (value + torch.log(-torch.expm1(-value))).clamp(-logit_clip, logit_clip)
    raise ValueError(f"Unknown Beta gate parameterization: {parameterization}")
