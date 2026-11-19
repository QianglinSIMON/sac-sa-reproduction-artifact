"""PiNet's Douglas--Rachford projection and implicit VJP, for a budget simplex.

Reference: Grontas et al., arXiv:2508.10480v2, Eqs. (4), (8), (9).
This is a PyTorch budget specialization, not the authors' JAX package.
Lift x>=0, sum(x)<=1 to y=[x,reserve]>=0, sum(y)=1. Only x is
penalized in ||x-proposal||^2; the reserve is an unpenalized auxiliary.
The small implicit adjoint system is solved by a dense Moore--Penrose
inverse instead of the reference implementation's iterative BiCGSTAB.
All solver operations stay on the input device. No analytic simplex
projector, straight-through gradient, or unrolled backward is used.
"""
from __future__ import annotations

import torch
from torch import nn


DEFAULT_SETTINGS = {
    "sigma": 1.0, "omega": 1.7, "iterations": 64,
    "max_iterations": 2048, "fixed_point_tolerance": 1e-9,
    "residual_tolerance": 1e-5,
}


@torch.jit.script
def _dr_steps(s: torch.Tensor, target: torch.Tensor, scale: torch.Tensor,
              steps: int, omega: float):
    for _ in range(steps):
        z = s + (1.0 - s.sum(-1, keepdim=True)) / s.size(-1)
        t = ((2.0 * z - s + target) * scale).clamp_min(0.0)
        s = s + omega * (t - z)
    return s


class _BudgetProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proposal, sigma, omega, iterations, max_iterations, tolerance):
        if not torch.isfinite(proposal).all():
            raise FloatingPointError("PiNet received a nonfinite proposal.")
        m = proposal.shape[-1]
        # Solver precision is local, as in the preceding CvxpyLayer adapter.
        u = proposal.to(torch.float64)
        reserve = torch.zeros_like(u[..., :1])
        target = torch.cat((2.0 * sigma * u, reserve), dim=-1)
        scale = torch.ones(m + 1, dtype=u.dtype, device=u.device)
        scale[:m] = 1.0 / (1.0 + 2.0 * sigma)
        s = torch.cat((u, reserve), dim=-1)
        used = 0
        block = iterations
        while used < max_iterations:
            block = min(block, max_iterations - used)
            s = _dr_steps(s, target, scale, block, omega)
            used += block
            z = s + (1.0 - s.sum(-1, keepdim=True)) / (m + 1)
            prox_argument = (2.0 * z - s + target) * scale
            residual = (prox_argument.clamp_min(0.0) - z).abs().amax(-1, keepdim=True)
            if float(residual.max()) <= tolerance:
                break
            block *= 2
        if not torch.isfinite(residual).all() or float(residual.max()) > tolerance:
            raise FloatingPointError(
                f"PiNet did not converge: residual={float(residual.max()):.3g}, iterations={used}.")
        ctx.save_for_backward(prox_argument > 0, scale)
        ctx.sigma, ctx.omega, ctx.input_dtype = sigma, omega, proposal.dtype
        count = torch.full_like(residual, used)
        ctx.mark_non_differentiable(residual, count)
        return z[..., :m].to(proposal.dtype), residual, count

    @staticmethod
    def backward(ctx, grad_output, grad_residual, grad_count):
        active, scale = ctx.saved_tensors
        n = active.shape[-1]
        eye = torch.eye(n, dtype=scale.dtype, device=scale.device)
        p = eye - torch.ones_like(eye) / n  # derivative of projection onto sum(y)=1
        d = active.to(scale.dtype) * scale
        # I - dPhi/ds = omega * (P - D(2P-I)).
        lhs = ctx.omega * (p - d.unsqueeze(-1) * (2.0 * p - eye))
        padded = torch.cat((grad_output.to(scale.dtype),
                            torch.zeros_like(grad_output[..., :1], dtype=scale.dtype)), dim=-1)
        rhs = (padded @ p).unsqueeze(-1)
        adjoint = torch.linalg.pinv(lhs.transpose(-1, -2), rtol=1e-12) @ rhs
        error = (lhs.transpose(-1, -2) @ adjoint - rhs).abs().amax()
        if not torch.isfinite(adjoint).all() or float(error) > 1e-7 * (1 + float(rhs.abs().max())):
            raise FloatingPointError("PiNet implicit adjoint residual is too large.")
        grad = (ctx.omega * 2.0 * ctx.sigma / (1.0 + 2.0 * ctx.sigma)
                * active[..., :-1] * adjoint.squeeze(-1)[..., :-1])
        return grad.to(ctx.input_dtype), None, None, None, None, None


class PiNetBudgetLayer(nn.Module):
    def __init__(self, settings=None):
        super().__init__()
        self.settings = dict(DEFAULT_SETTINGS if settings is None else settings)
        s = self.settings
        if not (s["sigma"] > 0 and 0 < s["omega"] < 2
                and 0 < s["iterations"] <= s["max_iterations"]
                and s["fixed_point_tolerance"] > 0 and s["residual_tolerance"] > 0):
            raise ValueError("Invalid PiNet solver settings.")
        self.last_diagnostics = {}

    def forward(self, proposal):
        s = self.settings
        solved, fixed_point_residual, iterations = _BudgetProjection.apply(
            proposal, s["sigma"], s["omega"], s["iterations"],
            s["max_iterations"], s["fixed_point_tolerance"])
        residual = torch.maximum((-solved).clamp_min(0).amax(-1, keepdim=True),
                                 (solved.sum(-1, keepdim=True) - 1).clamp_min(0))
        if not torch.isfinite(solved).all() or float(residual.max()) > s["residual_tolerance"]:
            raise FloatingPointError("PiNet output exceeds its declared feasibility tolerance.")
        # Repair only bounded floating-point residuals, and disclose the repair.
        positive = solved.clamp_min(0)
        feasible = positive / positive.sum(-1, keepdim=True).clamp_min(1)
        self.last_diagnostics = {
            "solver_constraint_residual": residual.detach(),
            "solver_numerical_repair_fraction": (feasible - solved).norm(dim=-1, keepdim=True).detach(),
            "pinet_fixed_point_residual": fixed_point_residual.detach(),
            "pinet_iterations": iterations.detach(),
        }
        return feasible
