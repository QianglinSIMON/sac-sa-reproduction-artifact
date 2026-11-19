"""
sacsa_rl_components.py
Shared SAC-SA / TD3 components for dynamic budgeted subsidy allocation.

Compared methods:
    - No Subsidy
    - Uniform Pacing
    - Demand-Supply Gap Heuristic
    - Proposed Beta-Dirichlet SAC-SA
    - Reserve-Gated Logistic-Normal SAC
    - Original Augmented-Dirichlet SAC-SA
    - Original Logistic-Normal SAC
    - TD3 + Feasible Shares

Main stability/evaluation changes in this version:
    1. SAC entropy regularization uses clipped share-space log-density by default
       in critic target, actor loss, and temperature tuning.
    2. For fairness, the same SAC entropy convention is applied to both
       Proposed SAC-SA and Logistic-Normal SAC.
    3. Q-functions are defined over executed monetary actions A, while the entropy
       regularizer is evaluated in share space for numerical stability.
    4. Dirichlet concentration logits are clamped and sanitized to avoid NaN/Inf.
    5. Optional alpha/reward/target safety guards are added for long CPU runs.
    6. The maintained experiment entry point is iterative_synthetic_sacsa.py.

Core conventions:
    1. One episode is one independent sample trajectory with T decision intervals.
    2. State normalization standardizes only [X, Y, C, O]. Remaining budget H stays raw.
    3. Environment projection is disabled because policies/heuristics construct feasible actions.
    4. A lightweight tensor feasibility guard is applied before env.step().
    5. Terminal next states are masked in Bellman targets.
    6. Per-method seeds are reset before each method for fair initialization/RNG control.
    7. SAC-SA, Logistic-Normal SAC, and TD3 share the same actor learning rate
       for the main fair comparison. Method-specific tuning should be reported as ablation.
       The alpha learning rate applies only to SAC-based methods.

Original augmented-Dirichlet SAC-SA policy:
    P ~ Dirichlet(alpha(s)) over M+1 shares.
    A_m = H_t * P_m, m=1,...,M. P_{M+1} is unspent budget.
    share-space log pi(P|S) = log Dir(P; alpha(S)).
    action-space log pi(A|S) = log Dir(P; alpha(S)) - M log H_t.

Reserve-gate SAC-SA policy:
    G ~ Beta(a(s), b(s)) determines the total spend fraction.
    P ~ Dirichlet(alpha(s)) or logistic-normal determines city allocation.
    A = H_t * G * P. Entropy regularization uses only the city-share log
    density, so the reserve gate learns through the Q gradient.

Important SAC convention in this stable version:
    - Q-functions are defined and evaluated on executed monetary actions A.
    - The entropy regularizer is evaluated using clipped share-space log-density
      for both SAC-SA and Logistic-Normal SAC.
    - This is a stabilized SAC objective: it removes the budget-dependent
      -M log H_t Jacobian term from the entropy bonus while keeping Q(A) over
      actual executed subsidies.
    - Temperature is tuned within user-specified bounds. In the main setting,
      max_temperature equals init_temperature, preventing excessive entropy
      pressure during long runs.
The current learned-simulator paper experiment uses this file as a component
library. The maintained experiment entry point is iterative_synthetic_sacsa.py.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
from budget_gate import positive_gate, inverse_positive_gate
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Dirichlet, Normal

BUDGET_SAFETY = 0.999
FEASIBILITY_SAFETY = 1.0 - 1e-7
EPS = 1e-8
SANITIZE_ABS_MAX = 1e4
STATE_SANITIZE_ABS_MAX = 1e6


# =========================================================
# 0. Utilities
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def finite_or_zero(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def sanitize_tensor(
    x: torch.Tensor,
    nan: float = 0.0,
    abs_max: float = SANITIZE_ABS_MAX,
) -> torch.Tensor:
    """Numerical guard for training scalars/targets/log densities."""
    return torch.nan_to_num(x, nan=nan, posinf=abs_max, neginf=-abs_max).clamp(-abs_max, abs_max)


def sanitize_state_tensor(x: torch.Tensor) -> torch.Tensor:
    """State values can be naturally larger than log-density/target values."""
    return torch.nan_to_num(
        x,
        nan=0.0,
        posinf=STATE_SANITIZE_ABS_MAX,
        neginf=-STATE_SANITIZE_ABS_MAX,
    ).clamp(-STATE_SANITIZE_ABS_MAX, STATE_SANITIZE_ABS_MAX)


def infer_remaining_budget(state: torch.Tensor, M: int) -> torch.Tensor:
    """State format: [X(M), Y(M), C(M), O(M), H(1), time_features(3)]."""
    return state[..., 4 * M]


def ensure_feasible_action(
    action: torch.Tensor,
    remaining_budget: torch.Tensor,
    safety: float = FEASIBILITY_SAFETY,
) -> torch.Tensor:
    """
    Feasibility guard.
    Supports both one action with shape [M] and batched actions with shape [B, M].
    """
    if not torch.isfinite(action).all():
        raise ValueError("Action contains nonfinite entries before feasibility correction.")
    h = remaining_budget.to(device=action.device, dtype=action.dtype)
    if not torch.isfinite(h).all():
        raise ValueError("Remaining budget contains nonfinite entries.")
    action = action.clamp_min(0.0)
    h = h.clamp_min(0.0)

    if action.ndim == 1:
        total = action.sum()
        max_total = h * safety
        scale = torch.minimum(torch.ones_like(total), max_total / total.clamp_min(EPS))
        return action * scale

    total = action.sum(dim=-1, keepdim=True)
    while h.ndim < action.ndim:
        h = h.unsqueeze(-1)
    max_total = h * safety
    scale = torch.minimum(torch.ones_like(total), max_total / total.clamp_min(EPS))
    return action * scale


def mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def get_last_linear(module: nn.Module) -> Optional[nn.Linear]:
    modules = list(module) if isinstance(module, nn.Sequential) else list(module.modules())
    for m in reversed(modules):
        if isinstance(m, nn.Linear):
            return m
    return None


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, min=EPS)
    return y + torch.log(-torch.expm1(-y))


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, pt in zip(source.parameters(), target.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)


@torch.no_grad()
def random_paced_action(
    env: Any,
    action_dim: int,
    safety: float = BUDGET_SAFETY,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    weights = torch.rand(action_dim, device=env.device, dtype=env.dtype, generator=generator)
    weights = weights / weights.sum().clamp_min(EPS)
    planned_spend = env.H / max(env.T - env.t, 1)
    return weights * planned_spend * safety


def has_nonfinite(module: nn.Module) -> bool:
    """
    Checks model parameters. Optimizer-state non-finiteness is rare and usually
    manifests in parameters on a subsequent update.
    """
    for p in module.parameters():
        if not torch.isfinite(p).all():
            return True
    return False


# =========================================================
# 1. State normalization and environment wrapper
# =========================================================
@dataclass
class StateNormalizer:
    """Standardize only [X, Y, C, O]. H and time features are left unnormalized."""

    M: int
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    @property
    def norm_dim(self) -> int:
        return 4 * self.M

    def to(self, device: str) -> "StateNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, state: torch.Tensor) -> torch.Tensor:
        out = state.clone()
        out[..., : self.norm_dim] = (out[..., : self.norm_dim] - self.mean) / self.std.clamp_min(self.eps)
        return sanitize_state_tensor(out)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "M": self.M,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "eps": self.eps,
        }


# =========================================================
# 2. Config
# =========================================================
@dataclass
class TrainConfig:
    seed: int = 2026
    device: str = "cpu"
    M: int = 8
    T: int = 24
    daily_budget: float = 10.0
    algo: str = "all"
    train_episodes: int = 4000
    warmup_episodes: int = 200
    update_after_episodes: int = 200
    eval_every_episodes: int = 100
    eval_episodes: int = 100
    log_every_episodes: int = 10
    verbose_episode_log: bool = False
    update_every_steps: int = 1
    gradient_steps_per_update: int = 2
    batch_size: int = 512
    replay_size: int = 500_000
    use_state_norm: bool = True
    hidden_dim: int = 256

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4

    gamma: float = 1.0
    tau: float = 0.005
    grad_clip_norm: float = 1.0
    critic_loss: str = "huber"
    huber_delta: float = 1.0

    auto_alpha: bool = True
    fixed_temperature: Optional[float] = None
    init_temperature: float = 0.2
    target_entropy_scale: float = 2.5
    alpha_logp_clip: float = 20.0
    min_temperature: float = 1e-4
    max_temperature: float = 1.0

    sac_entropy_space: str = "share_clipped"
    bd_gate_parameterization: str = "softplus"
    dirichlet_min_alpha: float = 1e-3
    dirichlet_max_alpha: float = 1e4
    dirichlet_logit_clip: float = 20.0
    stop_on_nonfinite: bool = True

    dirichlet_init_total_spend_share: Optional[float] = None
    dirichlet_init_concentration: float = 20.0
    ln_init_total_spend_share: Optional[float] = None
    ln_init_log_std: float = -2.0

    policy_delay: int = 2
    exploration_noise: float = 0.05
    target_noise: float = 0.05
    noise_clip: float = 0.10
    td3_init_total_spend_share: Optional[float] = None

    out_dir: str = "results_subsidy"
    save_models: bool = False
    make_plots: bool = True

    # New final evaluation controls.
    save_final_eval_raw: bool = True
    make_final_boxplots: bool = True
    final_eval_seed_offset: int = 20_000
    boxplot_showfliers: bool = False
    e1_options: Optional[Dict[str, Any]] = None

    @property
    def train_steps(self) -> int:
        return self.train_episodes * self.T

    @property
    def warmup_steps(self) -> int:
        return self.warmup_episodes * self.T

    @property
    def update_after_steps(self) -> int:
        return self.update_after_episodes * self.T

    @property
    def default_init_total_spend_share(self) -> float:
        return BUDGET_SAFETY / max(self.T, 1)


# =========================================================
# 3. Replay buffer
# =========================================================
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, size: int, device: str):
        self.device = device
        self.max_size = int(size)
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size, 1), dtype=torch.float32, device=device)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | float,
        next_state: Optional[torch.Tensor],
        done: bool,
    ) -> None:
        self.states[self.ptr] = sanitize_state_tensor(state.detach().to(self.device))
        self.actions[self.ptr] = torch.nan_to_num(action.detach().to(self.device), nan=0.0, posinf=0.0, neginf=0.0)
        self.rewards[self.ptr] = finite_or_zero(to_float(reward))
        if next_state is None:
            self.next_states[self.ptr].zero_()
        else:
            self.next_states[self.ptr] = sanitize_state_tensor(next_state.detach().to(self.device))
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: Optional[torch.Tensor],
        dones: torch.Tensor,
    ) -> None:
        states = sanitize_state_tensor(states.detach().to(self.device))
        actions = torch.nan_to_num(
            actions.detach().to(self.device),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        rewards = torch.nan_to_num(
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device).reshape(-1, 1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if next_states is None:
            next_states_t = torch.zeros_like(states)
        else:
            next_states_t = sanitize_state_tensor(next_states.detach().to(self.device))
        dones_t = torch.nan_to_num(
            torch.as_tensor(dones, dtype=torch.float32, device=self.device).reshape(-1, 1),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        ).clamp(0.0, 1.0)

        n = int(states.shape[0])
        if n <= 0:
            return
        if any(int(x.shape[0]) != n for x in (actions, rewards, next_states_t, dones_t)):
            raise ValueError("ReplayBuffer.add_batch inputs must share the same leading dimension.")
        if n > self.max_size:
            states = states[-self.max_size :]
            actions = actions[-self.max_size :]
            rewards = rewards[-self.max_size :]
            next_states_t = next_states_t[-self.max_size :]
            dones_t = dones_t[-self.max_size :]
            n = self.max_size

        idx = (torch.arange(n, device=self.device) + self.ptr) % self.max_size
        self.states[idx] = states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_states[idx] = next_states_t
        self.dones[idx] = dones_t
        self.ptr = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
        }


# =========================================================
# 4. Networks and actors
# =========================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(state_dim + action_dim, (hidden_dim, hidden_dim), 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = sanitize_state_tensor(x)
        return self.net(x)


class DirichletFeasibleActor(nn.Module):
    """SAC-SA actor with exact share-space and action-space log densities."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_alpha: float = 1e-3,
        max_alpha: float = 1e4,
        logit_clip: float = 20.0,
        min_budget: float = EPS,
        init_total_spend_share: Optional[float] = None,
        init_concentration: float = 20.0,
    ):
        super().__init__()
        self.M = action_dim
        self.entropy_dim = action_dim
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.logit_clip = logit_clip
        self.min_budget = min_budget
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), action_dim + 1)
        if init_total_spend_share is not None:
            self._init_pacing_alpha(init_total_spend_share, init_concentration)

    def _init_pacing_alpha(self, total_spend_share: float, concentration: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        concentration = max(float(concentration), (self.M + 1) * (self.min_alpha + 1e-4))
        target_alpha = torch.full((self.M + 1,), total_spend_share / self.M * concentration)
        target_alpha[-1] = (1.0 - total_spend_share) * concentration
        target_alpha = target_alpha.clamp(self.min_alpha + 1e-5, self.max_alpha)
        target_raw = inverse_softplus((target_alpha - self.min_alpha).clamp_min(1e-5))
        target_raw = target_raw.clamp(-self.logit_clip, self.logit_clip)
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.copy_(target_raw.to(last.bias.device, dtype=last.bias.dtype))

    def alpha(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        raw = raw.clamp(-self.logit_clip, self.logit_clip)
        alpha = F.softplus(raw) + self.min_alpha
        alpha = torch.nan_to_num(alpha, nan=1.0, posinf=self.max_alpha, neginf=self.min_alpha)
        return alpha.clamp(self.min_alpha, self.max_alpha)

    def distribution(self, state: torch.Tensor) -> Dirichlet:
        return Dirichlet(self.alpha(state), validate_args=False)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        return_share: bool = False,
    ):
        dist = self.distribution(state)
        alpha = dist.concentration
        if deterministic:
            share_raw = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(EPS)
        else:
            share_raw = dist.rsample()

        share = torch.nan_to_num(share_raw, nan=1.0 / (self.M + 1), posinf=1.0, neginf=0.0)
        share = share.clamp_min(EPS)
        share = share / share.sum(dim=-1, keepdim=True).clamp_min(EPS)

        h_raw = infer_remaining_budget(state, self.M).unsqueeze(-1)
        h_action = h_raw.clamp_min(0.0)
        h_log = h_raw.clamp_min(self.min_budget)
        action = h_action * share[..., : self.M]

        share_logp = dist.log_prob(share).unsqueeze(-1)
        share_logp = sanitize_tensor(share_logp)
        action_logp = share_logp - self.M * torch.log(h_log)
        action_logp = sanitize_tensor(action_logp)

        if return_share:
            return action, action_logp, share_logp, share
        return action, action_logp, share_logp


class LogisticNormalFeasibleActor(nn.Module):
    """Strict feasible SAC baseline with logistic-normal transformed density."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        min_budget: float = EPS,
        init_total_spend_share: Optional[float] = None,
        init_log_std: float = -2.0,
    ):
        super().__init__()
        self.M = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_budget = min_budget
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), 2 * action_dim)
        self._init_log_std(init_log_std)
        if init_total_spend_share is not None:
            self._init_pacing_mean(init_total_spend_share)

    def _init_log_std(self, init_log_std: float) -> None:
        last = get_last_linear(self.net)
        if last is None:
            return
        with torch.no_grad():
            last.bias[self.M:].fill_(float(init_log_std))

    def _init_pacing_mean(self, total_spend_share: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        city_share = total_spend_share / self.M
        save_share = 1.0 - total_spend_share
        z_mean = math.log(city_share / save_share)
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias[: self.M].fill_(z_mean)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.nan_to_num(self.net(state), nan=0.0, posinf=20.0, neginf=-20.0)
        mean, log_std = out.chunk(2, dim=-1)
        mean = mean.clamp(-20.0, 20.0)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _action_logp_from_z(self, state: torch.Tensor, z: torch.Tensor, dist: Normal):
        zeros = torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
        logits = torch.cat([z, zeros], dim=-1)
        log_share = F.log_softmax(logits, dim=-1)
        share = log_share.exp().clamp_min(EPS)
        share = share / share.sum(dim=-1, keepdim=True).clamp_min(EPS)

        h_raw = infer_remaining_budget(state, self.M).unsqueeze(-1)
        h_action = h_raw.clamp_min(0.0)
        h_log = h_raw.clamp_min(self.min_budget)
        action = h_action * share[..., : self.M]

        logp_z = dist.log_prob(z).sum(dim=-1, keepdim=True)
        share_logp = logp_z - log_share.sum(dim=-1, keepdim=True)
        action_logp = share_logp - self.M * torch.log(h_log)
        share_logp = sanitize_tensor(share_logp)
        action_logp = sanitize_tensor(action_logp)
        return action, action_logp, share_logp, share

    def sample(self, state: torch.Tensor, deterministic: bool = False, return_share: bool = False):
        mean, log_std = self.forward(state)
        dist = Normal(mean, log_std.exp())
        z = mean if deterministic else dist.rsample()
        z = torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        action, action_logp, share_logp, share = self._action_logp_from_z(state, z, dist)
        if return_share:
            return action, action_logp, share_logp, share
        return action, action_logp, share_logp


class FactorizedBetaDirichletFeasibleActor(nn.Module):
    """
    Reserve-aware SAC-SA actor.

    A Beta gate chooses the total spend fraction G, and a Dirichlet
    distribution chooses city shares P.  SAC entropy regularization uses only
    the city-share log density returned as the third sample output; the gate is
    learned through Q gradients.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        min_alpha: float = 1e-3,
        max_alpha: float = 1e4,
        logit_clip: float = 20.0,
        min_budget: float = EPS,
        init_total_spend_share: Optional[float] = None,
        init_concentration: float = 20.0,
        gate_parameterization: str = "softplus",
    ):
        super().__init__()
        self.M = action_dim
        self.gate_parameterization = gate_parameterization
        self.entropy_dim = max(action_dim - 1, 1)
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.logit_clip = logit_clip
        self.min_budget = min_budget
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), 2 + action_dim)
        self.entropy_logp_scope = "city_share"
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_gate_logp: Optional[torch.Tensor] = None
        self.last_gate_sample: Optional[torch.Tensor] = None
        self.last_city_share_logp: Optional[torch.Tensor] = None
        init_share = BUDGET_SAFETY / 24.0 if init_total_spend_share is None else init_total_spend_share
        self._init_factorized_params(init_share, init_concentration)

    def _init_factorized_params(self, total_spend_share: float, concentration: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        concentration = max(float(concentration), 2.0 + self.M * (self.min_alpha + 1e-4))
        gate_conc = max(float(concentration), 4.0)
        gate_alpha = max(total_spend_share * gate_conc, self.min_alpha + 1e-5)
        gate_beta = max((1.0 - total_spend_share) * gate_conc, self.min_alpha + 1e-5)
        city_alpha = max(concentration / max(self.M, 1), self.min_alpha + 1e-5)

        beta_target = torch.tensor([gate_alpha, gate_beta], dtype=torch.float32)
        city_target = torch.full((self.M,), city_alpha, dtype=torch.float32)
        target = torch.cat([beta_target, city_target], dim=0).clamp(self.min_alpha + 1e-5, self.max_alpha)
        target_raw = inverse_softplus((target - self.min_alpha).clamp_min(1e-5)).clamp(-self.logit_clip, self.logit_clip)
        gate_slice = slice(self.M, self.M + 2) if hasattr(self, 'gate_alpha_beta') else slice(0, 2)
        target_raw[gate_slice] = inverse_positive_gate(
            target[gate_slice], self.min_alpha, self.logit_clip, self.gate_parameterization)
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.copy_(target_raw.to(last.bias.device, dtype=last.bias.dtype))

    def _params(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(state)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        gate_raw = raw[..., :2]
        raw = raw.clamp(-self.logit_clip, self.logit_clip)
        params = F.softplus(raw) + self.min_alpha
        params = torch.nan_to_num(params, nan=1.0, posinf=self.max_alpha, neginf=self.min_alpha)
        params = params.clamp(self.min_alpha, self.max_alpha)
        beta_params = positive_gate(gate_raw, self.min_alpha, self.max_alpha,
                                    self.logit_clip, self.gate_parameterization)
        city_alpha = params[..., 2:]
        return beta_params, city_alpha

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        return_share: bool = False,
    ):
        beta_params, city_alpha = self._params(state)
        gate_dist = Beta(beta_params[..., 0], beta_params[..., 1], validate_args=False)
        city_dist = Dirichlet(city_alpha, validate_args=False)

        gate_mean = beta_params[..., 0:1] / beta_params.sum(dim=-1, keepdim=True).clamp_min(EPS)
        gate_raw = gate_mean if deterministic else gate_dist.rsample().unsqueeze(-1)
        gate = torch.nan_to_num(gate_raw, nan=0.0, posinf=1.0, neginf=0.0).clamp(EPS, 1.0 - EPS)

        if deterministic:
            city_raw = city_alpha / city_alpha.sum(dim=-1, keepdim=True).clamp_min(EPS)
        else:
            city_raw = city_dist.rsample()
        city_share = torch.nan_to_num(city_raw, nan=1.0 / self.M, posinf=1.0, neginf=0.0).clamp_min(EPS)
        city_share = city_share / city_share.sum(dim=-1, keepdim=True).clamp_min(EPS)

        h_raw = infer_remaining_budget(state, self.M).unsqueeze(-1)
        h_action = h_raw.clamp_min(0.0)
        h_log = h_raw.clamp_min(self.min_budget)
        action = h_action * gate * city_share

        gate_logp = gate_dist.log_prob(gate.squeeze(-1)).unsqueeze(-1)
        city_share_logp = city_dist.log_prob(city_share).unsqueeze(-1)
        gate_log = torch.log(gate.clamp_min(EPS))
        action_logp = gate_logp + city_share_logp - self.M * torch.log(h_log) - (self.M - 1) * gate_log
        gate_logp = sanitize_tensor(gate_logp)
        city_share_logp = sanitize_tensor(city_share_logp)
        action_logp = sanitize_tensor(action_logp)

        self.last_gate_mean = gate_mean.detach()
        self.last_gate_low_fraction = (gate_mean.detach() < 1e-3).float().mean()
        self.last_gate_high_fraction = (gate_mean.detach() > 1 - 1e-3).float().mean()
        self.last_gate_sample = gate.detach()
        self.last_gate_logp = gate_logp.detach()
        self.last_city_share_logp = city_share_logp.detach()

        if return_share:
            diagnostic_share = torch.cat([gate * city_share, 1.0 - gate], dim=-1)
            return action, action_logp, city_share_logp, diagnostic_share
        return action, action_logp, city_share_logp


class FactorizedBetaLogisticNormalFeasibleActor(nn.Module):
    """
    Reserve-gate comparator with a Beta total-spend gate and logistic-normal
    city allocation.  Entropy regularization again uses only city-share logp.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        min_budget: float = EPS,
        min_beta_param: float = 1e-3,
        logit_clip: float = 20.0,
        init_total_spend_share: Optional[float] = None,
        init_concentration: float = 20.0,
        init_log_std: float = -2.0,
    ):
        super().__init__()
        self.M = action_dim
        self.city_latent_dim = max(action_dim - 1, 0)
        self.entropy_dim = max(action_dim - 1, 1)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_budget = min_budget
        self.min_beta_param = min_beta_param
        self.logit_clip = logit_clip
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), 2 + 2 * self.city_latent_dim)
        self.entropy_logp_scope = "city_share"
        self.last_gate_mean: Optional[torch.Tensor] = None
        self.last_gate_logp: Optional[torch.Tensor] = None
        self.last_gate_sample: Optional[torch.Tensor] = None
        self.last_city_share_logp: Optional[torch.Tensor] = None
        init_share = BUDGET_SAFETY / 24.0 if init_total_spend_share is None else init_total_spend_share
        self._init_factorized_params(init_share, init_concentration, init_log_std)

    def _init_factorized_params(self, total_spend_share: float, concentration: float, init_log_std: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        gate_conc = max(float(concentration), 4.0)
        gate_alpha = max(total_spend_share * gate_conc, self.min_beta_param + 1e-5)
        gate_beta = max((1.0 - total_spend_share) * gate_conc, self.min_beta_param + 1e-5)
        beta_target = torch.tensor([gate_alpha, gate_beta], dtype=torch.float32)
        beta_raw = inverse_softplus((beta_target - self.min_beta_param).clamp_min(1e-5)).clamp(-self.logit_clip, self.logit_clip)
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()
                last.bias[:2].copy_(beta_raw.to(last.bias.device, dtype=last.bias.dtype))
                if self.city_latent_dim > 0:
                    last.bias[2 + self.city_latent_dim :].fill_(float(init_log_std))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = torch.nan_to_num(self.net(state), nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        out = out.clamp(-self.logit_clip, self.logit_clip)
        beta_raw = out[..., :2]
        beta_params = F.softplus(beta_raw) + self.min_beta_param
        beta_params = torch.nan_to_num(beta_params, nan=1.0, posinf=1e4, neginf=self.min_beta_param)
        beta_params = beta_params.clamp(self.min_beta_param, 1e4)
        if self.city_latent_dim <= 0:
            empty = out[..., 2:2]
            return beta_params, empty, empty
        mean = out[..., 2 : 2 + self.city_latent_dim].clamp(-20.0, 20.0)
        log_std = out[..., 2 + self.city_latent_dim :].clamp(self.log_std_min, self.log_std_max)
        return beta_params, mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False, return_share: bool = False):
        beta_params, mean, log_std = self.forward(state)
        gate_dist = Beta(beta_params[..., 0], beta_params[..., 1], validate_args=False)
        gate_mean = beta_params[..., 0:1] / beta_params.sum(dim=-1, keepdim=True).clamp_min(EPS)
        gate_raw = gate_mean if deterministic else gate_dist.rsample().unsqueeze(-1)
        gate = torch.nan_to_num(gate_raw, nan=0.0, posinf=1.0, neginf=0.0).clamp(EPS, 1.0 - EPS)

        if self.city_latent_dim <= 0:
            city_share = torch.ones((*state.shape[:-1], 1), device=state.device, dtype=state.dtype)
            city_share_logp = torch.zeros((*state.shape[:-1], 1), device=state.device, dtype=state.dtype)
        else:
            dist = Normal(mean, log_std.exp())
            z = mean if deterministic else dist.rsample()
            z = torch.nan_to_num(z, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            zeros = torch.zeros((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
            logits = torch.cat([z, zeros], dim=-1)
            log_city_share = F.log_softmax(logits, dim=-1)
            city_share = log_city_share.exp().clamp_min(EPS)
            city_share = city_share / city_share.sum(dim=-1, keepdim=True).clamp_min(EPS)
            logp_z = dist.log_prob(z).sum(dim=-1, keepdim=True)
            city_share_logp = logp_z - log_city_share.sum(dim=-1, keepdim=True)

        h_raw = infer_remaining_budget(state, self.M).unsqueeze(-1)
        h_action = h_raw.clamp_min(0.0)
        h_log = h_raw.clamp_min(self.min_budget)
        action = h_action * gate * city_share

        gate_logp = gate_dist.log_prob(gate.squeeze(-1)).unsqueeze(-1)
        gate_log = torch.log(gate.clamp_min(EPS))
        action_logp = gate_logp + city_share_logp - self.M * torch.log(h_log) - (self.M - 1) * gate_log
        gate_logp = sanitize_tensor(gate_logp)
        city_share_logp = sanitize_tensor(city_share_logp)
        action_logp = sanitize_tensor(action_logp)

        self.last_gate_mean = gate_mean.detach()
        self.last_gate_sample = gate.detach()
        self.last_gate_logp = gate_logp.detach()
        self.last_city_share_logp = city_share_logp.detach()

        if return_share:
            diagnostic_share = torch.cat([gate * city_share, 1.0 - gate], dim=-1)
            return action, action_logp, city_share_logp, diagnostic_share
        return action, action_logp, city_share_logp


class DirectSoftmaxTD3Actor(nn.Module):
    """TD3 actor with an M+1 softmax over city spending and unspent budget."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, init_total_spend_share: Optional[float] = None):
        super().__init__()
        self.M = action_dim
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), action_dim + 1)
        if init_total_spend_share is not None:
            self._init_pacing_logits(init_total_spend_share)

    def _init_pacing_logits(self, total_spend_share: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        save_bias = math.log(max(self.M * (1.0 - total_spend_share) / total_spend_share, EPS))
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()
                last.bias[-1] = save_bias

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.net(state), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

    def action_from_logits(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        h = infer_remaining_budget(state, self.M).unsqueeze(-1).clamp_min(0.0)
        share = F.softmax(logits, dim=-1)
        return h * share[..., : self.M]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_from_logits(state, self.logits(state))

    def sample_with_action_noise(
        self,
        state: torch.Tensor,
        noise_std: float,
        noise_clip: Optional[float],
    ) -> torch.Tensor:
        logits = self.logits(state)
        if noise_std > 0:
            noise = torch.randn_like(logits) * noise_std
            if noise_clip is not None:
                noise = noise.clamp(-noise_clip, noise_clip)
            logits = (logits + noise).clamp(-20.0, 20.0)
        return self.action_from_logits(state, logits)


def project_nonnegative_l1_ball(
    proposal: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Euclidean projection onto {x >= 0, sum(x) <= radius}, batchwise."""
    radius = radius.clamp_min(0.0)
    positive = proposal.clamp_min(0.0)
    inside = positive.sum(dim=-1, keepdim=True) <= radius

    sorted_values = torch.sort(proposal, dim=-1, descending=True).values
    indices = torch.arange(
        1,
        proposal.shape[-1] + 1,
        device=proposal.device,
        dtype=proposal.dtype,
    )
    shifted_cumsum = sorted_values.cumsum(dim=-1) - radius
    active = sorted_values - shifted_cumsum / indices > 0.0
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    threshold = shifted_cumsum.gather(dim=-1, index=rho)
    threshold = threshold / (rho.to(proposal.dtype) + 1.0)
    projected = (proposal - threshold).clamp_min(0.0)
    return torch.where(inside, positive, projected)


class ProjectedTD3Actor(nn.Module):
    """TD3 actor with an M-vector proposal and post-hoc Euclidean projection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        init_total_spend_share: Optional[float] = None,
    ):
        super().__init__()
        self.M = action_dim
        self.net = mlp(state_dim, (hidden_dim, hidden_dim), action_dim)
        if init_total_spend_share is not None:
            self._init_pacing_proposal(init_total_spend_share)

    def _init_pacing_proposal(self, total_spend_share: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        per_city_share = float(np.clip(total_spend_share / self.M, 1e-4, 1.0 - 1e-4))
        last = get_last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.fill_(per_city_share)

    def proposal(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            self.net(state),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(-20.0, 20.0)

    def action_from_fraction(
        self,
        state: torch.Tensor,
        fraction: torch.Tensor,
    ) -> torch.Tensor:
        h = infer_remaining_budget(state, self.M).unsqueeze(-1).clamp_min(0.0)
        feasible_fraction = project_nonnegative_l1_ball(
            fraction,
            torch.ones_like(h),
        )
        return h * feasible_fraction

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_from_fraction(state, self.proposal(state))

    def sample_with_action_noise(
        self,
        state: torch.Tensor,
        noise_std: float,
        noise_clip: Optional[float],
    ) -> torch.Tensor:
        fraction = self.proposal(state)
        if noise_std > 0:
            noise = torch.randn_like(fraction) * noise_std
            if noise_clip is not None:
                noise = noise.clamp(-noise_clip, noise_clip)
            fraction = fraction + noise
        return self.action_from_fraction(state, fraction)


# =========================================================
# 5. Agents
# =========================================================
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: TrainConfig, actor: nn.Module):
        self.cfg = cfg
        self.device = cfg.device
        self.action_dim = int(action_dim)
        self.actor = actor.to(cfg.device)
        self.q1 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.q2 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.q1_targ = copy.deepcopy(self.q1).to(cfg.device)
        self.q2_targ = copy.deepcopy(self.q2).to(cfg.device)

        self.actor_lr = cfg.lr_actor
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr_critic)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr_critic)

        init_temperature = cfg.init_temperature if cfg.fixed_temperature is None else float(cfg.fixed_temperature)
        self.log_alpha = torch.tensor(
            math.log(init_temperature),
            dtype=torch.float32,
            device=cfg.device,
            requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        entropy_dim = float(getattr(actor, "entropy_dim", action_dim))
        self.target_entropy = -cfg.target_entropy_scale * entropy_dim

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(self.cfg.min_temperature, self.cfg.max_temperature)

    def _critic_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.critic_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.cfg.huber_delta)
        return F.mse_loss(pred, target)

    def _alpha_logp(self, share_logp: torch.Tensor) -> torch.Tensor:
        clip = float(self.cfg.alpha_logp_clip)
        return share_logp.detach().clamp(min=-clip, max=clip)

    def _entropy_logp(self, action_logp: torch.Tensor, share_logp: torch.Tensor) -> torch.Tensor:
        if getattr(self.actor, "entropy_logp_scope", "") == "city_share":
            if self.cfg.sac_entropy_space == "share_unclipped":
                return sanitize_tensor(share_logp)
            return share_logp.clamp(min=-self.cfg.alpha_logp_clip, max=self.cfg.alpha_logp_clip)
        if self.cfg.sac_entropy_space == "action":
            return sanitize_tensor(action_logp)
        if self.cfg.sac_entropy_space == "share_unclipped":
            return sanitize_tensor(share_logp)
        if self.cfg.sac_entropy_space == "share_clipped":
            return share_logp.clamp(min=-self.cfg.alpha_logp_clip, max=self.cfg.alpha_logp_clip)
        raise ValueError(f"Unknown sac_entropy_space: {self.cfg.sac_entropy_space}")

    def _active_budget_mask(self, state: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Mask entropy terms when the feasible action set is degenerate at A=0."""
        mask = infer_remaining_budget(state, self.action_dim).unsqueeze(-1) > EPS
        return mask.to(device=like.device, dtype=like.dtype)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _, _ = self.actor.sample(state.unsqueeze(0), deterministic=deterministic)
            return action.squeeze(0)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        sp = batch["next_state"]
        d = batch["done"]
        alpha = self.temperature.detach()

        with torch.no_grad():
            backup = torch.zeros_like(r)
            nonterminal = (1.0 - d).squeeze(-1).bool()
            if nonterminal.any():
                sp_nt = sp[nonterminal]
                ap_nt, action_logp_nt, share_logp_nt = self.actor.sample(sp_nt, deterministic=False)
                q_nt = torch.min(self.q1_targ(sp_nt, ap_nt), self.q2_targ(sp_nt, ap_nt))
                entropy_logp_nt = self._entropy_logp(action_logp_nt, share_logp_nt)
                entropy_logp_nt = entropy_logp_nt * self._active_budget_mask(sp_nt, entropy_logp_nt)
                backup[nonterminal] = q_nt - alpha * entropy_logp_nt
            target = r + (1.0 - d) * self.cfg.gamma * backup
            target = sanitize_tensor(target)

        q1_loss = self._critic_loss(self.q1(s, a), target)
        q2_loss = self._critic_loss(self.q2(s, a), target)

        if hasattr(self, "critic_regularizer"):
            q1_loss = q1_loss + self.critic_regularizer(s, a, self.q1)
            q2_loss = q2_loss + self.critic_regularizer(s, a, self.q2)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.grad_clip_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.grad_clip_norm)
        self.q2_opt.step()

        a_new, action_logp, share_logp = self.actor.sample(s, deterministic=False)
        q_new = torch.min(self.q1(s, a_new), self.q2(s, a_new))
        entropy_logp = self._entropy_logp(action_logp, share_logp)
        active_budget = self._active_budget_mask(s, entropy_logp)
        entropy_logp = entropy_logp * active_budget
        actor_loss = (self.temperature.detach() * entropy_logp - q_new).mean()
        if hasattr(self, "actor_regularizer"):
            actor_loss = actor_loss + self.actor_regularizer(s, a_new)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
        self.actor_opt.step()

        if hasattr(self, "after_actor_update"):
            self.after_actor_update()

        alpha_logp = self._alpha_logp(share_logp)
        active = active_budget.squeeze(-1).bool()
        if self.cfg.auto_alpha and active.any():
            alpha_loss = -(self.log_alpha * (alpha_logp[active] + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(math.log(self.cfg.min_temperature), math.log(self.cfg.max_temperature))
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        if self.cfg.stop_on_nonfinite and (has_nonfinite(self.actor) or has_nonfinite(self.q1) or has_nonfinite(self.q2)):
            raise FloatingPointError("Non-finite parameter detected after SAC update.")

        soft_update(self.q1, self.q1_targ, self.cfg.tau)
        soft_update(self.q2, self.q2_targ, self.cfg.tau)

        info = {
            "q1_loss": to_float(q1_loss),
            "q2_loss": to_float(q2_loss),
            "actor_loss": to_float(actor_loss),
            "alpha_loss": to_float(alpha_loss),
            "temperature": to_float(self.temperature),
            "action_logp": to_float(action_logp.detach().mean()),
            "share_logp": to_float(share_logp.detach().mean()),
            "entropy_logp": to_float(entropy_logp.detach().mean()),
            "alpha_logp": to_float(alpha_logp.mean()),
            "actor_lr": self.actor_lr,
            "sac_entropy_space": self.cfg.sac_entropy_space,
        }
        for attr, key in (
            ("last_gate_mean", "gate_mean"),
            ("last_gate_sample", "gate_sample"),
            ("last_gate_logp", "gate_logp"),
            ("last_city_share_logp", "city_share_logp"),
        ):
            value = getattr(self.actor, attr, None)
            if isinstance(value, torch.Tensor):
                info[key] = to_float(value.detach().mean())
        info.update(getattr(self, "extra_update_info", {}))
        for attr, key in (("last_gate_mean", "gate_mean"),
                          ("last_gate_low_fraction", "gate_low_fraction"),
                          ("last_gate_high_fraction", "gate_high_fraction")):
            value = getattr(self.actor, attr, None)
            if value is not None:
                info[key] = to_float(value.mean())
        return info


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: TrainConfig,
        actor_kind: str = "td3_projection",
    ):
        self.cfg = cfg
        init_share = cfg.default_init_total_spend_share if cfg.td3_init_total_spend_share is None else cfg.td3_init_total_spend_share
        if actor_kind == "td3_direct_softmax":
            actor = DirectSoftmaxTD3Actor(
                state_dim,
                action_dim,
                cfg.hidden_dim,
                init_share,
            )
        elif actor_kind == "td3_projection":
            actor = ProjectedTD3Actor(
                state_dim,
                action_dim,
                cfg.hidden_dim,
                init_share,
            )
        elif actor_kind == "td3_pinet":
            from e1_baselines import PiNetActor
            actor = PiNetActor(state_dim, action_dim, cfg.hidden_dim, init_share,
                                   cfg.e1_options["pinet"])
        else:
            raise ValueError(f"Unknown TD3 actor kind: {actor_kind}")
        self.actor_kind = actor_kind
        self.actor = actor.to(cfg.device)
        self.actor_targ = copy.deepcopy(self.actor).to(cfg.device)
        self.q1 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.q2 = QNetwork(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
        self.q1_targ = copy.deepcopy(self.q1).to(cfg.device)
        self.q2_targ = copy.deepcopy(self.q2).to(cfg.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr_critic)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr_critic)
        self.update_count = 0
        self.actor_update_count = 0
        self.last_actor_loss = 0.0

    def _critic_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.critic_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.cfg.huber_delta)
        return F.mse_loss(pred, target)

    def act(self, state: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        with torch.no_grad():
            s = state.unsqueeze(0)
            if noise_std > 0:
                return self.actor.sample_with_action_noise(s, noise_std, self.cfg.noise_clip).squeeze(0)
            return self.actor(s).squeeze(0)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.update_count += 1
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        sp = batch["next_state"]
        d = batch["done"]

        with torch.no_grad():
            backup = torch.zeros_like(r)
            nonterminal = (1.0 - d).squeeze(-1).bool()
            if nonterminal.any():
                sp_nt = sp[nonterminal]
                ap_nt = self.actor_targ.sample_with_action_noise(
                    sp_nt,
                    self.cfg.target_noise,
                    self.cfg.noise_clip,
                )
                backup[nonterminal] = torch.min(self.q1_targ(sp_nt, ap_nt), self.q2_targ(sp_nt, ap_nt))
            target = r + (1.0 - d) * self.cfg.gamma * backup
            target = sanitize_tensor(target)

        q1_loss = self._critic_loss(self.q1(s, a), target)
        q2_loss = self._critic_loss(self.q2(s, a), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.cfg.grad_clip_norm)
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.cfg.grad_clip_norm)
        self.q2_opt.step()

        actor_updated_this_step = 0
        actor_loss_value = self.last_actor_loss
        if self.update_count % self.cfg.policy_delay == 0:
            actor_loss = -self.q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip_norm)
            self.actor_opt.step()
            actor_updated_this_step = 1
            self.actor_update_count += 1
            actor_loss_value = to_float(actor_loss)
            self.last_actor_loss = actor_loss_value

            soft_update(self.actor, self.actor_targ, self.cfg.tau)
            soft_update(self.q1, self.q1_targ, self.cfg.tau)
            soft_update(self.q2, self.q2_targ, self.cfg.tau)

        if self.cfg.stop_on_nonfinite and (has_nonfinite(self.actor) or has_nonfinite(self.q1) or has_nonfinite(self.q2)):
            raise FloatingPointError("Non-finite parameter detected after TD3 update.")

        return {
            "q1_loss": to_float(q1_loss),
            "q2_loss": to_float(q2_loss),
            "actor_loss": finite_or_zero(actor_loss_value),
            "actor_updated": self.actor_update_count,
            "actor_updated_this_step": actor_updated_this_step,
            "td3_update_count": self.update_count,
            "actor_lr": self.cfg.lr_actor,
        }


# =========================================================
# 6. Metric summaries and agent factories
# =========================================================
def summarize_eval_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    out: Dict[str, float] = {}
    metric_keys = ["net_reward", "response_score", "spend", "final_budget"]
    for optional_key in ["net_reward_pess", "uncertainty_penalty"]:
        if all(optional_key in r for r in rows):
            metric_keys.append(optional_key)
    for k in metric_keys:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = float(vals.mean())
        out[k + "_se"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[k + "_median"] = float(np.median(vals))
        out[k + "_p25"] = float(np.percentile(vals, 25))
        out[k + "_p75"] = float(np.percentile(vals, 75))
    return out


def make_sacsa_agent(state_dim: int, action_dim: int, cfg: TrainConfig) -> SACAgent:
    init_share = cfg.default_init_total_spend_share if cfg.dirichlet_init_total_spend_share is None else cfg.dirichlet_init_total_spend_share
    actor = DirichletFeasibleActor(
        state_dim,
        action_dim,
        cfg.hidden_dim,
        min_alpha=cfg.dirichlet_min_alpha,
        max_alpha=cfg.dirichlet_max_alpha,
        logit_clip=cfg.dirichlet_logit_clip,
        init_total_spend_share=init_share,
        init_concentration=cfg.dirichlet_init_concentration,
    )
    return SACAgent(state_dim, action_dim, cfg, actor)


def make_sac_logistic_normal_agent(state_dim: int, action_dim: int, cfg: TrainConfig) -> SACAgent:
    init_share = cfg.default_init_total_spend_share if cfg.ln_init_total_spend_share is None else cfg.ln_init_total_spend_share
    actor = LogisticNormalFeasibleActor(
        state_dim,
        action_dim,
        cfg.hidden_dim,
        init_total_spend_share=init_share,
        init_log_std=cfg.ln_init_log_std,
    )
    return SACAgent(state_dim, action_dim, cfg, actor)


def make_sacsa_reserve_agent(state_dim: int, action_dim: int, cfg: TrainConfig) -> SACAgent:
    init_share = cfg.default_init_total_spend_share if cfg.dirichlet_init_total_spend_share is None else cfg.dirichlet_init_total_spend_share
    actor = FactorizedBetaDirichletFeasibleActor(
        state_dim,
        action_dim,
        cfg.hidden_dim,
        min_alpha=cfg.dirichlet_min_alpha,
        max_alpha=cfg.dirichlet_max_alpha,
        logit_clip=cfg.dirichlet_logit_clip,
        init_total_spend_share=init_share,
        init_concentration=cfg.dirichlet_init_concentration,
        gate_parameterization=cfg.bd_gate_parameterization,
    )
    return SACAgent(state_dim, action_dim, cfg, actor)


def make_sac_logistic_normal_reserve_agent(state_dim: int, action_dim: int, cfg: TrainConfig) -> SACAgent:
    init_share = cfg.default_init_total_spend_share if cfg.ln_init_total_spend_share is None else cfg.ln_init_total_spend_share
    actor = FactorizedBetaLogisticNormalFeasibleActor(
        state_dim,
        action_dim,
        cfg.hidden_dim,
        init_total_spend_share=init_share,
        init_concentration=cfg.dirichlet_init_concentration,
        init_log_std=cfg.ln_init_log_std,
    )
    return SACAgent(state_dim, action_dim, cfg, actor)


# =========================================================
# 7. CSV output
# =========================================================
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# 8. Component library guard
# =========================================================


if __name__ == "__main__":
    raise SystemExit(
        "sacsa_rl_components.py is a component library. "
        "Use iterative_synthetic_sacsa.py as the maintained experiment entry point."
    )
