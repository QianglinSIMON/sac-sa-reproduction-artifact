"""Neural components for the public city-panel subsidy experiment.

This module intentionally contains no true-DGP construction.  The frozen
environment lives in :mod:`public_city_panel_dgp`; the classes below implement
only the learned simulator, the feasible Beta-Dirichlet SAC actor, critics,
and the one-step DFL score model.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Beta, Dirichlet


EPS = 1e-8


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.append(nn.Linear(width, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.SiLU())
        width = hidden
    layers.append(nn.Linear(width, int(output_dim)))
    return nn.Sequential(*layers)


def _last_linear(module: nn.Module) -> nn.Linear:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            return layer
    raise ValueError("Network has no Linear output layer.")


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(EPS)
    return value + torch.log(-torch.expm1(-value))


def remaining_budget(state: torch.Tensor) -> torch.Tensor:
    """Return raw remaining budget from [..., H, four time features]."""
    if state.shape[-1] < 5:
        raise ValueError("City-panel state is too short to contain H and time features.")
    return state[..., -5]


def feasible_action(action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Numerical feasibility guard; valid actors should already satisfy it."""
    action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    budget = remaining_budget(state).clamp_min(0.0).unsqueeze(-1)
    total = action.sum(dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(total), budget / total.clamp_min(EPS))
    return action * scale


@dataclass
class FeatureNormalizer:
    """Standardize state features while preserving raw H and time coordinates."""

    mean: torch.Tensor
    scale: torch.Tensor
    normalized_dim: int

    @classmethod
    def fit(cls, states: torch.Tensor) -> "FeatureNormalizer":
        if states.ndim != 2 or states.shape[-1] < 5:
            raise ValueError("Expected flattened states with shape [n,state_dim].")
        normalized_dim = int(states.shape[-1] - 5)
        values = states[:, :normalized_dim].float()
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(1e-4)
        return cls(mean=mean, scale=scale, normalized_dim=normalized_dim)

    def to(self, device: torch.device | str) -> "FeatureNormalizer":
        self.mean = self.mean.to(device)
        self.scale = self.scale.to(device)
        return self

    def transform(self, states: torch.Tensor) -> torch.Tensor:
        out = states.clone()
        out[..., : self.normalized_dim] = (
            out[..., : self.normalized_dim] - self.mean
        ) / self.scale
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    def state_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.detach().cpu(),
            "scale": self.scale.detach().cpu(),
            "normalized_dim": int(self.normalized_dim),
        }


class LearnedSimulatorNet(nn.Module):
    """Baseline plus zero-at-zero action-uplift response-transition model.

    The public-city intervention acts city by city and is exactly zero when a
    city's subsidy is zero.  Encoding that known structure prevents the neural
    simulator from manufacturing an action effect at zero spend while retaining
    a flexible context-dependent uplift for positive actions.
    """

    def __init__(
        self,
        input_dim: int,
        M: int,
        dynamic_dim: int,
        hidden_dims: Sequence[int] = (768, 512, 256),
        head_dim: int = 256,
        state_dim: int | None = None,
        action_feature_dim: int | None = None,
    ):
        super().__init__()
        if state_dim is None or action_feature_dim is None:
            raise ValueError(
                "LearnedSimulatorNet requires explicit state_dim and "
                "action_feature_dim for the structured action-uplift model."
            )
        self.M = int(M)
        self.dynamic_dim = int(dynamic_dim)
        self.state_dim = int(state_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.time_dim = int(input_dim - self.state_dim - self.action_feature_dim)
        if self.time_dim <= 0:
            raise ValueError("Simulator input must contain explicit time features.")
        context_dim = self.state_dim + self.time_dim
        layers: list[nn.Module] = []
        width = int(context_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()])
            width = hidden
        self.context_trunk = nn.Sequential(*layers)
        self.baseline_response_head = nn.Sequential(
            nn.Linear(width, int(head_dim)), nn.SiLU(), nn.Linear(int(head_dim), int(M))
        )
        self.baseline_transition_head = nn.Sequential(
            nn.Linear(width, int(head_dim)),
            nn.SiLU(),
            nn.Linear(int(head_dim), int(M * dynamic_dim)),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_feature_dim, int(head_dim)),
            nn.LayerNorm(int(head_dim)),
            nn.SiLU(),
        )
        self.uplift_trunk = nn.Sequential(
            nn.Linear(width + int(head_dim), int(head_dim)),
            nn.LayerNorm(int(head_dim)),
            nn.SiLU(),
        )
        self.uplift_response_head = nn.Linear(int(head_dim), int(M))
        self.uplift_transition_head = nn.Linear(
            int(head_dim), int(M * dynamic_dim)
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.shape[-1] != self.state_dim + self.action_feature_dim + self.time_dim:
            raise ValueError(
                f"Expected simulator input dimension "
                f"{self.state_dim + self.action_feature_dim + self.time_dim}, "
                f"got {inputs.shape[-1]}."
            )
        state = inputs[..., : self.state_dim]
        action_features = inputs[
            ..., self.state_dim : self.state_dim + self.action_feature_dim
        ]
        time = inputs[..., self.state_dim + self.action_feature_dim :]
        context = self.context_trunk(torch.cat([state, time], dim=-1))
        encoded_action = self.action_encoder(action_features)
        uplift = self.uplift_trunk(torch.cat([context, encoded_action], dim=-1))

        # The first M action features are city subsidies divided by B/M, the
        # same reference scale used by the frozen public-city true DGP.
        # Multiplication by this gate imposes exactly zero city-level uplift at
        # zero city-level subsidy without constraining the sign of transitions.
        city_gate = action_features[..., : self.M]
        response = self.baseline_response_head(context)
        response = response + city_gate * self.uplift_response_head(uplift)
        transition = self.baseline_transition_head(context).reshape(
            *inputs.shape[:-1], self.M, self.dynamic_dim
        )
        transition = transition + city_gate.unsqueeze(-1) * self.uplift_transition_head(
            uplift
        ).reshape(*inputs.shape[:-1], self.M, self.dynamic_dim)
        return response, transition.reshape(*inputs.shape[:-1], -1)


class BetaDirichletActor(nn.Module):
    """Feasible policy: Beta total-spend gate times Dirichlet city shares."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 256),
        init_spend_share: float = 0.10,
        init_concentration: float = 20.0,
        min_concentration: float = 1e-3,
        max_concentration: float = 1e3,
    ):
        super().__init__()
        if action_dim < 2:
            raise ValueError("Beta-Dirichlet actor requires at least two cities.")
        self.M = int(action_dim)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.net = build_mlp(state_dim, hidden_dims, self.M + 2, layer_norm=True)
        self._initialize_pacing(init_spend_share, init_concentration)

    def _initialize_pacing(self, spend_share: float, concentration: float) -> None:
        spend_share = float(min(max(spend_share, 1e-4), 1.0 - 1e-4))
        concentration = max(float(concentration), 2.0)
        city = torch.full((self.M,), concentration / self.M)
        gate = torch.tensor([spend_share * concentration, (1.0 - spend_share) * concentration])
        target = torch.cat([city, gate]).clamp_min(self.min_concentration + 1e-5)
        raw = _inverse_softplus(target - self.min_concentration).clamp(-15.0, 15.0)
        output = _last_linear(self.net)
        with torch.no_grad():
            output.weight.zero_()
            output.bias.copy_(raw.to(output.bias))

    def concentrations(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = torch.nan_to_num(self.net(state), nan=0.0, posinf=15.0, neginf=-15.0).clamp(-15.0, 15.0)
        params = torch.nn.functional.softplus(raw) + self.min_concentration
        params = params.clamp(self.min_concentration, self.max_concentration)
        return params[..., : self.M], params[..., self.M : self.M + 1], params[..., self.M + 1 :]

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        city_alpha, gate_alpha, gate_beta = self.concentrations(state)
        city_dist = Dirichlet(city_alpha, validate_args=False)
        gate_dist = Beta(gate_alpha, gate_beta, validate_args=False)
        if deterministic:
            city_share = city_alpha / city_alpha.sum(dim=-1, keepdim=True).clamp_min(EPS)
            gate = gate_alpha / (gate_alpha + gate_beta).clamp_min(EPS)
        else:
            city_share = city_dist.rsample()
            gate = gate_dist.rsample()
        city_share = torch.nan_to_num(city_share, nan=1.0 / self.M, posinf=1.0, neginf=0.0)
        city_share = city_share.clamp_min(EPS)
        city_share = city_share / city_share.sum(dim=-1, keepdim=True).clamp_min(EPS)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(EPS, 1.0 - EPS)
        action = remaining_budget(state).clamp_min(0.0).unsqueeze(-1) * gate * city_share
        action = feasible_action(action, state)
        city_logp = city_dist.log_prob(city_share).unsqueeze(-1).clamp(-30.0, 30.0)
        gate_logp = gate_dist.log_prob(gate).sum(dim=-1, keepdim=True).clamp(-30.0, 30.0)
        return action, city_logp, gate_logp, gate


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512, 256),
    ):
        super().__init__()
        self.net = build_mlp(state_dim + action_dim, hidden_dims, 1, layer_norm=True)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


@dataclass
class SACConfig:
    gamma: float = 1.0
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    fixed_temperature: float = 0.05
    auto_alpha: bool = False
    min_temperature: float = 1e-3
    max_temperature: float = 0.30
    target_entropy_scale: float = 1.0
    grad_clip: float = 5.0


class SACAgent:
    """Twin-critic SAC using city-share entropy and monetary feasible actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: SACConfig,
        policy_hidden_dims: Sequence[int] = (512, 512, 256),
        init_spend_share: float = 0.10,
    ):
        self.device = device
        self.config = config
        self.actor = BetaDirichletActor(
            state_dim, action_dim, policy_hidden_dims, init_spend_share=init_spend_share
        ).to(device)
        self.q1 = QNetwork(state_dim, action_dim, policy_hidden_dims).to(device)
        self.q2 = QNetwork(state_dim, action_dim, policy_hidden_dims).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.log_temperature = torch.tensor(
            math.log(config.fixed_temperature), device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_temperature], lr=config.alpha_lr)
        self.target_entropy = -float(max(action_dim - 1, 1)) * float(config.target_entropy_scale)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(
            self.config.min_temperature, self.config.max_temperature
        )

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action, _, _, _ = self.actor.sample(state.unsqueeze(0), deterministic=deterministic)
        return action.squeeze(0)

    def update(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        update_actor: bool = True,
    ) -> Dict[str, float]:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        done = batch["done"]
        with torch.no_grad():
            next_action, next_city_logp, _, _ = self.actor.sample(next_state)
            next_q = torch.minimum(
                self.q1_target(next_state, next_action), self.q2_target(next_state, next_action)
            )
            next_q = next_q - self.temperature.detach() * next_city_logp
            target = reward + self.config.gamma * (1.0 - done) * next_q

        q1_loss = torch.nn.functional.smooth_l1_loss(self.q1(state, action), target)
        q2_loss = torch.nn.functional.smooth_l1_loss(self.q2(state, action), target)
        self.q1_optimizer.zero_grad(set_to_none=True)
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.grad_clip)
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(set_to_none=True)
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.grad_clip)
        self.q2_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        alpha_loss = torch.zeros((), device=self.device)
        if update_actor:
            sampled_action, city_logp, _, _ = self.actor.sample(state)
            actor_q = torch.minimum(self.q1(state, sampled_action), self.q2(state, sampled_action))
            actor_loss = (self.temperature.detach() * city_logp - actor_q).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_optimizer.step()

            if self.config.auto_alpha:
                alpha_loss = -(
                    self.log_temperature * (city_logp.detach() + self.target_entropy)
                ).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_temperature.clamp_(
                        math.log(self.config.min_temperature), math.log(self.config.max_temperature)
                    )

        with torch.no_grad():
            for source, target_net in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for parameter, target_parameter in zip(source.parameters(), target_net.parameters()):
                    target_parameter.mul_(1.0 - self.config.tau).add_(
                        parameter, alpha=self.config.tau
                    )
        return {
            "q1_loss": float(q1_loss.detach()),
            "q2_loss": float(q2_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "temperature": float(self.temperature.detach()),
            "actor_updated": float(update_actor),
        }


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.state = torch.empty((self.capacity, state_dim), device=device)
        self.action = torch.empty((self.capacity, action_dim), device=device)
        self.reward = torch.empty((self.capacity, 1), device=device)
        self.next_state = torch.empty((self.capacity, state_dim), device=device)
        self.done = torch.empty((self.capacity, 1), device=device)
        self.position = 0
        self.size = 0

    def add_batch(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        n = int(state.shape[0])
        if n > self.capacity:
            state, action, reward, next_state, done = (
                value[-self.capacity :] for value in (state, action, reward, next_state, done)
            )
            n = self.capacity
        indices = (torch.arange(n, device=self.device) + self.position) % self.capacity
        self.state[indices] = state.detach()
        self.action[indices] = action.detach()
        self.reward[indices] = reward.detach().reshape(-1, 1)
        self.next_state[indices] = next_state.detach()
        self.done[indices] = done.detach().reshape(-1, 1)
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("Replay buffer does not contain a full batch.")
        indices = torch.randint(self.size, (int(batch_size),), device=self.device)
        return {
            "state": self.state[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_state": self.next_state[indices],
            "done": self.done[indices],
        }


class OneStepScoreNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
    ):
        super().__init__()
        # Match the DFL-PTO score network used by the synthetic and AA
        # experiments: two SiLU hidden layers without LayerNorm.
        self.net = build_mlp(input_dim, hidden_dims, 1, layer_norm=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)
