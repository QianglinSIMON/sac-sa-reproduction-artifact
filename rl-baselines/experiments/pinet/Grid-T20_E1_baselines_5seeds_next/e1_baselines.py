"""E1 Grid baselines, using Grid's own state layout, MLPs and SAC update.

PiNet uses Douglas--Rachford projection and an implicit derivative on the input device.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Beta, Dirichlet, Normal
from city_panel_models import (EPS, SACAgent, SACConfig, QNetwork, BetaDirichletActor,
    build_mlp, _last_linear, remaining_budget, feasible_action)

from e1_methods import METHODS, NEW_METHODS
ACTOR_CRITIC_METHODS = set(METHODS) - {"dfl_pto"}



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


class DirectSoftmaxTD3Actor(nn.Module):
    """TD3 actor with an M+1 softmax over city spending and unspent budget."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, init_total_spend_share: Optional[float] = None):
        super().__init__()
        self.M = action_dim
        self.net = build_mlp(state_dim, hidden_dim, action_dim + 1, layer_norm=True)
        if init_total_spend_share is not None:
            self._init_pacing_logits(init_total_spend_share)

    def _init_pacing_logits(self, total_spend_share: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        save_bias = math.log(max(self.M * (1.0 - total_spend_share) / total_spend_share, EPS))
        last = _last_linear(self.net)
        if last is not None:
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()
                last.bias[-1] = save_bias

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(self.net(state), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

    def action_from_logits(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        h = remaining_budget(state).unsqueeze(-1).clamp_min(0.0)
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

    def sample(self, state, deterministic=False):
        action = self(state) if deterministic else self.sample_with_action_noise(state, 0.1, 0.5)
        zero = torch.zeros((len(state), 1), device=state.device)
        return action, zero, zero, zero



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
        self.net = build_mlp(state_dim, hidden_dim, action_dim, layer_norm=True)
        if init_total_spend_share is not None:
            self._init_pacing_proposal(init_total_spend_share)

    def _init_pacing_proposal(self, total_spend_share: float) -> None:
        total_spend_share = float(np.clip(total_spend_share, 1e-4, 1.0 - 1e-4))
        per_city_share = float(np.clip(total_spend_share / self.M, 1e-4, 1.0 - 1e-4))
        last = _last_linear(self.net)
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
        h = remaining_budget(state).unsqueeze(-1).clamp_min(0.0)
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

    def sample(self, state, deterministic=False):
        action = self(state) if deterministic else self.sample_with_action_noise(state, 0.1, 0.5)
        zero = torch.zeros((len(state), 1), device=state.device)
        return action, zero, zero, zero



class PiNetActor(ProjectedTD3Actor):
    """TD3 with PiNet's DR projection and implicit derivative, specialized to budgets."""

    def __init__(self, state_dim, action_dim, hidden_dim, init_share=None, settings=None):
        super().__init__(state_dim, action_dim, hidden_dim, init_share)
        from pinet_budget import PiNetBudgetLayer
        self.pinet_layer = PiNetBudgetLayer(settings)
        self.last_constraint_diagnostics = {}

    def action_from_fraction(self, state, fraction):
        h = remaining_budget(state).unsqueeze(-1).clamp_min(0)
        feasible = self.pinet_layer(fraction)
        diagnostics = dict(self.pinet_layer.last_diagnostics)
        diagnostics["solver_numerical_repair"] = h.detach() * diagnostics.pop("solver_numerical_repair_fraction")
        self.last_constraint_diagnostics = diagnostics
        return h * feasible



class CQLAgent(SACAgent):
    """CQL on the state-dependent monetary simplex, using true action densities."""

    def critic_regularizer(self, state, data_action, critic):
        opt = self.options["cql"]
        active = remaining_budget(state) > EPS
        if not active.any():
            return critic(state, data_action).sum() * 0.0
        s, a = state[active], data_action[active]
        n = int(opt["n_action_samples"])
        repeated = s.repeat_interleave(n, dim=0)
        h = remaining_budget(repeated).unsqueeze(-1)
        with torch.no_grad():
            # Uniform on {a >= 0, sum(a) <= H}: add a slack coordinate.
            share = Dirichlet(torch.ones((len(repeated), self.actor.M + 1), device=s.device)).sample()
            uniform_action = h * share[:, :self.actor.M]
            uniform_logp = math.lgamma(self.actor.M + 1) - self.actor.M * h.log()
            policy_action, _, _, _ = self.actor.sample(repeated)
            # The Grid SAC actor returns clipped share log probabilities for its
            # original entropy objective. CQL needs the unclipped monetary density.
            gate = (policy_action.sum(-1, keepdim=True) / h).clamp(EPS, 1-EPS)
            shares = (policy_action / (h * gate)).clamp_min(EPS)
            shares = shares / shares.sum(-1, keepdim=True)
            city_alpha, gate_alpha, gate_beta = self.actor.concentrations(repeated)
            policy_logp = (Dirichlet(city_alpha).log_prob(shares).unsqueeze(-1)
                           + Beta(gate_alpha, gate_beta).log_prob(gate)
                           - (self.actor.M - 1) * gate.log() - self.actor.M * h.log())
        temp = float(opt["temperature"])
        # Each proposal estimates the same integral; divide by the total count.
        values = torch.cat([
            (critic(repeated, uniform_action) / temp - uniform_logp).reshape(len(s), n),
            (critic(repeated, policy_action) / temp - policy_logp).reshape(len(s), n),
        ], dim=1)
        integral = temp * (torch.logsumexp(values, dim=1) - math.log(2 * n))
        penalty = (integral - critic(s, a).squeeze(-1)).mean()
        self.extra_update_info = {"cql_conservative_loss": float(penalty.detach()),
                                  "cql_weight": float(opt["weight"])}
        return float(opt["weight"]) * penalty


class FilteredGaussianActor(nn.Module):
    """Sigmoid-Normal proposal fractions; Euclidean filtering before execution.

    Entropy is defined on proposal fractions. It is NOT the density of the
    projected action, which can have boundary mass.
    """

    def __init__(self, state_dim, action_dim, hidden_dims, initial, options):
        super().__init__()
        self.M = action_dim
        self.entropy_dim = action_dim
        self.entropy_logp_scope = "proposal"
        self.net = build_mlp(state_dim, hidden_dims, 2 * action_dim, layer_norm=True)
        fraction = max(min(initial / action_dim, 1.0 - 1e-4), 1e-4)
        layer = _last_linear(self.net)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias[:action_dim].fill_(math.log(fraction / (1.0 - fraction)))
            layer.bias[action_dim:].fill_(float(options["lagrangian"]["initial_log_std"]))
        self.last_constraint_diagnostics = {}

    def sample(self, state, deterministic=False):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5.0, 1.0).exp())
        z = mean if deterministic else distribution.rsample()
        fraction = torch.sigmoid(z)
        h = remaining_budget(state).unsqueeze(-1).clamp_min(0)
        feasible = project_nonnegative_l1_ball(fraction, torch.ones_like(h))
        action = h * feasible
        proposal = h * fraction
        active = (h > EPS).to(h.dtype)
        self.proposal_cost = (fraction.sum(-1, keepdim=True) - 1.0).clamp_min(0) * active
        self.last_constraint_diagnostics = {
            "proposal_violation": (self.proposal_cost.detach() > 1e-7).float(),
            "proposal_overspend": (proposal.sum(-1, keepdim=True) - h).clamp_min(0).detach(),
            "filter_distance": (proposal - action).norm(dim=-1, keepdim=True).detach(),
        }
        # log sigmoid'(z) = log sigmoid(z) + log sigmoid(-z).
        log_jac = nn.functional.logsigmoid(z) + nn.functional.logsigmoid(-z)
        logp = (distribution.log_prob(z) - log_jac).sum(-1, keepdim=True)
        return action, logp, logp, fraction.sum(-1, keepdim=True)


class LagrangianSACAgent(SACAgent):
    def __init__(self, state_dim, action_dim, device, config, hidden_dims, init_share, options):
        super().__init__(state_dim, action_dim, device, config, hidden_dims, init_share)
        self.options = options
        self.actor = FilteredGaussianActor(state_dim, action_dim, hidden_dims, init_share, options).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.target_entropy = -float(action_dim) * config.target_entropy_scale
        self.multiplier = torch.tensor(float(options["lagrangian"]["initial_multiplier"]), device=device)

    def actor_regularizer(self, state, action):
        # The last actor sample produced this differentiable, pre-filter cost.
        self.last_proposal_cost = self.actor.proposal_cost.mean()
        return self.multiplier.detach() * self.last_proposal_cost

    def after_actor_update(self):
        opt = self.options["lagrangian"]
        with torch.no_grad():
            self.multiplier.add_((0.0 if getattr(self, "fixed_penalty", False) else float(opt["dual_lr"])) *
                                 (self.last_proposal_cost.detach() - float(opt["cost_limit"]))).clamp_(min=0)
        self.extra_update_info = {
            "lagrange_multiplier": float(self.multiplier),
            "proposal_cost": float(self.last_proposal_cost.detach()),
        }
        self.last_proposal_cost = self.last_proposal_cost.detach()
        self.actor.proposal_cost = self.actor.proposal_cost.detach()


class TD3Agent:
    def __init__(self, state_dim, action_dim, device, config, hidden_dims, init_share, method, options):
        self.device, self.config, self.method, self.options = device, config, method, options
        actor_type = {"td3_direct_softmax": DirectSoftmaxTD3Actor,
                      "td3_projection": ProjectedTD3Actor, "td3_pinet": PiNetActor}[method]
        args = (state_dim, action_dim, hidden_dims, init_share)
        self.actor = (actor_type(*args, settings=options["pinet"]) if method == "td3_pinet"
                      else actor_type(*args)).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.update_count = self.actor_update_count = 0

    def update(self, batch, *, update_actor=True):
        self.update_count += 1
        s, a, r, sp, done = (batch[k] for k in ["state", "action", "reward", "next_state", "done"])
        with torch.no_grad():
            ap = self.actor_target.sample_with_action_noise(sp, .2, .5)
            target = r + self.config.gamma * (1-done) * torch.minimum(self.q1_target(sp, ap), self.q2_target(sp, ap))
        q1_loss = F.smooth_l1_loss(self.q1(s, a), target)
        q2_loss = F.smooth_l1_loss(self.q2(s, a), target)
        for loss, critic, optimizer in [(q1_loss, self.q1, self.q1_optimizer), (q2_loss, self.q2, self.q2_optimizer)]:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.config.grad_clip)
            optimizer.step()
        actor_loss = torch.zeros((), device=self.device)
        changed = update_actor and self.update_count % 2 == 0
        if changed:
            actor_loss = -self.q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_optimizer.step()
            self.actor_update_count += 1
            with torch.no_grad():
                for source, target_net in [(self.actor, self.actor_target), (self.q1, self.q1_target), (self.q2, self.q2_target)]:
                    for parameter, target_parameter in zip(source.parameters(), target_net.parameters()):
                        target_parameter.mul_(1-self.config.tau).add_(parameter, alpha=self.config.tau)
        return dict(q1_loss=float(q1_loss.detach()), q2_loss=float(q2_loss.detach()),
                    actor_loss=float(actor_loss.detach()), actor_updated=float(changed),
                    td3_update_count=self.update_count, td3_actor_update_count=self.actor_update_count)


def make_agent(method, state_dim, M, device, policy_config, options):
    cfg = SACConfig(auto_alpha=policy_config.auto_alpha, fixed_temperature=policy_config.fixed_temperature,
                    actor_lr=policy_config.actor_lr, critic_lr=policy_config.critic_lr,
                    alpha_lr=policy_config.alpha_lr,
                    bd_gate_parameterization=policy_config.bd_gate_parameterization)
    args = (state_dim, M, device, cfg, policy_config.hidden_dims, policy_config.init_spend_share)
    if method.startswith("td3"):
        return TD3Agent(*args, method, options)
    if method in {"sac_lag", "sac_fixed_penalty"}:
        agent = LagrangianSACAgent(*args, options)
        agent.fixed_penalty = method == "sac_fixed_penalty"
        return agent
    agent = (CQLAgent if method == "cql_ft" else SACAgent)(*args)
    agent.options = options
    return agent


# Keep checkpoint writers and readers on one schema.
CHECKPOINT_STATE_FIELDS = (
    'actor',
    'actor_target',
    'q1',
    'q2',
    'q1_target',
    'q2_target',
    'actor_optimizer',
    'q1_optimizer',
    'q2_optimizer',
    'alpha_optimizer',
)
CHECKPOINT_TENSORS = (
    'log_temperature',
    'multiplier',
)
CHECKPOINT_COUNTERS = (
    'update_count',
    'actor_update_count',
    'offline_gradient_updates_total',
    'offline_gradient_updates_this_round',
    'mopo_generated_transitions_total',
)


def checkpoint_agent(path, agent, metadata=None):
    data = {"metadata": metadata or {}, "sac_config": asdict(agent.config)}
    for name in CHECKPOINT_STATE_FIELDS:
        value = getattr(agent, name, None)
        if value is not None:
            data[name] = value.state_dict()
    for name in CHECKPOINT_TENSORS:
        if hasattr(agent, name):
            data[name] = getattr(agent, name).detach().cpu()
    for name in CHECKPOINT_COUNTERS:
        if hasattr(agent, name):
            data[name] = getattr(agent, name)
    if hasattr(agent, "mopo_dynamics"):
        data["mopo_dynamics"] = agent.mopo_dynamics.export()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["bd_gate_parameterization"] = getattr(agent.actor, "gate_parameterization", None)
    data["fixed_penalty"] = getattr(agent, "fixed_penalty", False)
    torch.save(data, path)


def restore_agent(path, agent):
    data = torch.load(path, map_location=agent.device, weights_only=False)
    expected_gate = getattr(agent.actor, "gate_parameterization", None)
    saved_gate = data.get("bd_gate_parameterization", "softplus" if expected_gate else None)
    if expected_gate != saved_gate:
        raise ValueError("Checkpoint Beta gate parameterization differs; use its original configuration.")
    if data.get("fixed_penalty", False) != getattr(agent, "fixed_penalty", False):
        raise ValueError("Checkpoint penalty adaptation differs from the requested method.")
    for name in CHECKPOINT_STATE_FIELDS:
        if name in data:
            getattr(agent, name).load_state_dict(data[name])
    for name in CHECKPOINT_TENSORS:
        if name in data:
            with torch.no_grad():
                getattr(agent, name).copy_(data[name])
    for name in CHECKPOINT_COUNTERS:
        if name in data:
            setattr(agent, name, data[name])
    if "mopo_dynamics" in data:
        from e1_offline import ProbabilisticEnsemble
        agent.mopo_dynamics = ProbabilisticEnsemble.from_export(data["mopo_dynamics"], agent.device)
    return data["metadata"]


def save_input_fingerprints(out_dir, splits):
    hashes = {}
    for name, data in splits.items():
        digest = hashlib.sha256()
        for key in data.__dataclass_fields__:
            value = getattr(data, key).detach().cpu().contiguous()
            digest.update(str((key, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        hashes[name] = digest.hexdigest()
    cache = Path(__file__).parent / ".cache/public_city_panel/calibration_v8_59e87f571650ad804f9d.pt"
    hashes["frozen_dgp"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    Path(out_dir, "input_fingerprints.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
