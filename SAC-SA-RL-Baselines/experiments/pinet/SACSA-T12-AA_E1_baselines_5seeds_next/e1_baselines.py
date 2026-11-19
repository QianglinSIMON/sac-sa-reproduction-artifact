"""E1 baselines. Original SAC/TD3 losses are unchanged unless a hook is present.

CQL: Kumar et al., NeurIPS 2020 (continuous importance-sampled CQL(H)).
PiNet: https://arxiv.org/abs/2508.10480 (PyTorch budget specialization).
The Gaussian Lagrangian policy penalizes proposals, not filtered actions.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Dirichlet, Normal

from sacsa_rl_components import (
    EPS, SACAgent, TD3Agent, FactorizedBetaDirichletFeasibleActor,
    ProjectedTD3Actor, get_last_linear, infer_remaining_budget, mlp,
    project_nonnegative_l1_ball,
)

from e1_methods import METHODS, NEW_METHODS


def configure_parser(parser):
    """Load a saved complete argument set; explicit CLI options still win."""
    parser.add_argument("--e1_config", default=None)
    parser.add_argument("--e1_options", default=str(Path(__file__).with_name("baseline_settings.json")))
    pre, _ = parser.parse_known_args()
    if pre.e1_config:
        values = json.loads(Path(pre.e1_config).read_text(encoding="utf-8"))
        known = {a.dest for a in parser._actions}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown frozen configuration fields: {sorted(unknown)}")
        parser.set_defaults(**values)


def load_settings(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_bd_actor(state_dim, action_dim, cfg):
    initial = cfg.dirichlet_init_total_spend_share
    if initial is None:
        initial = cfg.default_init_total_spend_share
    return FactorizedBetaDirichletFeasibleActor(
        state_dim, action_dim, cfg.hidden_dim,
        cfg.dirichlet_min_alpha, cfg.dirichlet_max_alpha,
        cfg.dirichlet_logit_clip, init_total_spend_share=initial,
        init_concentration=cfg.dirichlet_init_concentration,
        gate_parameterization=cfg.bd_gate_parameterization,
    )


class CQLAgent(SACAgent):
    """CQL on the state-dependent monetary simplex, using true action densities."""

    def critic_regularizer(self, state, data_action, critic):
        opt = self.cfg.e1_options["cql"]
        active = infer_remaining_budget(state, self.action_dim) > EPS
        if not active.any():
            return critic(state, data_action).sum() * 0.0
        s, a = state[active], data_action[active]
        n = int(opt["n_action_samples"])
        repeated = s.repeat_interleave(n, dim=0)
        h = infer_remaining_budget(repeated, self.action_dim).unsqueeze(-1)
        with torch.no_grad():
            # Uniform on {a >= 0, sum(a) <= H}: add a slack coordinate.
            share = Dirichlet(torch.ones((len(repeated), self.action_dim + 1), device=s.device)).sample()
            uniform_action = h * share[:, :self.action_dim]
            uniform_logp = math.lgamma(self.action_dim + 1) - self.action_dim * h.log()
            policy_action, policy_logp, _ = self.actor.sample(repeated)
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

    def __init__(self, state_dim, action_dim, cfg):
        super().__init__()
        self.M = action_dim
        self.entropy_dim = action_dim
        self.entropy_logp_scope = "proposal"
        self.net = mlp(state_dim, (cfg.hidden_dim, cfg.hidden_dim), 2 * action_dim)
        initial = cfg.dirichlet_init_total_spend_share
        if initial is None:
            initial = cfg.default_init_total_spend_share
        fraction = max(min(initial / action_dim, 1.0 - 1e-4), 1e-4)
        layer = get_last_linear(self.net)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias[:action_dim].fill_(math.log(fraction / (1.0 - fraction)))
            layer.bias[action_dim:].fill_(float(cfg.e1_options["lagrangian"]["initial_log_std"]))
        self.last_constraint_diagnostics = {}

    def sample(self, state, deterministic=False):
        mean, log_std = self.net(state).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5.0, 1.0).exp())
        z = mean if deterministic else distribution.rsample()
        fraction = torch.sigmoid(z)
        h = infer_remaining_budget(state, self.M).unsqueeze(-1).clamp_min(0)
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
        return action, logp, logp


class LagrangianSACAgent(SACAgent):
    def __init__(self, state_dim, action_dim, cfg, actor):
        super().__init__(state_dim, action_dim, cfg, actor)
        self.multiplier = torch.tensor(float(cfg.e1_options["lagrangian"]["initial_multiplier"]), device=cfg.device)

    def actor_regularizer(self, state, action):
        # The last actor sample produced this differentiable, pre-filter cost.
        self.last_proposal_cost = self.actor.proposal_cost.mean()
        return self.multiplier.detach() * self.last_proposal_cost

    def after_actor_update(self):
        opt = self.cfg.e1_options["lagrangian"]
        with torch.no_grad():
            self.multiplier.add_((0.0 if getattr(self, "fixed_penalty", False) else float(opt["dual_lr"])) *
                                 (self.last_proposal_cost.detach() - float(opt["cost_limit"]))).clamp_(min=0)
        self.extra_update_info = {
            "lagrange_multiplier": float(self.multiplier),
            "proposal_cost": float(self.last_proposal_cost.detach()),
        }
        self.last_proposal_cost = self.last_proposal_cost.detach()
        self.actor.proposal_cost = self.actor.proposal_cost.detach()


class PiNetActor(ProjectedTD3Actor):
    """TD3 with PiNet's DR projection and implicit derivative, specialized to budgets."""

    def __init__(self, state_dim, action_dim, hidden_dim, init_share=None, settings=None):
        super().__init__(state_dim, action_dim, hidden_dim, init_share)
        from pinet_budget import PiNetBudgetLayer
        self.pinet_layer = PiNetBudgetLayer(settings)
        self.last_constraint_diagnostics = {}

    def action_from_fraction(self, state, fraction):
        h = infer_remaining_budget(state, self.M).unsqueeze(-1).clamp_min(0)
        feasible = self.pinet_layer(fraction)
        diagnostics = dict(self.pinet_layer.last_diagnostics)
        diagnostics["solver_numerical_repair"] = h.detach() * diagnostics.pop("solver_numerical_repair_fraction")
        self.last_constraint_diagnostics = diagnostics
        return h * feasible



def make_e1_agent(method, state_dim, action_dim, cfg):
    if method == "td3_pinet":
        return TD3Agent(state_dim, action_dim, cfg, actor_kind=method)
    if method in {"sac_lag", "sac_fixed_penalty"}:
        agent = LagrangianSACAgent(state_dim, action_dim, cfg, FilteredGaussianActor(state_dim, action_dim, cfg))
        agent.fixed_penalty = method == "sac_fixed_penalty"
        return agent
    actor = make_bd_actor(state_dim, action_dim, cfg)
    if method == "cql_ft":
        return CQLAgent(state_dim, action_dim, cfg, actor)
    if method in {"mopo_ft", "mopo_short_ft"}:
        return SACAgent(state_dim, action_dim, cfg, actor)
    raise ValueError(method)


# Keep checkpoint writers and readers on one schema.
CHECKPOINT_STATE_FIELDS = (
    'actor',
    'actor_targ',
    'q1',
    'q2',
    'q1_targ',
    'q2_targ',
    'actor_opt',
    'q1_opt',
    'q2_opt',
    'alpha_opt',
)
CHECKPOINT_TENSORS = (
    'log_alpha',
    'multiplier',
)
CHECKPOINT_COUNTERS = (
    'update_count',
    'actor_update_count',
    'last_actor_loss',
    'offline_gradient_updates_total',
    'offline_gradient_updates_this_round',
    'mopo_generated_transitions_total',
)


def checkpoint_agent(path, agent, metadata=None):
    """Save networks, optimizers and new-method state, not just policy weights."""
    data = {"train_config": asdict(agent.cfg), "metadata": metadata or {}}
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
    data = torch.load(path, map_location=agent.cfg.device, weights_only=False)
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
        agent.mopo_dynamics = ProbabilisticEnsemble.from_export(data["mopo_dynamics"], agent.cfg.device)
    return data["metadata"]


def collect_constraint_diagnostics(policy, totals):
    for key, value in getattr(policy, "last_constraint_diagnostics", {}).items():
        number = float(value.detach().mean())
        totals[key] = totals.get(key, 0.0) + number


def save_baseline_checkpoint(out_dir, q, method, agent, offline=False):
    if agent is None:
        return
    folder = Path(out_dir) / ("offline_pre_online" if offline else "policies") / f"round_{q}"
    checkpoint_agent(folder / "e1_agent.pt", agent, {"method": method, "round": q,
                                                    "phase": "offline" if offline else "post_online"})


def save_input_fingerprints(out_dir, coefs, splits):
    def digest_tensors(tensors):
        digest = hashlib.sha256()
        for value in tensors:
            value = torch.as_tensor(value).detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()
    hashes = {"dgp": digest_tensors(coefs[key] for key in sorted(coefs))}
    for name, logged in splits.items():
        hashes[name] = digest_tensors(value for trajectory in logged for tr in trajectory
                                     for value in (tr.state, tr.action, [tr.reward, tr.done, tr.t], tr.w))
    Path(out_dir, "input_fingerprints.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
