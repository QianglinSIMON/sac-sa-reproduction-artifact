"""Grid logged-only CQL and probabilistic-ensemble MOPO training.

The fixed learned-simulator validation gate stays in city_panel_training.
Neither offline policy-training branch queries the target DGP.
"""
from __future__ import annotations
import math
import time
from dataclasses import asdict
import torch
from torch import nn
from city_panel_models import ReplayBuffer, build_mlp
from public_city_panel_dgp import (PublicCityPanelConfig, CitySpec, _project_dynamic,
    city_supply_demand_pressure, _time_features_tensor)


def logged_batch(logs, bundle, config):
    device = config.device
    state = logs.states.reshape(-1, logs.states.shape[-1]).to(device)
    next_state = logs.next_states.reshape_as(state).to(device)
    t_index = torch.arange(config.T, device=device).repeat(logs.days)
    reward = (logs.rewards.reshape(-1).to(device) - bundle.reward_center_by_t[t_index]) / bundle.reward_scale
    return {"state": bundle.state_normalizer.transform(state),
            "action": logs.actions.reshape(-1, config.M).to(device),
            "reward": reward.unsqueeze(-1),
            "next_state": bundle.state_normalizer.transform(next_state),
            "done": logs.dones.reshape(-1, 1).to(device)}


def sample_logged(data, count):
    idx = torch.randint(len(data["state"]), (count,), device=data["state"].device)
    return {key: value[idx] for key, value in data.items()}

class ProbabilisticEnsemble(nn.Module):
    def __init__(self, state_dim, M, T, budget, hidden_dim, options, device, normalizer_mean, normalizer_scale, panel_config):
        super().__init__()
        self.spec = dict(state_dim=state_dim, M=M, T=T, budget=budget,
                         hidden_dim=hidden_dim, options=options, normalizer_mean=normalizer_mean,
                         normalizer_scale=normalizer_scale, panel_config=panel_config)
        self.M, self.T, self.budget = M, T, budget
        self.dynamic_dim = 10 * M  # Nine dynamic coordinates plus derived pressure.
        restored = dict(panel_config)
        if restored.get("city_specs") is not None:
            restored["city_specs"] = tuple(CitySpec(**v) if isinstance(v, dict) else v for v in restored["city_specs"])
        self.panel_config = PublicCityPanelConfig(**restored)
        self.register_buffer("normalizer_mean", torch.as_tensor(normalizer_mean).clone())
        self.register_buffer("normalizer_scale", torch.as_tensor(normalizer_scale).clone())
        self.options = options
        dim_in, dim_out = state_dim + M, 1 + self.dynamic_dim
        self.models = nn.ModuleList([build_mlp(dim_in, hidden_dim, 2 * dim_out, layer_norm=True)
                                     for _ in range(options["ensemble_size"])])
        for key, size, fill in [("x_mean", dim_in, 0), ("x_std", dim_in, 1),
                                ("y_mean", dim_out, 0), ("y_std", dim_out, 1)]:
            self.register_buffer(key, torch.full((size,), float(fill)))
        self.elites = list(range(options["elite_size"]))
        self.to(device)

    def _predict(self, model, x):
        mean, raw = model((x - self.x_mean) / self.x_std).chunk(2, -1)
        logvar = 0.5 - nn.functional.softplus(0.5 - raw)
        logvar = -10.0 + nn.functional.softplus(logvar + 10.0)
        return mean, logvar

    def fit(self, data):
        n = len(data["state"])
        if n < 4:
            raise ValueError("MOPO requires at least four logged transitions for model train/holdout splitting.")
        x = torch.cat([data["state"], data["action"] / self.budget], -1)
        y = torch.cat([data["reward"],
                       data["next_state"][:, :self.dynamic_dim] - data["state"][:, :self.dynamic_dim]], -1)
        mask = torch.cat([torch.ones_like(data["done"]),
                          (1 - data["done"]).expand(-1, self.dynamic_dim)], -1)
        permutation = torch.randperm(n, device=x.device)
        n_holdout = max(1, min(n - 2, int(n * self.options["holdout_fraction"])))
        holdout, training = permutation[:n_holdout], permutation[n_holdout:]
        with torch.no_grad():
            self.x_mean.copy_(x[training].mean(0))
            self.x_std.copy_(x[training].std(0, unbiased=False).clamp_min(1e-4))
            valid = mask[training]
            mean = (y[training] * valid).sum(0) / valid.sum(0).clamp_min(1)
            variance = ((y[training] - mean).square() * valid).sum(0) / valid.sum(0).clamp_min(1)
            self.y_mean.copy_(mean)
            self.y_std.copy_(variance.sqrt().clamp_min(1e-4))
        target = (y - self.y_mean) / self.y_std
        losses = []
        for model in self.models:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.options["model_lr"])
            bootstrap = training[torch.randint(len(training), (len(training),), device=x.device)]
            for _ in range(self.options["model_epochs"]):
                order = bootstrap[torch.randperm(len(bootstrap), device=x.device)]
                for start in range(0, len(order), self.options["model_batch_size"]):
                    idx = order[start:start + self.options["model_batch_size"]]
                    mean, logvar = self._predict(model, x[idx])
                    nll = ((mean - target[idx]).square() * (-logvar).exp() + logvar) * mask[idx]
                    loss = nll.sum() / mask[idx].sum().clamp_min(1)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite MOPO dynamics loss.")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
            with torch.no_grad():
                mean, logvar = self._predict(model, x[holdout])
                nll = ((mean - target[holdout]).square() * (-logvar).exp() + logvar) * mask[holdout]
                losses.append(float(nll.sum() / mask[holdout].sum().clamp_min(1)))
        self.elites = sorted(range(len(losses)), key=losses.__getitem__)[:self.options["elite_size"]]
        self.eval()
        return {"mopo_model_holdout_nll": sum(losses[i] for i in self.elites) / len(self.elites),
                "mopo_model_training_transitions": len(training), "mopo_model_holdout_transitions": len(holdout)}

    @torch.no_grad()
    def step(self, state, action):
        x = torch.cat([state, action / self.budget], -1)
        outputs = [self._predict(self.models[i], x) for i in self.elites]
        means = torch.stack([v[0] for v in outputs])
        stds = torch.stack([(0.5 * v[1]).exp() for v in outputs])
        selected = torch.randint(len(outputs), (len(state),), device=state.device)
        rows = torch.arange(len(state), device=state.device)
        sampled = means[selected, rows] + stds[selected, rows] * torch.randn_like(means[0])
        prediction = sampled * self.y_std + self.y_mean
        # MOPO's max ensemble predictive-standard-deviation norm, in standardized
        # response/transition coordinates, converted back to reward units.
        uncertainty = stds.norm(dim=-1).amax(0, keepdim=False).unsqueeze(-1) * self.y_std[0]
        reward = prediction[:, :1] - float(self.options["penalty"]) * uncertainty
        next_state = state.clone()
        next_state[:, :self.dynamic_dim] += prediction[:, 1:]
        # Apply the same known state constraints and derived pressure identity.
        raw = next_state[:, :self.dynamic_dim] * self.normalizer_scale[:self.dynamic_dim] + self.normalizer_mean[:self.dynamic_dim]
        blocks = [raw[:, :self.M], raw[:, self.M:2*self.M]]
        blocks.extend(raw[:, i*self.M:(i+1)*self.M] for i in range(3, 10))
        dynamic = _project_dynamic(torch.stack(blocks, -1), self.panel_config)
        corrected = [dynamic[..., 0], dynamic[..., 1], city_supply_demand_pressure(dynamic[..., 0], dynamic[..., 1])]
        corrected.extend(dynamic[..., i] for i in range(2, 9))
        raw = torch.cat(corrected, -1)
        next_state[:, :self.dynamic_dim] = (raw - self.normalizer_mean[:self.dynamic_dim]) / self.normalizer_scale[:self.dynamic_dim]
        next_state[:, -5] = (state[:, -5] - action.sum(-1)).clamp_min(0)
        remaining = (state[:, -1] * self.T).round().clamp(1, self.T)
        done = (remaining <= 1).float().unsqueeze(-1)
        next_t = (self.T - remaining + 1).long().clamp(max=self.T-1)
        next_state[:, -4:] = _time_features_tensor(self.T, state.device, state.dtype)[next_t]
        next_state[done.squeeze(-1).bool()] = 0
        if not torch.isfinite(next_state).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("Nonfinite MOPO model rollout.")
        return next_state, reward, done, uncertainty

    def export(self):
        return {"spec": self.spec, "state_dict": self.state_dict(), "elites": self.elites}

    @classmethod
    def from_export(cls, data, device):
        model = cls(**data["spec"], device=device)
        model.load_state_dict(data["state_dict"])
        model.elites = list(data["elites"])
        return model.eval()

@torch.no_grad()
def generate_model_rollouts(agent, dynamics, data, model_buffer, options):
    states = sample_logged(data, options["rollout_batch_size"])["state"]
    generated = 0
    penalties = []
    for _ in range(options["rollout_length"]):
        actions, _, _, _ = agent.actor.sample(states)
        next_states, rewards, dones, uncertainty = dynamics.step(states, actions)
        model_buffer.add_batch(states, actions, rewards, next_states, dones)
        generated += len(states)
        penalties.append(float(uncertainty.mean()))
        states = next_states[~dones.squeeze(-1).bool()]
        if not len(states):
            break
    return generated, sum(penalties) / len(penalties)


def train_logged_baseline(agent, method, logs, bundle, config, policy_config, q):
    data = logged_batch(logs, bundle, config)
    pc = policy_config
    episodes = pc.later_round_train_episodes if q > 1 and pc.later_round_train_episodes > 0 else pc.train_episodes
    # Exactly the update count of Grid's original rollout-batch loop.
    count = trained = 0
    while trained < episodes:
        trained += min(pc.rollout_batch_size, episodes - trained)
        if min(trained * config.T, pc.replay_size) >= pc.batch_size:
            count += pc.updates_per_rollout_batch
    if count <= 0:
        raise ValueError("Offline update budget is zero.")
    opt = agent.options["mopo"]
    info, rows, model_info = {}, [], {}
    generated = 0
    started = time.perf_counter()
    if method == "mopo_ft":
        model = ProbabilisticEnsemble(data["state"].shape[1], config.M, config.T, config.daily_budget,
            pc.hidden_dims, opt, config.device,
            bundle.state_normalizer.mean.detach().cpu(), bundle.state_normalizer.scale.detach().cpu(),
            asdict(bundle.panel_config))
        model_info = model.fit(data)
        agent.mopo_dynamics = model
        buffer = ReplayBuffer(opt["model_buffer_size"], data["state"].shape[1], config.M, torch.device(config.device))
    for step in range(count):
        if method == "mopo_ft":
            if step % opt["rollout_every_updates"] == 0:
                n, uncertainty = generate_model_rollouts(agent, model, data, buffer, opt)
                generated += n
                model_info["mopo_rollout_uncertainty"] = uncertainty
            real_n = max(1, int(pc.batch_size * opt["real_ratio"]))
            real = sample_logged(data, real_n)
            synthetic = buffer.sample(pc.batch_size - real_n)
            batch = {key: torch.cat([real[key], synthetic[key]], 0) for key in real}
        else:
            batch = sample_logged(data, pc.batch_size)
        info = agent.update(batch)
        if step == count - 1 or (step + 1) % max(1, count // 10) == 0:
            rows.append({"round": q, "method": method,
                "training_source": "logged_only" if method == "cql_ft" else "logged_and_mopo_model",
                "offline_gradient_updates": step + 1, "planned_offline_gradient_updates": count,
                "offline_logged_transitions": len(data["state"]), "mopo_generated_transitions": generated,
                "offline_elapsed_seconds": time.perf_counter()-started, **model_info, **info})
    agent.offline_gradient_updates_total = getattr(agent, "offline_gradient_updates_total", 0) + count
    agent.offline_gradient_updates_this_round = count
    agent.mopo_generated_transitions_total = getattr(agent, "mopo_generated_transitions_total", 0) + generated
    return agent, rows
