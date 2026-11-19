"""Logged-data CQL and MOPO training, separate from SAC-SA's synthetic loop.

MOPO uses a bootstrapped Gaussian dynamics ensemble, held-out elite selection,
short rollouts from logged states, uncertainty-penalized model rewards, and
mixed real/model replay. Neither learner queries the target DGP here.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
from sacsa_rl_components import ReplayBuffer, mlp


def logged_batch(logged, normalizer, device):
    transitions = [tr for trajectory in logged for tr in trajectory]
    states = torch.stack([tr.state for tr in transitions]).to(device)
    next_states = torch.stack([tr.next_state if tr.next_state is not None else torch.zeros_like(tr.state)
                               for tr in transitions]).to(device)
    if normalizer is not None:
        states = normalizer.transform(states)
        next_states = normalizer.transform(next_states)
    return {
        "state": states,
        "action": torch.stack([tr.action for tr in transitions]).to(device),
        "reward": torch.tensor([tr.reward for tr in transitions], device=device).float().unsqueeze(-1),
        "next_state": next_states,
        "done": torch.tensor([tr.done for tr in transitions], device=device).float().unsqueeze(-1),
    }


def sample_logged(data, count):
    idx = torch.randint(len(data["state"]), (count,), device=data["state"].device)
    return {key: value[idx] for key, value in data.items()}


def planned_updates(cfg, rollout_batch_size, update_cap):
    """Translate the existing synthetic update schedule into a fixed SGD budget."""
    if rollout_batch_size <= 1:
        return sum(cfg.gradient_steps_per_update for step in range(1, cfg.train_steps + 1)
                   if step >= cfg.update_after_steps and min(step, cfg.replay_size) >= cfg.batch_size
                   and step % cfg.update_every_steps == 0)
    count = episode = steps = 0
    cap = max(update_cap, cfg.gradient_steps_per_update)
    while episode < cfg.train_episodes:
        batch = min(rollout_batch_size, cfg.train_episodes - episode)
        boundaries = [b for b in (cfg.warmup_episodes, cfg.update_after_episodes)
                      if episode < b < episode + batch]
        if boundaries:
            batch = min(boundaries) - episode
        for _ in range(cfg.T):
            steps += batch
            if (episode + 1 > cfg.update_after_episodes
                    and min(steps, cfg.replay_size) >= cfg.batch_size
                    and steps % cfg.update_every_steps == 0):
                count += max(cfg.gradient_steps_per_update, min(cap, cfg.gradient_steps_per_update * batch))
        episode += batch
    return count


class ProbabilisticEnsemble(nn.Module):
    def __init__(self, state_dim, M, T, budget, hidden_dim, options, device):
        super().__init__()
        self.spec = dict(state_dim=state_dim, M=M, T=T, budget=budget,
                         hidden_dim=hidden_dim, options=options)
        self.M, self.T, self.budget = M, T, budget
        self.dynamic_dim = 3 * M  # X, Y, C; O, H and clock follow known identities.
        self.options = options
        dim_in, dim_out = state_dim + M, 1 + self.dynamic_dim
        self.models = nn.ModuleList([mlp(dim_in, (hidden_dim, hidden_dim), 2 * dim_out)
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
        next_state[:, 4 * self.M] = (state[:, 4 * self.M] - action.sum(-1)).clamp_min(0)
        remaining = (state[:, -1] * self.T).round().clamp(1, self.T)
        done = (remaining <= 1).float().unsqueeze(-1)
        angle = 2 * math.pi * (self.T - remaining + 2) / self.T
        next_state[:, -3:] = torch.stack([angle.sin(), angle.cos(), (remaining - 1) / self.T], -1)
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
        actions, _, _ = agent.actor.sample(states)
        next_states, rewards, dones, uncertainty = dynamics.step(states, actions)
        model_buffer.add_batch(states, actions, rewards, next_states, dones)
        generated += len(states)
        penalties.append(float(uncertainty.mean()))
        states = next_states[~dones.squeeze(-1).bool()]
        if not len(states):
            break
    return generated, sum(penalties) / len(penalties)


def train_logged_baseline(agent, method, data, cfg, q, rollout_batch_size, update_cap):
    updates = planned_updates(cfg, rollout_batch_size, update_cap)
    if updates <= 0:
        raise ValueError("Offline update budget is zero; adjust only the smoke override or the declared new-method schedule.")
    started = time.perf_counter()
    rows, info, model_info = [], {}, {}
    opt = cfg.e1_options["mopo"]
    model_buffer = None
    generated = 0
    if method == "mopo_ft":
        dynamics = ProbabilisticEnsemble(data["state"].shape[1], cfg.M, cfg.T, cfg.daily_budget,
                                        cfg.hidden_dim, opt, cfg.device)
        model_info = dynamics.fit(data)
        agent.mopo_dynamics = dynamics
        model_buffer = ReplayBuffer(data["state"].shape[1], cfg.M, opt["model_buffer_size"], cfg.device)
    for step in range(updates):
        if method == "mopo_ft":
            if step % opt["rollout_every_updates"] == 0:
                count, uncertainty = generate_model_rollouts(agent, dynamics, data, model_buffer, opt)
                generated += count
                model_info["mopo_rollout_uncertainty"] = uncertainty
            real_count = max(1, int(cfg.batch_size * opt["real_ratio"]))
            real = sample_logged(data, real_count)
            model = model_buffer.sample(cfg.batch_size - real_count)
            batch = {key: torch.cat([real[key], model[key]], 0) for key in real}
        else:
            batch = sample_logged(data, cfg.batch_size)
        info = agent.update(batch)
        if step == updates - 1 or (step + 1) % max(1, updates // 10) == 0:
            row = {"round": q, "method": method, "method_key": method,
                   "episode": 0, "step": step + 1, "training_source": "logged_only" if method == "cql_ft" else "logged_and_mopo_model",
                   "offline_gradient_updates": step + 1, "planned_offline_gradient_updates": updates,
                   "offline_logged_transitions": len(data["state"]), "mopo_generated_transitions": generated,
                   "offline_elapsed_seconds": time.perf_counter() - started, **model_info, **info}
            rows.append(row)
            print(f"[E1 {method}] round={q} update={step+1}/{updates}", flush=True)
    agent.offline_gradient_updates_total = getattr(agent, "offline_gradient_updates_total", 0) + updates
    agent.offline_gradient_updates_this_round = updates
    agent.mopo_generated_transitions_total = getattr(agent, "mopo_generated_transitions_total", 0) + generated
    return agent, rows
