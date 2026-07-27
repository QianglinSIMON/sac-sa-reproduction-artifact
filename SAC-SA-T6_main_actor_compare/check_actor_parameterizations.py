"""Fast feasibility and one-update checks for the five synthetic T=6 actors."""
from __future__ import annotations

import math

import torch

from DGP_sim_env import SimConfig, SubsidyAllocationEnv, generate_dgp_coefs
from sacsa_rl_components import (
    TrainConfig,
    make_sac_gaussian_projection_agent,
    make_sac_logistic_normal_agent,
    make_sac_logistic_normal_reserve_agent,
    make_sacsa_agent,
    make_sacsa_reserve_agent,
    project_to_budget_simplex,
    random_paced_action,
)


def main() -> None:
    torch.manual_seed(2026)
    m = 6
    state_dim = 4 * m + 1 + 3
    cfg = TrainConfig(
        seed=2026,
        device="cpu",
        M=m,
        T=6,
        daily_budget=10.0,
        hidden_dim=64,
        batch_size=32,
        auto_alpha=False,
        fixed_temperature=0.05,
        dirichlet_init_total_spend_share=0.20,
        dirichlet_init_concentration=30.0,
        ln_init_total_spend_share=0.20,
        gaussian_projection_init_log_std=-4.0,
    )
    factories = {
        "beta_dirichlet": make_sacsa_reserve_agent,
        "beta_logistic_normal": make_sac_logistic_normal_reserve_agent,
        "direct_dirichlet": make_sacsa_agent,
        "direct_logistic_normal": make_sac_logistic_normal_agent,
        "gaussian_projection": make_sac_gaussian_projection_agent,
    }

    states = torch.randn(256, state_dim)
    budgets = torch.linspace(0.0, 10.0, states.shape[0])
    states[:, 4 * m] = budgets
    next_states = states.clone()
    next_states[:, 4 * m] = (budgets - 0.1).clamp_min(0.0)

    for name, factory in factories.items():
        agent = factory(state_dim, m, cfg)
        actions, _, _ = agent.actor.sample(states, deterministic=False)
        deterministic_actions, _, _ = agent.actor.sample(
            states, deterministic=True
        )
        if not torch.isfinite(actions).all():
            raise AssertionError(f"{name}: nonfinite sampled action")
        if bool((actions < -1e-7).any()):
            raise AssertionError(f"{name}: negative sampled action")
        excess = actions.sum(dim=-1) - budgets
        if float(excess.max()) > 2e-5:
            raise AssertionError(f"{name}: budget violation {float(excess.max())}")

        positive_budget = budgets > 1e-6
        stochastic_fraction = (
            actions.sum(dim=-1)[positive_budget] / budgets[positive_budget]
        ).mean()
        deterministic_fraction = (
            deterministic_actions.sum(dim=-1)[positive_budget]
            / budgets[positive_budget]
        ).mean()
        if not 0.14 <= float(stochastic_fraction) <= 0.28:
            raise AssertionError(
                f"{name}: initial spend fraction "
                f"{float(stochastic_fraction):.4f} is not aligned with 0.20"
            )

        metrics = agent.update(
            {
                "state": states[:32],
                "action": actions[:32].detach(),
                "reward": torch.randn(32, 1),
                "next_state": next_states[:32],
                "done": torch.zeros(32, 1),
            }
        )
        required = ("q1_loss", "q2_loss", "actor_loss", "temperature")
        if not all(math.isfinite(float(metrics[key])) for key in required):
            raise AssertionError(f"{name}: nonfinite one-update metrics")
        print(
            f"[ok] {name}: max_excess={float(excess.max()):.3e} "
            f"stochastic_spend={float(stochastic_fraction):.4f} "
            f"deterministic_spend={float(deterministic_fraction):.4f}"
        )

    proposal = torch.tensor([[8.0, 4.0, 2.0], [1.0, 2.0, 3.0]])
    budget = torch.tensor([[5.0], [10.0]])
    projected = project_to_budget_simplex(proposal, budget)
    if bool((projected < 0.0).any()):
        raise AssertionError("Projection produced a negative action.")
    if bool((projected.sum(dim=-1, keepdim=True) > budget + 1e-6).any()):
        raise AssertionError("Projection violated the remaining budget.")
    if not torch.allclose(projected[1], proposal[1]):
        raise AssertionError("Projection changed an already feasible proposal.")
    print("[ok] Euclidean budget-simplex projection")

    dgp_cfg = SimConfig(
        M=m,
        Time_intervals=6,
        daily_budget=10.0,
        device="cpu",
        seed=2026,
    )
    coef_generator = torch.Generator(device="cpu")
    coef_generator.manual_seed(2026)
    coefs = generate_dgp_coefs(dgp_cfg, generator=coef_generator)
    env_a = SubsidyAllocationEnv(
        dgp_cfg, coefs=coefs, seed=22026, project_action=False
    )
    env_b = SubsidyAllocationEnv(
        dgp_cfg, coefs=coefs, seed=22026, project_action=False
    )
    env_a.reset()
    env_b.reset()
    if not torch.equal(
        random_paced_action(env_a, m),
        random_paced_action(env_b, m),
    ):
        raise AssertionError(
            "Random pacing is not reproducible under a paired evaluation seed."
        )
    print("[ok] paired random-pacing baseline")


if __name__ == "__main__":
    main()
