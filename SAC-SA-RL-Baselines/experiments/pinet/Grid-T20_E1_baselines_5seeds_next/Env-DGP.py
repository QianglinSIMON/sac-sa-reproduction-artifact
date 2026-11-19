import torch
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from torch.distributions.normal import Normal

# ==============================================================================
# 1. Global Constants
# ==============================================================================
NUMBER_OF_GRID_TILES_X = 9
NUMBER_OF_GRID_TILES_Y = 9
NUMBER_OF_TIME_STEPS = 20
NUMBER_OF_ORDERS = 100
NUMBER_OF_DRIVERS = 25
MAX_MANHATTAN_DISTANCE = 2
DISCOUNT_FACTOR = 0.9
BASE_REWARD_PER_TRIP = 1.0
REWARD_FOR_DISTANCE_PARAMETER = 1.0
V_DIMS = (NUMBER_OF_GRID_TILES_X, NUMBER_OF_GRID_TILES_Y, NUMBER_OF_TIME_STEPS)


@dataclass(frozen=True)
class MicroGridConfig:
    """Configuration for one no-subsidy microscopic city-day simulation."""

    grid_x: int = NUMBER_OF_GRID_TILES_X
    grid_y: int = NUMBER_OF_GRID_TILES_Y
    T: int = NUMBER_OF_TIME_STEPS
    num_orders: int = NUMBER_OF_ORDERS
    num_drivers: int = NUMBER_OF_DRIVERS
    max_manhattan_distance: int = MAX_MANHATTAN_DISTANCE
    base_reward_per_trip: float = BASE_REWARD_PER_TRIP
    reward_for_distance: float = REWARD_FOR_DISTANCE_PARAMETER
    reward_scale: float = 1.0
    demand_center_1: Tuple[float, float, float] = (3.0, 3.0, 2.0)
    demand_center_2: Tuple[float, float, float] = (6.0, 6.0, 14.0)
    demand_std_1: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    demand_std_2: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    driver_mode: str = "uniform"
    driver_center: Tuple[float, float] = (4.0, 4.0)
    driver_std: Tuple[float, float] = (2.0, 2.0)
    wait_mean: float = 1.5
    wait_std: float = 0.5
    wait_max: float = 3.0


def _truncated_normal_with_generator(
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    size: Tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    normal = Normal(
        torch.tensor(0.0, device=mean.device, dtype=mean.dtype),
        torch.tensor(1.0, device=mean.device, dtype=mean.dtype),
    )
    alpha = (lower - mean) / std
    beta = (upper - mean) / std
    cdf_alpha = normal.cdf(alpha)
    cdf_beta = normal.cdf(beta)
    u = torch.rand(size, device=mean.device, dtype=mean.dtype, generator=generator)
    p = (cdf_alpha + u * (cdf_beta - cdf_alpha)).clamp(1e-9, 1.0 - 1e-9)
    return normal.icdf(p) * std + mean


class MicroGridDayEnv:
    """Reusable no-subsidy version of the original microscopic grid process.

    The environment keeps order and driver states at the original cell level.
    ``observe`` exposes only pre-dispatch city-time summaries; ``step`` performs
    the same distance-prioritized greedy one-to-one matching used by the
    original distance policy.
    """

    def __init__(self, config: Optional[MicroGridConfig] = None, seed: int = 0):
        self.cfg = config or MicroGridConfig()
        if self.cfg.T != NUMBER_OF_TIME_STEPS:
            raise ValueError(f"MicroGridDayEnv requires T={NUMBER_OF_TIME_STEPS}.")
        if self.cfg.grid_x != NUMBER_OF_GRID_TILES_X or self.cfg.grid_y != NUMBER_OF_GRID_TILES_Y:
            raise ValueError(
                f"MicroGridDayEnv requires a {NUMBER_OF_GRID_TILES_X}x{NUMBER_OF_GRID_TILES_Y} grid."
            )
        if self.cfg.driver_mode not in {"uniform", "center", "mismatch"}:
            raise ValueError(f"Unknown driver_mode={self.cfg.driver_mode!r}.")
        self.generator = torch.Generator(device="cpu")
        self.seed = int(seed)
        self.t = 0
        self.orders = torch.empty(0)
        self.drivers = torch.empty(0)
        self.reset(seed)

    def _generate_orders(self) -> torch.Tensor:
        cfg = self.cfg
        n = int(cfg.num_orders)
        choices = torch.multinomial(
            torch.tensor([1.0 / 3.0, 2.0 / 3.0]),
            n,
            replacement=True,
            generator=self.generator,
        )
        mean1 = torch.tensor(cfg.demand_center_1, dtype=torch.float32)
        mean2 = torch.tensor(cfg.demand_center_2, dtype=torch.float32)
        std1 = torch.tensor(cfg.demand_std_1, dtype=torch.float32)
        std2 = torch.tensor(cfg.demand_std_2, dtype=torch.float32)
        means = torch.where((choices == 0).unsqueeze(1), mean1, mean2)
        stds = torch.where((choices == 0).unsqueeze(1), std1, std2)
        lower = torch.zeros_like(means)
        upper = torch.tensor(
            [cfg.grid_x - 1, cfg.grid_y - 1, cfg.T - 1], dtype=torch.float32
        ).expand_as(means)
        request = _truncated_normal_with_generator(
            means, stds, lower, upper, (n, 3), self.generator
        ).round()
        wait = _truncated_normal_with_generator(
            torch.tensor(float(cfg.wait_mean)),
            torch.tensor(float(cfg.wait_std)),
            torch.tensor(0.0),
            torch.tensor(float(cfg.wait_max)),
            (n,),
            self.generator,
        ).round()
        destination = torch.stack(
            [
                torch.randint(0, cfg.grid_x, (n,), generator=self.generator),
                torch.randint(0, cfg.grid_y, (n,), generator=self.generator),
            ],
            dim=-1,
        ).to(dtype=torch.float32)
        orders = torch.zeros((n, 7), dtype=torch.float32)
        orders[:, 1:4] = request
        orders[:, 4] = wait
        orders[:, 5:7] = destination
        return orders

    def _generate_drivers(self) -> torch.Tensor:
        cfg = self.cfg
        n = int(cfg.num_drivers)
        drivers = torch.zeros((n, 3), dtype=torch.float32)
        if cfg.driver_mode == "uniform":
            positions = torch.stack(
                [
                    torch.randint(0, cfg.grid_x, (n,), generator=self.generator),
                    torch.randint(0, cfg.grid_y, (n,), generator=self.generator),
                ],
                dim=-1,
            ).to(dtype=torch.float32)
        else:
            center = torch.tensor(cfg.driver_center, dtype=torch.float32).expand(n, 2)
            std = torch.tensor(cfg.driver_std, dtype=torch.float32).expand(n, 2)
            positions = _truncated_normal_with_generator(
                center,
                std,
                torch.zeros((n, 2), dtype=torch.float32),
                torch.tensor([cfg.grid_x - 1, cfg.grid_y - 1], dtype=torch.float32).expand(n, 2),
                (n, 2),
                self.generator,
            ).round()
        drivers[:, 1:3] = positions
        return drivers

    def reset(self, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            self.seed = int(seed)
        self.generator.manual_seed(self.seed)
        self.orders = self._generate_orders()
        self.drivers = self._generate_drivers()
        self.t = 0
        return self.observe()

    def _masks_and_distances(self):
        t = int(self.t)
        active = (
            (self.orders[:, 0] == 0)
            & (self.orders[:, 3] <= t)
            & ((self.orders[:, 3] + self.orders[:, 4]) >= t)
        )
        available = self.drivers[:, 0] <= t
        active_idx = torch.where(active)[0]
        available_idx = torch.where(available)[0]
        if active_idx.numel() and available_idx.numel():
            distances = torch.cdist(
                self.orders[active_idx, 1:3], self.drivers[available_idx, 1:3], p=1
            )
            feasible = distances <= float(self.cfg.max_manhattan_distance)
        else:
            distances = torch.empty((active_idx.numel(), available_idx.numel()))
            feasible = torch.zeros_like(distances, dtype=torch.bool)
        return active, available, active_idx, available_idx, distances, feasible

    def observe(self) -> Dict[str, float]:
        active, available, active_idx, _available_idx, distances, feasible = self._masks_and_distances()
        x = float(active.sum().item())
        y = float(available.sum().item())
        if active_idx.numel():
            remaining = (
                self.orders[active_idx, 3] + self.orders[active_idx, 4] - float(self.t)
            ).clamp_min(0.0)
            mean_remaining_wait = float(remaining.mean().item())
            expiring_orders = float((remaining <= 1.0).sum().item())
        else:
            mean_remaining_wait = 0.0
            expiring_orders = 0.0
        if feasible.numel() and feasible.any():
            candidate_mask = feasible.any(dim=1)
            candidate_orders = float(candidate_mask.sum().item())
            feasible_pair_count = float(feasible.sum().item())
            nearest = torch.where(
                feasible,
                distances,
                torch.full_like(distances, float("inf")),
            ).min(dim=1).values[candidate_mask]
            mean_pickup_distance = float(nearest.mean().item())
            candidate_idx = active_idx[candidate_mask]
            candidate_trip_distance = (
                self.orders[candidate_idx, 1:3] - self.orders[candidate_idx, 5:7]
            ).abs().sum(dim=-1)
            candidate_trip_score = (
                float(self.cfg.base_reward_per_trip)
                + float(self.cfg.reward_for_distance) * candidate_trip_distance
            ) * float(self.cfg.reward_scale)
            mean_candidate_trip_score = float(candidate_trip_score.mean().item())
        else:
            candidate_orders = 0.0
            feasible_pair_count = 0.0
            mean_pickup_distance = float(self.cfg.max_manhattan_distance + 1)
            mean_candidate_trip_score = 0.0
        returning_drivers = float(
            ((self.drivers[:, 0] > float(self.t)) & (self.drivers[:, 0] <= float(self.t + 1))).sum().item()
        )
        return {
            "active_orders": x,
            "available_drivers": y,
            "candidate_orders": candidate_orders,
            "matchability": candidate_orders / max(x, 1.0),
            "mean_remaining_wait": mean_remaining_wait,
            "expiring_orders": expiring_orders,
            "returning_drivers": returning_drivers,
            "mean_pickup_distance": mean_pickup_distance,
            "feasible_pair_count": feasible_pair_count,
            "mean_candidate_trip_score": mean_candidate_trip_score,
        }

    def step(self) -> Dict[str, float]:
        if self.t >= self.cfg.T:
            raise RuntimeError("MicroGridDayEnv episode has already terminated.")
        observation = self.observe()
        _active, _available, active_idx, available_idx, distances, feasible = self._masks_and_distances()
        completed = 0
        gross_response = 0.0
        if feasible.numel() and feasible.any():
            affinity = torch.where(
                feasible,
                torch.exp(-distances / 0.1),
                torch.zeros_like(distances),
            ).unsqueeze(0)
            matched_rows, matched_cols = greedy_decode_from_sinkhorn_batched(affinity)
            valid_pairs = matched_rows[0] >= 0
            if valid_pairs.any():
                order_idx = active_idx[matched_rows[0, valid_pairs]]
                driver_idx = available_idx[matched_cols[0, valid_pairs]]
                trip_distance = (
                    self.orders[order_idx, 1:3] - self.orders[order_idx, 5:7]
                ).abs().sum(dim=-1)
                pickup_distance = (
                    self.orders[order_idx, 1:3] - self.drivers[driver_idx, 1:3]
                ).abs().sum(dim=-1)
                trip_reward = (
                    float(self.cfg.base_reward_per_trip)
                    + float(self.cfg.reward_for_distance) * trip_distance
                ) * float(self.cfg.reward_scale)
                gross_response = float(trip_reward.sum().item())
                completed = int(order_idx.numel())
                self.orders[order_idx, 0] = 1.0
                self.drivers[driver_idx, 0] = (
                    float(self.t) + 1.0 + pickup_distance + trip_distance
                )
                self.drivers[driver_idx, 1:3] = self.orders[order_idx, 5:7]
        result = {
            **observation,
            "gross_response": gross_response,
            "completed_orders": float(completed),
            "t": float(self.t),
        }
        self.t += 1
        return result


def simulate_micro_city_day(config: MicroGridConfig, seed: int) -> Dict[str, torch.Tensor]:
    """Simulate one city-day and return pre-dispatch city-time aggregates."""

    env = MicroGridDayEnv(config, seed=seed)
    keys = (
        "active_orders",
        "available_drivers",
        "matchability",
        "mean_remaining_wait",
        "expiring_orders",
        "returning_drivers",
        "mean_pickup_distance",
        "feasible_pair_count",
        "mean_candidate_trip_score",
        "gross_response",
        "completed_orders",
    )
    rows = {key: torch.zeros(config.T, dtype=torch.float32) for key in keys}
    for t in range(config.T):
        result = env.step()
        for key in keys:
            rows[key][t] = float(result[key])
    return rows

# ==============================================================================
# 2. GPU-based Data Generators
# ==============================================================================
def truncated_normal_rvs_pt(mean, std, lower_bound, upper_bound, size):
    device = mean.device
    normal_dist = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    alpha = (lower_bound - mean) / std
    beta = (upper_bound - mean) / std
    cdf_alpha = normal_dist.cdf(alpha)
    cdf_beta = normal_dist.cdf(beta)
    p = cdf_alpha + torch.rand(size, device=device) * (cdf_beta - cdf_alpha)
    p = p.clamp(1e-9, 1 - 1e-9)
    x_std_truncated = normal_dist.icdf(p)
    return x_std_truncated * std + mean

def generate_initial_orders_from_mixture_pt(BATCH_SIZE, NUMBER_OF_ORDERS, device):
    total_samples = BATCH_SIZE * NUMBER_OF_ORDERS
    weights = torch.tensor([1./3, 2./3], device=device)
    choices = torch.multinomial(weights, total_samples, replacement=True)
    
    mean1 = torch.tensor([3.0, 3.0, 2.0], device=device)
    std1 = torch.tensor([2.0, 2.0, 2.0], device=device)
    mean2 = torch.tensor([6.0, 6.0, 14.0], device=device)
    std2 = torch.tensor([2.0, 2.0, 2.0], device=device)
    lower = torch.tensor([0.0, 0.0, 0.0], device=device)
    upper = torch.tensor([8.0, 8.0, 19.0], device=device)

    means = torch.where((choices == 0).unsqueeze(1), mean1, mean2)
    stds = torch.where((choices == 0).unsqueeze(1), std1, std2)

    generated_data = truncated_normal_rvs_pt(
        means, stds, lower.expand_as(means), upper.expand_as(means), size=(total_samples, 3)
    )
    return generated_data.round().long().view(BATCH_SIZE, NUMBER_OF_ORDERS, 3)

def generate_initial_waiting_times_pt(BATCH_SIZE, NUMBER_OF_ORDERS, device):
    total_samples = BATCH_SIZE * NUMBER_OF_ORDERS
    mean_t = torch.tensor(1.5, device=device)
    std_t = torch.tensor(0.5, device=device)
    lower_t = torch.tensor(0.0, device=device)
    upper_t = torch.tensor(3.0, device=device)
    
    waiting_times_float = truncated_normal_rvs_pt(
        mean_t, std_t, lower_t, upper_t, size=(total_samples,)
    )
    return waiting_times_float.round().long().view(BATCH_SIZE, NUMBER_OF_ORDERS)

# ==============================================================================
# 3. Core Matching Algorithm
# ==============================================================================
def greedy_decode_from_sinkhorn_batched(T):
    """
    Greedy decoding from Sinkhorn output on GPU.
    """
    b, n, m = T.shape
    k = min(n, m)
    
    matched_rows = torch.full((b, k), -1, dtype=torch.long, device=T.device)
    matched_cols = torch.full((b, k), -1, dtype=torch.long, device=T.device)
    
    T_copy = T.clone()

    for i in range(k):
        if T_copy.view(b, -1).shape[1] == 0:
            break
        
        max_probs, flat_indices = torch.max(T_copy.view(b, -1), dim=1)
        
        active_mask = max_probs > 1e-9
        
        if not active_mask.any():
            break
            
        row_indices_active = flat_indices[active_mask] // m
        col_indices_active = flat_indices[active_mask] % m
        
        matched_rows[active_mask, i] = row_indices_active
        matched_cols[active_mask, i] = col_indices_active
        
        T_copy[active_mask, row_indices_active, :] = -1.0
        T_copy[active_mask, :, col_indices_active] = -1.0

    return matched_rows, matched_cols

# ==============================================================================
# 4. Policy Evaluation (V-function Learning) GPU Version
# ==============================================================================
def linearize_state(states_tensor):
    return states_tensor[..., 0].long() + \
           states_tensor[..., 1].long() * NUMBER_OF_GRID_TILES_X + \
           states_tensor[..., 2].long() * NUMBER_OF_GRID_TILES_X * NUMBER_OF_GRID_TILES_Y

def policy_evaluation_gpu(transactions, V=None, N=None, device='cuda'):
    states, actions, rewards, next_states = transactions
    
    if V is None:
        V = torch.zeros(V_DIMS, device=device, dtype=torch.float32)
    if N is None:
        N = torch.zeros(V_DIMS, device=device, dtype=torch.float32)

    V_flat_lookup = V.view(-1)
    def v_lookup_gpu(s_tensor):
        lin_s = linearize_state(s_tensor)
        return V_flat_lookup[lin_s]

    print("Policy evaluation on GPU...")
    for t in range(NUMBER_OF_TIME_STEPS - 1, -1, -1):
        mask_t = (states[:, 2] == t)
        if mask_t.sum() == 0:
            continue

        s_t, a_t, r_t, ns_t = states[mask_t], actions[mask_t], rewards[mask_t], next_states[mask_t]

        lin_s_t = linearize_state(s_t)
        ones = torch.ones_like(lin_s_t, dtype=torch.float32)
        N.view(-1).scatter_add_(0, lin_s_t, ones)
        
        delta_t = (ns_t[:, 2] - s_t[:, 2]).float().clamp_min(1.0)
        future_value = torch.pow(torch.tensor(DISCOUNT_FACTOR, device=device), delta_t) * v_lookup_gpu(ns_t)
        
        target = r_t + future_value
        
        old_v = v_lookup_gpu(s_t)
        td_error = target - old_v
        current_N = N.view(-1)[lin_s_t].clamp_min(1.0)
        update_amount = td_error / current_N
        
        V.view(-1).scatter_add_(0, lin_s_t, update_amount)

    print("Policy evaluation finished.")
    return V, N

# ==============================================================================
# 5. Main Simulator GPU Version
# ==============================================================================
def run_simulation_gpu(
    BATCH_SIZE, NUM_EPISODES, V_tensor, device, method, return_transitions=False
):
    print(f"Running simulation: BATCH_SIZE={BATCH_SIZE}, NUM_EPISODES={NUM_EPISODES}, METHOD={method.upper()}...")
    B, E = BATCH_SIZE, NUM_EPISODES
    X, Y, T = NUMBER_OF_GRID_TILES_X, NUMBER_OF_GRID_TILES_Y, NUMBER_OF_TIME_STEPS
    BE = B * E
    
    V_be = V_tensor.to(device=device, dtype=torch.float32)
    V_be = V_be.unsqueeze(0).unsqueeze(0).expand(B, E, -1, -1, -1).contiguous()
    V_flat_lookup = V_be.view(B, E, -1)
    def v_lookup(xs, ys, ts):
        lin = xs.long() + X * ys.long() + (X * Y) * ts.long()
        return V_flat_lookup.gather(-1, lin.view(B, E, -1)).view_as(xs)

    orders = torch.zeros(B, E, NUMBER_OF_ORDERS, 7, device=device, dtype=torch.float32)
    st_data = generate_initial_orders_from_mixture_pt(BE, NUMBER_OF_ORDERS, device).view(B, E, NUMBER_OF_ORDERS, 3).float()
    wait_data = generate_initial_waiting_times_pt(BE, NUMBER_OF_ORDERS, device).view(B, E, NUMBER_OF_ORDERS).float()
    dest_data = torch.stack([
        torch.randint(0, X, size=(B, E, NUMBER_OF_ORDERS), device=device),
        torch.randint(0, Y, size=(B, E, NUMBER_OF_ORDERS), device=device)
    ], dim=-1).float()
    orders[..., 1:4] = st_data; orders[..., 4] = wait_data; orders[..., 5:7] = dest_data

    drivers = torch.zeros(B, E, NUMBER_OF_DRIVERS, 3, device=device, dtype=torch.float32)
    drivers[..., 1] = torch.randint(0, X, size=drivers[..., 1].shape, device=device).float()
    drivers[..., 2] = torch.randint(0, Y, size=drivers[..., 2].shape, device=device).float()

    b_grid, e_grid = torch.meshgrid(torch.arange(B, device=device), torch.arange(E, device=device), indexing='ij')
    all_rows_list = []
    
    if return_transitions:
        s_list, a_list, r_list, ns_list = [], [], [], []

    for t in range(T):
        active_orders_mask = (orders[..., 0] == 0) & (orders[..., 3] <= t) & ((orders[..., 3] + orders[..., 4]) >= t)
        available_drv_mask = drivers[..., 0] <= t
        
        o_pos = orders[..., 1:3]; d_pos = drivers[..., 1:3]; dstpos = orders[..., 5:7]
        dist_mat = torch.cdist(o_pos.reshape(BE, NUMBER_OF_ORDERS, 2), d_pos.reshape(BE, NUMBER_OF_DRIVERS, 2), p=1).view(B, E, NUMBER_OF_ORDERS, NUMBER_OF_DRIVERS)
        allowed = dist_mat <= MAX_MANHATTAN_DISTANCE
        
        valid_mask = active_orders_mask.unsqueeze(-1) & available_drv_mask.unsqueeze(-2) & allowed
        
        trip_dist = (o_pos - dstpos).abs().sum(dim=-1)
        delta_t = (1 + dist_mat + trip_dist.unsqueeze(-1)).long()
        reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * trip_dist
        
        curV = v_lookup(d_pos[..., 0], d_pos[..., 1], torch.full_like(d_pos[..., 0], t)).unsqueeze(-2)
        fut_t = (t + delta_t).clamp_max(T - 1)
        dst_x_expanded = dstpos[..., 0].unsqueeze(-1).expand_as(fut_t)
        dst_y_expanded = dstpos[..., 1].unsqueeze(-1).expand_as(fut_t)
        futV  = v_lookup(dst_x_expanded, dst_y_expanded, fut_t)
        
        delta_t_f = delta_t.clamp_min(1).float()
        gamma = torch.tensor(DISCOUNT_FACTOR, device=device, dtype=delta_t_f.dtype)
        disc_dt = torch.pow(gamma, delta_t_f)
        imm = (1 - disc_dt).clamp_min(1e-9) / (1 - gamma).clamp_min(1e-9) * reward.unsqueeze(-1).to(delta_t_f.dtype) / delta_t_f
        adv_mdp = imm + disc_dt * futV - curV
        
        adv_mdp_masked = torch.where(valid_mask, adv_mdp, -1e5)
        adv_dist_masked = torch.where(valid_mask, -dist_mat.float(), -1e5)
        
        if method == 'mdp': adv = adv_mdp_masked
        else: adv = adv_dist_masked
        policy_is_mdp = (method == 'mdp')

        K = torch.exp(torch.clamp(adv, max=50) / 0.1)
        K = torch.where(valid_mask, K, torch.zeros_like(K))
        T_flat = K.view(BE, NUMBER_OF_ORDERS, NUMBER_OF_DRIVERS)
        mr, mc = greedy_decode_from_sinkhorn_batched(T_flat)
        
        valid_pairs_mask = mr >= 0
        num_matches = valid_pairs_mask.sum()

        matched_revenue_per_be = torch.zeros(BE, device=device, dtype=torch.float32)
        
        b_idx, e_idx, r_sel, c_sel = [torch.tensor([], dtype=torch.long, device=device)] * 4
        revenues = torch.tensor([], dtype=torch.float32, device=device)

        if num_matches > 0:
            be_idx_flat = torch.arange(BE, device=device).unsqueeze(1).expand_as(mr)
            be_sel = be_idx_flat[valid_pairs_mask]
            r_sel = mr[valid_pairs_mask]; c_sel = mc[valid_pairs_mask]
            b_idx = be_sel // E; e_idx = be_sel % E
            
            trip = (orders[b_idx, e_idx, r_sel, 1:3] - orders[b_idx, e_idx, r_sel, 5:7]).abs().sum(dim=-1).long()
            revenues = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * trip.float()
            matched_revenue_per_be.scatter_add_(0, be_sel, revenues)
            
            orders[b_idx, e_idx, r_sel, 0] = 1.0
            pickup = (orders[b_idx, e_idx, r_sel, 1:3] - drivers[b_idx, e_idx, c_sel, 1:3]).abs().sum(dim=-1).long()
            dt = (1 + pickup + trip).long()
            drivers[b_idx, e_idx, c_sel, 0] = t + dt.float()
            drivers[b_idx, e_idx, c_sel, 1:3] = orders[b_idx, e_idx, r_sel, 5:7]

        if return_transitions:
            s_x = drivers[..., 1].long(); s_y = drivers[..., 2].long()
            s_t = torch.full_like(s_x, t)
            states = torch.stack([s_x, s_y, s_t], dim=-1)
            
            current_rewards = torch.zeros_like(s_x, dtype=torch.float32)
            actions = torch.zeros_like(s_x)
            if num_matches > 0: 
                current_rewards[b_idx, e_idx, c_sel] = revenues
                actions[b_idx, e_idx, c_sel] = 1
                
            next_states = torch.stack([
                drivers[..., 1].long(), drivers[..., 2].long(), drivers[..., 0].long().clamp_max(T - 1)
            ], dim=-1)

            s_list.append(states[available_drv_mask]); a_list.append(actions[available_drv_mask])
            r_list.append(current_rewards[available_drv_mask]); ns_list.append(next_states[available_drv_mask])

        orders_now_cnt = active_orders_mask.sum(dim=-1)
        drivers_now_cnt = available_drv_mask.sum(dim=-1)
        
        t_next = t + 1
        if t_next < T:
            orders_next_cnt = ((orders[..., 0] == 0) & (orders[..., 3] <= t_next) & ((orders[..., 3] + orders[..., 4]) >= t_next)).sum(dim=-1)
            drivers_next_cnt = (drivers[..., 0] <= t_next).sum(dim=-1)
        else:
            orders_next_cnt = torch.zeros_like(orders_now_cnt)
            drivers_next_cnt = torch.zeros_like(drivers_now_cnt)

        rows = torch.stack([
            (b_grid + 1).reshape(-1), e_grid.reshape(-1), torch.full((BE,), t, device=device),
            orders_now_cnt.reshape(-1), drivers_now_cnt.reshape(-1),
            torch.full((BE,), float(policy_is_mdp), device=device),
            matched_revenue_per_be.reshape(-1),
            orders_next_cnt.reshape(-1), drivers_next_cnt.reshape(-1),
        ], dim=1)
        all_rows_list.append(rows)

    all_rows = torch.cat(all_rows_list, dim=0)
    df = pd.DataFrame(all_rows.cpu().numpy(), columns=[
        'simu_time', 'n', 'T', 'orders', 'drivers', 'A', 'revenue', 'ordersNext', 'driversNext'
    ])
    df_sorted = df.sort_values(['simu_time', 'n', 'T']).reset_index(drop=True)

    if return_transitions:
        all_states = torch.cat(s_list, dim=0); all_actions = torch.cat(a_list, dim=0)
        all_rewards = torch.cat(r_list, dim=0); all_next_states = torch.cat(ns_list, dim=0)
        return df_sorted, (all_states, all_actions, all_rewards, all_next_states)
    else:
        return df_sorted


# ==============================================================================
# 6. Main Execution Script
# ==============================================================================
if __name__ == '__main__':
    start_main = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    NUM_EXPLORE_EPISODES = 2000
    NUM_BENCHMARK_EPISODES = 1500
    BATCH_SIZE = 20

    initial_V = torch.zeros(V_DIMS, device=device)

    # Phase 1: Exploration with distance strategy
    print("\n" + "="*50)
    print("--- Phase 1: Exploration and Data Collection (distance strategy) ---")
    print("="*50)
    _, collected_transactions = run_simulation_gpu(
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPISODES=NUM_EXPLORE_EPISODES,
        V_tensor=initial_V,
        device=device,
        method="distance",
        return_transitions=True
    )
    
    # Phase 2: Policy evaluation and V-function learning
    print("\n" + "="*50)
    print("--- Phase 2: Policy Evaluation and V-function Learning ---")
    print("="*50)
    learned_V, N = policy_evaluation_gpu(
        transactions=collected_transactions,
        V=initial_V.clone(),
        device=device
    )

    # Phase 3: Benchmark with learned V-function
    print("\n" + "="*50)
    print("--- Phase 3: Benchmark with Learned V-function ---")
    print("="*50)
    
    # Run MDP
    df_mdp_benchmark = run_simulation_gpu(
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPISODES=NUM_BENCHMARK_EPISODES,
        V_tensor=learned_V,
        device=device,
        method="mdp",
        return_transitions=False
    )

    # Run Distance baseline
    df_dist_benchmark = run_simulation_gpu(
        BATCH_SIZE=BATCH_SIZE,
        NUM_EPISODES=NUM_BENCHMARK_EPISODES,
        V_tensor=initial_V,
        device=device,
        method="distance",
        return_transitions=False
    )
    print(learned_V)

    output_filename = f"Value_function_gpu_learned_T{NUMBER_OF_TIME_STEPS}.npz"

    learned_V_cpu = learned_V.cpu()
    learned_V_numpy = learned_V_cpu.numpy()
    np.savez(output_filename, learned_V_numpy)

    # Phase 4: Analysis and reporting
    avg_revenue_mdp = df_mdp_benchmark['revenue'].mean()
    avg_revenue_dist = df_dist_benchmark['revenue'].mean()
    ate = avg_revenue_mdp - avg_revenue_dist
    
    end_main = time.time()
    total_time = end_main - start_main

    print("\n\n" + "="*50)
    print("--- Final Benchmark Results ---")
    print("="*50)
    print(f"Configuration: BATCH_SIZE={BATCH_SIZE}, EXPLORE_EPISODES={NUM_EXPLORE_EPISODES}, BENCHMARK_EPISODES={NUM_BENCHMARK_EPISODES}")
    print("-" * 50)
    print(f"Average revenue per step (MDP): {avg_revenue_mdp:.4f}")
    print(f"Average revenue per step (Distance): {avg_revenue_dist:.4f}")
    print(f"ATE (MDP - Distance): {ate:.4f}")
    print("-" * 50)
    print(f"Total elapsed time: {total_time:.2f} seconds")
