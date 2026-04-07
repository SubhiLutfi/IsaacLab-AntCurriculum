# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete MDP functions for Ant obstacle avoidance project.

All observation and reward functions for the entire curriculum are defined here.
Functions not yet active in early stages return dummy values to maintain
a fixed observation dimension across all training stages.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# OBSERVATION FUNCTIONS — Proprioception helpers
# =============================================================================
# (These are handled by the built-in mdp module, listed here for reference)


# =============================================================================
# OBSERVATION FUNCTIONS — Navigation (active from Stage 4)
# =============================================================================

def waypoint_angle_dummy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dummy waypoint angle observation. Returns 0.0 (facing target).
    Used in Stages 0-3 before waypoint system is active.
    """
    return torch.zeros((env.num_envs, 1), device=env.device)


def waypoint_heading_proj_dummy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dummy waypoint heading projection. Returns 1.0 (perfectly aligned).
    Used in Stages 0-3 before waypoint system is active.
    """
    return torch.ones((env.num_envs, 1), device=env.device)


def waypoint_distance_dummy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dummy waypoint distance. Returns 1.0 (far from target).
    Used in Stages 0-3 before waypoint system is active.
    """
    return torch.ones((env.num_envs, 1), device=env.device)


# =============================================================================
# RANDOM TARGET STATE — Stage 4.5
# =============================================================================

def _respawn_targets(env, env_ids, asset_cfg, target_range: float = 5.0):
    """Spawn new random targets for given envs.

    If env._rand_target_obstacle_names is set (via set_obstacle_biased_targets startup event),
    targets are biased toward obstacle-dense zones: N candidates are sampled and the one
    nearest to the most obstacles wins.
    """
    n = len(env_ids)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    ant_xy = pos_local[:, :2]

    obstacle_names = getattr(env, "_rand_target_obstacle_names", ())

    if obstacle_names:
        # Sample candidates biased toward obstacle-dense zones, but NOT too close to any obstacle.
        # nearby_radius: target counts as "near" an obstacle if within this distance (for scoring)
        # min_target_obs_dist: target must be at least this far from every obstacle surface
        n_cand = 16
        nearby_radius = 6.0       # score candidates near obstacles
        min_target_obs_dist = 2.5  # but keep target reachable (not right next to one)

        angles = torch.rand(n, n_cand, device=env.device) * 2 * math.pi
        dists = 2.0 + torch.rand(n, n_cand, device=env.device) * (target_range - 2.0)
        cand_xy = ant_xy.unsqueeze(1) + torch.stack(
            [torch.cos(angles) * dists, torch.sin(angles) * dists], dim=-1
        )  # (n, n_cand, 2)

        scores = torch.zeros(n, n_cand, device=env.device)
        too_close = torch.zeros(n, n_cand, dtype=torch.bool, device=env.device)

        for obs_name in obstacle_names:
            obs_xy = (
                env.scene[obs_name].data.root_pos_w[env_ids, :2]
                - env.scene.env_origins[env_ids, :2]
            )  # (n, 2)
            dist_sq = ((cand_xy - obs_xy.unsqueeze(1)) ** 2).sum(dim=-1)  # (n, n_cand)
            scores += (dist_sq < nearby_radius ** 2).float()
            too_close |= (dist_sq < min_target_obs_dist ** 2)

        # Disqualify candidates too close to any obstacle
        scores[too_close] = -1.0

        # Softmax sampling for variety (not always same cluster)
        # Fall back to uniform if all candidates are disqualified
        temp = 1.5
        scores_exp = torch.exp(scores * temp)
        scores_exp[too_close] = 0.0
        row_sum = scores_exp.sum(dim=1, keepdim=True)
        # Where all candidates disqualified, use uniform over all candidates
        all_bad = (row_sum.squeeze(1) < 1e-6)
        scores_exp[all_bad] = 1.0
        row_sum = scores_exp.sum(dim=1, keepdim=True).clamp(min=1e-6)
        probs = scores_exp / row_sum
        best = torch.multinomial(probs, num_samples=1).squeeze(1)  # (n,)

        target_xy = cand_xy[torch.arange(n, device=env.device), best]  # (n, 2)
        chosen_dist = torch.norm(target_xy - ant_xy, dim=-1)
    else:
        angle = torch.rand(n, device=env.device) * 2 * math.pi
        chosen_dist = 2.0 + torch.rand(n, device=env.device) * (target_range - 2.0)
        target_xy = ant_xy + torch.stack(
            [torch.cos(angle) * chosen_dist, torch.sin(angle) * chosen_dist], dim=-1
        )

    env._rand_target[env_ids] = target_xy

    if hasattr(env, "_target_marker_body") and env._target_marker_body is not None:
        marker = env._target_marker_body
        marker_state = marker.data.default_root_state[env_ids].clone()
        marker_state[:, 0] = target_xy[:, 0] + env.scene.env_origins[env_ids, 0]
        marker_state[:, 1] = target_xy[:, 1] + env.scene.env_origins[env_ids, 1]
        marker_state[:, 2] = 0.5 + env.scene.env_origins[env_ids, 2]
        marker_state[:, 7:] = 0.0
        marker.write_root_state_to_sim(marker_state, env_ids=env_ids)

    env._rand_target_prev_dist[env_ids] = chosen_dist


def set_obstacle_biased_targets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: tuple[str, ...] = (),
) -> None:
    """Startup event: register obstacle names so target respawning is biased toward them."""
    env._rand_target_obstacle_names = obstacle_names


def _ensure_random_target_state(env: ManagerBasedRLEnv, asset_cfg, target_range: float = 5.0, reach_radius: float = 0.8):
    """Initialize or reset random target points per env."""
    device = env.device
    n = env.num_envs

    if not hasattr(env, "_rand_target"):
        env._rand_target = torch.zeros(n, 2, device=device)
        env._rand_target_prev_dist = torch.full((n,), 5.0, device=device)
        env._rand_targets_reached = torch.zeros(n, device=device)
        # Try to grab visual marker from scene
        if not hasattr(env, "_target_marker_body"):
            try:
                env._target_marker_body = env.scene["target_marker"]
            except KeyError:
                env._target_marker_body = None
        _respawn_targets(env, torch.arange(n, device=device), asset_cfg, target_range)

    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
        env._rand_targets_reached[reset_ids] = 0.0
        _respawn_targets(env, reset_ids, asset_cfg, target_range)

    return env._rand_target, env._rand_target_prev_dist, env._rand_targets_reached


def random_target_angle(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
) -> torch.Tensor:
    """Angle to random target, normalized to [-1, 1]. Shape (N, 1)."""
    target, _, _ = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    direction = target - xy
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    angle_diff = target_angle - yaw
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    return (angle_diff / math.pi).unsqueeze(-1)


def random_target_heading_proj(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
) -> torch.Tensor:
    """Cosine of angle to random target. Shape (N, 1). 1.0 = facing target."""
    target, _, _ = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    direction = target - xy
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    angle_diff = target_angle - yaw
    return torch.cos(angle_diff).unsqueeze(-1)


def random_target_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
    max_dist: float = 10.0,
) -> torch.Tensor:
    """Normalized distance to random target. Shape (N, 1)."""
    target, _, _ = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    dist = torch.norm(xy - target, dim=-1)
    return torch.clamp(dist / max_dist, 0.0, 1.0).unsqueeze(-1)


def random_target_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
) -> torch.Tensor:
    """Reward for reducing distance to random target. Respawns target when reached."""
    target, prev_dist, targets_reached = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    current_dist = torch.norm(xy - target, dim=-1)

    reached = current_dist < reach_radius
    if reached.any():
        reached_ids = reached.nonzero(as_tuple=False).squeeze(-1)
        env._rand_targets_reached[reached_ids] += 1.0
        _respawn_targets(env, reached_ids, asset_cfg, target_range)
        new_dist = torch.norm(xy[reached_ids] - env._rand_target[reached_ids], dim=-1)
        current_dist = current_dist.clone()
        current_dist[reached_ids] = new_dist

    reward = prev_dist - current_dist
    env._rand_target_prev_dist = current_dist.detach()
    return torch.clamp(reward, -0.5, 0.5)


def random_target_speed_toward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
    speed_cap: float = 0.5,
) -> torch.Tensor:
    """Velocity component toward random target, clamped to [0, speed_cap]."""
    target, _, _ = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]

    delta = target - xy
    dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
    direction = delta / dist
    speed_proj = (vel_xy * direction).sum(dim=-1)
    return torch.clamp(speed_proj, 0.0, speed_cap)


def random_target_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
) -> torch.Tensor:
    """Reward facing toward random target. Returns clamp(cos(angle), 0, 1)."""
    target, _, _ = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    direction = target - xy
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    angle_diff = target_angle - yaw
    return torch.clamp(torch.cos(angle_diff), 0.0, 1.0)


def random_targets_reached_count(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_range: float = 5.0,
    reach_radius: float = 0.8,
) -> torch.Tensor:
    """Number of targets reached this episode."""
    _, _, targets_reached = _ensure_random_target_state(env, asset_cfg, target_range, reach_radius)
    return targets_reached


def _ensure_waypoint_state(env: ManagerBasedRLEnv, waypoints, asset_cfg):
    """Initialize or reset per-env waypoint tracking state."""
    device = env.device
    n = env.num_envs
    wp = torch.tensor(waypoints, dtype=torch.float32, device=device)

    if not hasattr(env, "_wp_points"):
        env._wp_points = wp
        env._wp_idx = torch.zeros(n, dtype=torch.long, device=device)
        env._wp_prev_dist = torch.full((n,), 10.0, device=device)

    # Reset envs at episode start
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._wp_idx[reset_mask] = 0
        asset = env.scene[asset_cfg.name]
        pos_local = asset.data.root_pos_w - env.scene.env_origins
        xy = pos_local[:, :2]
        target_xy = env._wp_points[0, :2].unsqueeze(0).expand(n, -1)
        dist = torch.norm(xy - target_xy, dim=-1)
        env._wp_prev_dist[reset_mask] = dist[reset_mask]

    return env._wp_points, env._wp_idx, env._wp_prev_dist


def waypoint_angle_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
) -> torch.Tensor:
    """Angle between robot heading and direction to current waypoint.
    Returns shape (N, 1), range [-1, 1] where 0 = facing target.
    """
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    # Get current waypoint per env
    target_xy = wp[idx, :2]  # (N, 2)
    direction = target_xy - xy  # (N, 2)
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])  # (N,)

    # Get robot yaw
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]

    # Angle difference, normalized to [-pi, pi] then to [-1, 1]
    angle_diff = target_angle - yaw
    angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
    return (angle_diff / math.pi).unsqueeze(-1)


def waypoint_heading_proj(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
) -> torch.Tensor:
    """Cosine of angle between robot forward and direction to current waypoint.
    Returns shape (N, 1), range [-1, 1] where 1 = perfectly aligned.
    """
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    target_xy = wp[idx, :2]
    direction = target_xy - xy
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    angle_diff = target_angle - yaw
    return torch.cos(angle_diff).unsqueeze(-1)


def waypoint_distance_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
    max_dist: float = 10.0,
) -> torch.Tensor:
    """Normalized distance to current waypoint. Shape (N, 1), range [0, 1]."""
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    target_xy = wp[idx, :2]
    dist = torch.norm(xy - target_xy, dim=-1)
    return torch.clamp(dist / max_dist, 0.0, 1.0).unsqueeze(-1)


# =============================================================================
# OBSERVATION FUNCTIONS — Obstacle Scan (active from Stage 2)
# =============================================================================

def obstacle_scan_dummy(
    env: ManagerBasedRLEnv,
    num_rays: int = 5,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Placeholder scan returning max normalized range (1.0) for all rays.
    Used in Stages 0-1 before real raycasting is active.
    Shape: (N, num_rays).
    """
    return torch.ones((env.num_envs, num_rays), device=env.device)


# --- Wall AABB helpers for raycasting ---

_obs_wall_cache: torch.Tensor | None = None
_obs_wall_cache_margin: float | None = None


def _get_wall_aabbs(env: ManagerBasedRLEnv, margin: float = 0.35) -> torch.Tensor | None:
    """Extract AABBs for all kinematic rigid bodies in env_0 (local env coordinates)."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    env0_path = env.scene.env_prim_paths[0]
    root = stage.GetPrimAtPath(env0_path)
    if not root.IsValid():
        return None

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    env_origin = env.scene.env_origins[0].detach().cpu()
    aabbs = []

    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        kin_attr = rb.GetKinematicEnabledAttr()
        if not kin_attr or not kin_attr.Get():
            continue
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.GetBox()
        min_pt = rng.GetMin()
        max_pt = rng.GetMax()
        aabbs.append([
            float(min_pt[0]) - float(env_origin[0]) - margin,
            float(max_pt[0]) - float(env_origin[0]) + margin,
            float(min_pt[1]) - float(env_origin[1]) - margin,
            float(max_pt[1]) - float(env_origin[1]) + margin,
        ])

    return torch.tensor(aabbs, device=env.device) if aabbs else None


def _get_obs_wall_aabbs(env: ManagerBasedRLEnv, wall_margin: float = 0.35) -> torch.Tensor | None:
    global _obs_wall_cache, _obs_wall_cache_margin
    if _obs_wall_cache is None or _obs_wall_cache_margin != wall_margin:
        _obs_wall_cache = _get_wall_aabbs(env, margin=wall_margin)
        _obs_wall_cache_margin = wall_margin
    return _obs_wall_cache


def _ray_aabb_distance_2d(
    ray_origins_xy: torch.Tensor,
    ray_dirs_xy: torch.Tensor,
    walls: torch.Tensor | None,
    max_range: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Nearest hit distance from 2D rays to axis-aligned wall boxes."""
    device = ray_origins_xy.device
    dtype = ray_origins_xy.dtype
    num_envs = ray_origins_xy.shape[0]

    if walls is None or walls.numel() == 0:
        return torch.full((num_envs,), max_range, device=device, dtype=dtype)

    num_walls = walls.shape[0]
    ox = ray_origins_xy[:, 0:1]
    oy = ray_origins_xy[:, 1:2]
    dx = ray_dirs_xy[:, 0:1]
    dy = ray_dirs_xy[:, 1:2]

    wx_min = walls[:, 0].view(1, num_walls)
    wx_max = walls[:, 1].view(1, num_walls)
    wy_min = walls[:, 2].view(1, num_walls)
    wy_max = walls[:, 3].view(1, num_walls)

    neg_inf = torch.full((num_envs, num_walls), -torch.inf, device=device, dtype=dtype)
    pos_inf = torch.full((num_envs, num_walls), torch.inf, device=device, dtype=dtype)

    parallel_x = dx.abs() < eps
    dx_safe = torch.where(parallel_x, torch.ones_like(dx), dx)
    tx1 = (wx_min - ox) / dx_safe
    tx2 = (wx_max - ox) / dx_safe
    tx_min = torch.minimum(tx1, tx2)
    tx_max = torch.maximum(tx1, tx2)
    inside_x = (ox >= wx_min) & (ox <= wx_max)
    tx_min = torch.where(parallel_x.expand(-1, num_walls), torch.where(inside_x, neg_inf, pos_inf), tx_min)
    tx_max = torch.where(parallel_x.expand(-1, num_walls), torch.where(inside_x, pos_inf, neg_inf), tx_max)

    parallel_y = dy.abs() < eps
    dy_safe = torch.where(parallel_y, torch.ones_like(dy), dy)
    ty1 = (wy_min - oy) / dy_safe
    ty2 = (wy_max - oy) / dy_safe
    ty_min = torch.minimum(ty1, ty2)
    ty_max = torch.maximum(ty1, ty2)
    inside_y = (oy >= wy_min) & (oy <= wy_max)
    ty_min = torch.where(parallel_y.expand(-1, num_walls), torch.where(inside_y, neg_inf, pos_inf), ty_min)
    ty_max = torch.where(parallel_y.expand(-1, num_walls), torch.where(inside_y, pos_inf, neg_inf), ty_max)

    t_enter = torch.maximum(tx_min, ty_min)
    t_exit = torch.minimum(tx_max, ty_max)
    valid = t_exit >= torch.clamp(t_enter, min=0.0)
    hit_dist = torch.where(t_enter > 0.0, t_enter, t_exit)
    hit_dist = torch.where(valid, hit_dist, torch.full_like(hit_dist, max_range + 1.0))

    nearest = hit_dist.min(dim=1).values
    return torch.clamp(nearest, 0.0, max_range)


def obstacle_scan_5(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
) -> torch.Tensor:
    """5-direction obstacle scan in robot yaw frame, normalized to [0, 1].
    0.0 = wall very close, 1.0 = no wall within max_range.
    Shape: (N, 5).
    """
    asset = env.scene[asset_cfg.name]
    num_envs = env.num_envs

    root_pos_local = asset.data.root_pos_w - env.scene.env_origins
    root_xy = root_pos_local[:, :2]

    root_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
    ray_angles = torch.tensor(ray_angles_deg, device=env.device, dtype=root_xy.dtype) * (math.pi / 180.0)

    local_dirs = torch.stack(
        [torch.cos(ray_angles), torch.sin(ray_angles), torch.zeros_like(ray_angles)],
        dim=-1,
    )

    num_rays = local_dirs.shape[0]
    local_dirs = local_dirs.unsqueeze(0).repeat(num_envs, 1, 1)
    yaw_rep = root_yaw_quat.unsqueeze(1).repeat(1, num_rays, 1)

    ray_dirs_world = math_utils.quat_apply(
        yaw_rep.reshape(-1, 4),
        local_dirs.reshape(-1, 3),
    ).reshape(num_envs, num_rays, 3)

    ray_dirs_xy = ray_dirs_world[:, :, :2]
    walls = _get_obs_wall_aabbs(env, wall_margin=wall_margin)

    distances = torch.stack([
        _ray_aabb_distance_2d(root_xy, ray_dirs_xy[:, i, :], walls, max_range)
        for i in range(num_rays)
    ], dim=-1)

    return torch.clamp(distances / max_range, 0.0, 1.0)


# =============================================================================
# OBSERVATION FUNCTIONS — Scan History (active from Stage 9)
# =============================================================================

def scan_history_dummy(
    env: ManagerBasedRLEnv,
    num_rays: int = 5,
    num_frames: int = 2,
) -> torch.Tensor:
    """Dummy scan history returning 1.0 (max range) for all rays and frames.
    Used in Stages 0-8 before temporal sensing is active.
    Shape: (N, num_rays * num_frames).
    """
    return torch.ones((env.num_envs, num_rays * num_frames), device=env.device)


def scan_history_buffer(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    num_frames: int = 2,
) -> torch.Tensor:
    """Returns stacked scan history: [scan_t-1, scan_t-2].
    Provides implicit velocity information for moving obstacle detection.
    Shape: (N, num_rays * num_frames).
    """
    num_rays = len(ray_angles_deg)

    if not hasattr(env, "_scan_history"):
        env._scan_history = torch.ones(
            (env.num_envs, num_frames, num_rays), device=env.device
        )

    # Reset envs at episode start
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._scan_history[reset_mask] = 1.0

    # Get current scan
    current = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )

    # Return the history (t-1, t-2), then shift buffer
    output = env._scan_history.reshape(env.num_envs, -1).clone()

    # Shift: t-2 = old t-1, t-1 = current
    env._scan_history[:, 1:] = env._scan_history[:, :-1].clone()
    env._scan_history[:, 0] = current

    return output


# =============================================================================
# OBSERVATION FUNCTIONS — Agent Detection (active from Stage 11)
# =============================================================================

def agent_scan_dummy(
    env: ManagerBasedRLEnv,
    num_rays: int = 5,
) -> torch.Tensor:
    """Dummy agent scan returning 1.0 (no agents detected) for all rays.
    Used in Stages 0-10 before multi-agent sensing is active.
    Shape: (N, num_rays).
    """
    return torch.ones((env.num_envs, num_rays), device=env.device)


# =============================================================================
# REWARD FUNCTIONS — Locomotion
# =============================================================================

def grounded_speed_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_cap: float = 0.3,
    min_force: float = 1.0,
) -> torch.Tensor:
    """Reward horizontal speed in ANY direction while grounded (2+ feet on ground).
    Zero reward during jumps — direction-agnostic version of grounded_forward_speed_reward.
    """
    asset = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    speed = torch.clamp(speed, max=speed_cap)

    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, foot_cfg.body_ids, :3]
    force_mag = torch.norm(foot_forces, dim=-1)
    feet_on_ground = (force_mag > min_force).sum(dim=-1)

    return torch.where(feet_on_ground >= 2, speed, torch.zeros_like(speed))


def grounded_forward_speed_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_cap: float = 0.5,
    min_force: float = 1.0,
) -> torch.Tensor:
    """Reward forward speed only while grounded (2+ feet on ground).
    Zero reward during jumps — removes incentive for ballistic forward motion.
    """
    asset = env.scene[asset_cfg.name]
    vx = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=speed_cap)

    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, foot_cfg.body_ids, :3]
    force_mag = torch.norm(foot_forces, dim=-1)
    feet_on_ground = (force_mag > min_force).sum(dim=-1)

    return torch.where(feet_on_ground >= 2, vx, torch.zeros_like(vx))


def feet_contact_count_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_force: float = 1.0,
) -> torch.Tensor:
    """Reward proportional to fraction of feet in contact. Range [0, 1]."""
    asset = env.scene[asset_cfg.name]
    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, asset_cfg.body_ids, :3]
    force_mag = torch.norm(foot_forces, dim=-1)
    in_contact = (force_mag > min_force).float()
    return in_contact.mean(dim=-1)


def feet_airborne_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_force: float = 1.0,
) -> torch.Tensor:
    """Penalize jumping: 1.0 if all 4 feet off ground, 0.5 if 3, else 0.0."""
    asset = env.scene[asset_cfg.name]
    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, asset_cfg.body_ids, :3]
    force_mag = torch.norm(foot_forces, dim=-1)
    num_airborne = (force_mag < min_force).float().sum(dim=-1)

    penalty = torch.zeros(env.num_envs, device=env.device)
    penalty = torch.where(num_airborne >= 4, torch.ones_like(penalty), penalty)
    penalty = torch.where(num_airborne == 3, torch.full_like(penalty, 0.5), penalty)
    return penalty


def vertical_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cap: float = 1.0,
) -> torch.Tensor:
    """Absolute z-velocity clamped to cap. Use with negative weight."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 2].abs(), max=cap)


def lateral_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_cap: float = 0.5,
) -> torch.Tensor:
    """Absolute y-velocity clamped to speed_cap. Use with negative weight."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_lin_vel_w[:, 1].abs(), max=speed_cap)


def body_frame_lateral_speed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute lateral velocity in robot body frame. Shape (N,)."""
    asset = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_w[:, :2]
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    left_axis = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
    return (vel_xy * left_axis).sum(dim=-1)


def lateral_motion_when_clear(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_cap: float = 0.5,
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    front_threshold: float = 3.5,
) -> torch.Tensor:
    """Penalize lateral motion only when path ahead is clear.
    When wall is ahead (front ray < threshold), returns zeros to allow dodging.
    """
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    front_dist_norm = scan[:, 0]
    threshold_norm = front_threshold / max_range

    vy_abs = torch.clamp(body_frame_lateral_speed(env, asset_cfg).abs(), max=speed_cap)

    path_clear = front_dist_norm > threshold_norm
    return torch.where(path_clear, vy_abs, torch.zeros_like(vy_abs))


def yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rate_cap: float = 2.0,
) -> torch.Tensor:
    """Absolute yaw rate clamped to rate_cap. Use with negative weight."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.root_ang_vel_w[:, 2].abs(), max=rate_cap)


def roll_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Absolute roll angle clamped to 1.0. Use with negative weight.
    Penalizes side lean that creates asymmetric gaits.
    """
    asset = env.scene[asset_cfg.name]
    roll = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[0]
    return torch.clamp(roll.abs(), max=1.0)


# =============================================================================
# REWARD FUNCTIONS — Navigation (active from Stage 4)
# =============================================================================

def waypoint_progress_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
) -> torch.Tensor:
    """Reward reduction in distance to current waypoint.
    Advances to next waypoint when within reach_radius.
    Returns clamped delta distance in [-0.5, 0.5].
    """
    wp, idx, prev_dist = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    # Distance to current waypoint
    target_xy = wp[idx, :2]
    current_dist = torch.norm(xy - target_xy, dim=-1)

    # Advance waypoint if close enough
    close_enough = current_dist < reach_radius
    max_idx = len(waypoints) - 1
    new_idx = torch.clamp(idx + close_enough.long(), max=max_idx)

    # If waypoint advanced, recalculate distance to new waypoint
    advanced = new_idx != idx
    if advanced.any():
        new_target_xy = wp[new_idx, :2]
        new_dist = torch.norm(xy - new_target_xy, dim=-1)
        current_dist = torch.where(advanced, new_dist, current_dist)

    env._wp_idx = new_idx

    # Reward = reduction in distance
    reward = prev_dist - current_dist
    env._wp_prev_dist = current_dist.detach()

    return torch.clamp(reward, -0.5, 0.5)


def waypoint_index_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
) -> torch.Tensor:
    """Per-step reward proportional to current waypoint index.
    Returns idx / num_waypoints each step, so:
      - At waypoint 0: 0.0/step (haven't reached anything yet)
      - At waypoint 1: 0.2/step (reached first waypoint, 5 total)
      - At waypoint 4: 0.8/step
      - At waypoint 5: 1.0/step (final)
    Creates compounding incentive: reaching wp2 at step 500 gives
    0.4 * remaining_steps of reward for the rest of the episode.
    Must be called AFTER waypoint_progress_reward which advances _wp_idx.
    """
    _, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    num_wp = len(waypoints)
    return idx.float() / num_wp


def speed_toward_waypoint(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
    speed_cap: float = 0.5,
) -> torch.Tensor:
    """Reward velocity component directed toward the current waypoint.
    Returns dot(velocity_xy, unit_direction_to_waypoint), clamped to (0, speed_cap).
    Gives ~0.3/step when walking toward waypoint, 0.0 when standing or moving away.
    """
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]

    target_xy = wp[idx, :2]
    delta = target_xy - xy
    dist = torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
    direction = delta / dist

    speed_proj = (vel_xy * direction).sum(dim=-1)
    return torch.clamp(speed_proj, 0.0, speed_cap)


def waypoint_heading_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((5.0, 1.5, 0.0), (6.5, 6.5, 0.0)),
    reach_radius: float = 1.0,
) -> torch.Tensor:
    """Reward facing toward the current waypoint. Returns clamp(cos(angle_diff), 0, 1).
    Gives continuous reward for turning to face the waypoint even when standing still.
    1.0 = perfectly facing waypoint, 0.0 = facing away or perpendicular.
    """
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]

    target_xy = wp[idx, :2]
    direction = target_xy - xy
    target_angle = torch.atan2(direction[:, 1], direction[:, 0])
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    angle_diff = target_angle - yaw
    return torch.clamp(torch.cos(angle_diff), 0.0, 1.0)


def final_waypoint_reached(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((6.5, 8.0, 0.0),),
    reach_radius: float = 0.6,
) -> torch.Tensor:
    """Returns True when agent reaches the final waypoint."""
    wp, idx, _ = _ensure_waypoint_state(env, waypoints, asset_cfg)
    asset = env.scene[asset_cfg.name]
    pos_local = asset.data.root_pos_w - env.scene.env_origins
    xy = pos_local[:, :2]
    final_idx = len(waypoints) - 1
    at_final = idx == final_idx
    dist = torch.norm(xy - wp[final_idx, :2].unsqueeze(0), dim=-1)
    return at_final & (dist < reach_radius)


def final_goal_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoints: tuple = ((6.5, 8.0, 0.0),),
    reach_radius: float = 0.6,
) -> torch.Tensor:
    """Returns 1.0 when final waypoint reached, else 0.0."""
    reached = final_waypoint_reached(env, asset_cfg, waypoints, reach_radius)
    return reached.float()


# =============================================================================
# REWARD FUNCTIONS — Obstacle Avoidance (active from Stage 2)
# =============================================================================

def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    threshold: float = 0.8,
) -> torch.Tensor:
    """Weighted proximity penalty across all rays. Use with negative weight."""
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    weights = torch.tensor([1.5, 1.0, 1.0, 0.6, 0.6], device=scan.device, dtype=scan.dtype)
    closeness = torch.clamp(threshold - scan, min=0.0) / threshold
    return (closeness * weights).sum(dim=1) / weights.sum()


def front_wall_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Stronger penalty for obstacles in front arc. Use with negative weight."""
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    front_block = torch.clamp(threshold - scan[:, 0], min=0.0) / threshold
    fl_block = torch.clamp(threshold - scan[:, 1], min=0.0) / threshold
    fr_block = torch.clamp(threshold - scan[:, 2], min=0.0) / threshold
    return 0.5 * front_block + 0.25 * fl_block + 0.25 * fr_block


def forward_heading_when_clear(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    min_front_dist: float = 1.5,
) -> torch.Tensor:
    """Reward facing +x only when path ahead is clear.
    Zero near obstacles — lets the ant turn freely.
    """
    asset = env.scene[asset_cfg.name]
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    front_dist_norm = scan[:, 0]
    threshold_norm = min_front_dist / max_range

    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
    cos_yaw = torch.clamp(torch.cos(yaw), min=0.0)

    path_clear = front_dist_norm > threshold_norm
    return torch.where(path_clear, cos_yaw, torch.zeros_like(cos_yaw))


def lateral_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
) -> torch.Tensor:
    """Reward having open space on at least one side.
    Returns max of left-side and right-side average ray distances.
    """
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    left_avg = (scan[:, 1] + scan[:, 3]) / 2.0
    right_avg = (scan[:, 2] + scan[:, 4]) / 2.0
    return torch.maximum(left_avg, right_avg)


def directed_dodge_when_blocked(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float = 5.0,
    wall_margin: float = 0.35,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    front_threshold: float = 3.5,
) -> torch.Tensor:
    """Reward dodging toward the open side when wall is ahead.
    Zero when path is clear. Directional: +y if left open, -y if right open.
    """
    scan = obstacle_scan_5(
        env, asset_cfg=asset_cfg, max_range=max_range,
        wall_margin=wall_margin, ray_angles_deg=ray_angles_deg,
    )
    front_dist_norm = scan[:, 0]
    threshold_norm = front_threshold / max_range

    vy = body_frame_lateral_speed(env, asset_cfg)

    left_avg = (scan[:, 1] + scan[:, 3]) / 2.0
    right_avg = (scan[:, 2] + scan[:, 4]) / 2.0
    left_has_more_space = left_avg >= right_avg

    reward_left = torch.clamp(vy, min=0.0, max=0.5)
    reward_right = torch.clamp(-vy, min=0.0, max=0.5)
    directed_reward = torch.where(left_has_more_space, reward_left, reward_right)

    blocked = front_dist_norm < threshold_norm
    return torch.where(blocked, directed_reward, torch.zeros_like(vy))


# =============================================================================
# UTILITY FUNCTIONS — Reset / Spawn
# =============================================================================

def reset_root_state_random_corridor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_range: tuple[float, float] = (-2.5, 4.1),
    y_range: tuple[float, float] = (-1.5, 1.5),
    exclusion_zones: list[tuple[float, float, float, float]] | None = None,
    z_offset: float = 0.6,
    random_yaw: bool = True,
    max_tries: int = 200,
) -> None:
    """Reset root state to a uniformly random valid position in the corridor.

    Samples (x, y) uniformly from the bounding box, rejecting positions that
    fall inside any exclusion zone (wall interiors, obstacle footprints + buffer).

    Args:
        x_range: (x_min, x_max) in env-local coords.
        y_range: (y_min, y_max) in env-local coords.
        exclusion_zones: list of (x_min, x_max, y_min, y_max) rectangles to reject.
        z_offset: height above ground for spawn.
        random_yaw: if True, sample full 0–2π yaw; otherwise preserve default.
        max_tries: rejection sampling iterations before falling back to safe point.
    """
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device

    if exclusion_zones is None:
        exclusion_zones = []

    x_min, x_max = x_range
    y_min, y_max = y_range

    # Start everyone at safe fallback; rejection sampling replaces valid ones
    valid_x = torch.full((n,), -2.0, dtype=torch.float32, device=device)
    valid_y = torch.zeros(n, dtype=torch.float32, device=device)
    needs_sample = torch.ones(n, dtype=torch.bool, device=device)

    for _ in range(max_tries):
        if not needs_sample.any():
            break
        n_rem = int(needs_sample.sum().item())
        x_cand = torch.rand(n_rem, device=device) * (x_max - x_min) + x_min
        y_cand = torch.rand(n_rem, device=device) * (y_max - y_min) + y_min

        in_exclusion = torch.zeros(n_rem, dtype=torch.bool, device=device)
        for (ex0, ex1, ey0, ey1) in exclusion_zones:
            in_zone = (x_cand >= ex0) & (x_cand <= ex1) & (y_cand >= ey0) & (y_cand <= ey1)
            in_exclusion = in_exclusion | in_zone

        accepted = ~in_exclusion
        rem_idx = needs_sample.nonzero(as_tuple=True)[0]
        newly_accepted = rem_idx[accepted]
        valid_x[newly_accepted] = x_cand[accepted]
        valid_y[newly_accepted] = y_cand[accepted]
        needs_sample[newly_accepted] = False

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, 0] = valid_x + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = valid_y + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = z_offset + env.scene.env_origins[env_ids, 2]

    if random_yaw:
        yaw = torch.rand(n, device=device) * (2.0 * torch.pi)
        half = yaw * 0.5
        root_state[:, 3] = torch.cos(half)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(half)

    root_state[:, 7:] = 0.0
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)

def kinematic_object_scan(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_names: tuple[str, ...] = (),
    obstacle_half_size: float = 0.35,
    max_range: float = 5.0,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
) -> torch.Tensor:
    """5-ray obstacle scan reading actual RigidObject positions (not cached AABB).

    Works correctly when obstacles are repositioned each episode via write_root_state_to_sim.
    Each obstacle is approximated as a cylinder of radius obstacle_half_size.
    Returns (N, num_rays) normalized to [0, 1]: 0 = very close, 1 = max range.
    """
    asset = env.scene[asset_cfg.name]
    num_envs = env.num_envs
    num_rays = len(ray_angles_deg)
    device = env.device

    root_pos_w = asset.data.root_pos_w           # (N, 3)
    root_xy = root_pos_w[:, :2]                  # (N, 2)
    root_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)  # (N, 4)

    # Build world-frame ray directions
    ray_angles = torch.tensor(ray_angles_deg, device=device, dtype=root_xy.dtype) * (math.pi / 180.0)
    local_dirs = torch.stack(
        [torch.cos(ray_angles), torch.sin(ray_angles), torch.zeros_like(ray_angles)], dim=-1
    )  # (num_rays, 3)
    local_dirs_exp = local_dirs.unsqueeze(0).expand(num_envs, -1, -1)    # (N, num_rays, 3)
    yaw_rep = root_yaw_quat.unsqueeze(1).expand(-1, num_rays, -1)        # (N, num_rays, 4)
    ray_dirs_world = math_utils.quat_apply(
        yaw_rep.reshape(-1, 4), local_dirs_exp.reshape(-1, 3)
    ).reshape(num_envs, num_rays, 3)
    ray_dirs_xy = ray_dirs_world[:, :, :2]  # (N, num_rays, 2)

    distances = torch.full((num_envs, num_rays), max_range, device=device)

    for obs_name in obstacle_names:
        obs_xy = env.scene[obs_name].data.root_pos_w[:, :2]  # (N, 2) — always current
        rel_xy = obs_xy - root_xy                             # (N, 2)

        for r in range(num_rays):
            ray_d = ray_dirs_xy[:, r, :]                      # (N, 2)
            along = (rel_xy * ray_d).sum(dim=-1)              # (N,) — signed distance along ray
            lateral_sq = torch.clamp((rel_xy ** 2).sum(dim=-1) - along ** 2, min=0.0)
            lateral = torch.sqrt(lateral_sq)                  # (N,) — perpendicular distance

            hit = (along > 0.0) & (along < max_range) & (lateral < obstacle_half_size)
            hit_dist = torch.clamp(along - obstacle_half_size, min=0.0)
            distances[:, r] = torch.minimum(
                distances[:, r],
                torch.where(hit, hit_dist, torch.full_like(hit_dist, max_range)),
            )

    return torch.clamp(distances / max_range, 0.0, 1.0)


def kinematic_obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_names: tuple[str, ...] = (),
    obstacle_half_size: float = 0.35,
    max_range: float = 5.0,
    ray_angles_deg: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0, -90.0),
    threshold: float = 0.8,
) -> torch.Tensor:
    """Proximity penalty based on kinematic obstacle scan (reads actual positions).

    Penalizes being within `threshold` (normalised) of any obstacle on any ray.
    Front ray weighted 1.5×, side rays 0.6×. Use with negative reward weight.
    """
    scan = kinematic_object_scan(
        env, asset_cfg=asset_cfg,
        obstacle_names=obstacle_names,
        obstacle_half_size=obstacle_half_size,
        max_range=max_range,
        ray_angles_deg=ray_angles_deg,
    )
    weights = torch.tensor([1.5, 1.0, 1.0, 0.6, 0.6], device=scan.device, dtype=scan.dtype)
    closeness = torch.clamp(threshold - scan, min=0.0) / threshold
    return (closeness * weights).sum(dim=1) / weights.sum()


def reset_random_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    obstacle_names: tuple[str, ...] = (),
    x_range: tuple[float, float] = (-8.0, 12.0),
    y_range: tuple[float, float] = (-8.0, 12.0),
    spawn_safe_radius: float = 3.0,
    min_obs_dist: float = 1.8,
    z_height: float = 0.6,
    max_tries: int = 200,
) -> None:
    """Place each kinematic obstacle at a random position across the full arena.

    Rejection-samples positions to avoid:
      - spawn safe zone (circle of spawn_safe_radius around origin)
      - minimum distance to every previously placed obstacle (min_obs_dist)
    This prevents clustering while spreading obstacles across the full plane.
    """
    n = len(env_ids)
    device = env.device
    x_min, x_max = x_range
    y_min, y_max = y_range

    placed: list[torch.Tensor] = []  # (n, 2) tensors for each obstacle placed so far

    for obs_name in obstacle_names:
        obs = env.scene[obs_name]

        valid_x = torch.zeros(n, device=device)
        valid_y = torch.zeros(n, device=device)
        needs = torch.ones(n, dtype=torch.bool, device=device)

        for _ in range(max_tries):
            if not needs.any():
                break
            n_rem = int(needs.sum().item())
            rem_idx = needs.nonzero(as_tuple=True)[0]
            x_c = torch.rand(n_rem, device=device) * (x_max - x_min) + x_min
            y_c = torch.rand(n_rem, device=device) * (y_max - y_min) + y_min

            # Exclude spawn safe zone
            bad = (x_c ** 2 + y_c ** 2) < spawn_safe_radius ** 2

            # Exclude positions too close to already-placed obstacles
            for prev in placed:
                px = prev[rem_idx, 0]
                py = prev[rem_idx, 1]
                bad = bad | (((x_c - px) ** 2 + (y_c - py) ** 2) < min_obs_dist ** 2)

            valid = ~bad
            newly_valid = rem_idx[valid]
            valid_x[newly_valid] = x_c[valid]
            valid_y[newly_valid] = y_c[valid]
            needs[newly_valid] = False

        placed.append(torch.stack([valid_x, valid_y], dim=-1))

        state = obs.data.default_root_state[env_ids].clone()
        state[:, 0] = valid_x + env.scene.env_origins[env_ids, 0]
        state[:, 1] = valid_y + env.scene.env_origins[env_ids, 1]
        state[:, 2] = z_height + env.scene.env_origins[env_ids, 2]
        state[:, 3] = 1.0
        state[:, 4:7] = 0.0
        state[:, 7:] = 0.0
        obs.write_root_state_to_sim(state, env_ids=env_ids)


def reset_root_state_fixed_spawns(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spawn_points: list[tuple[float, float, float]] | None = None,
    xy_jitter: float = 0.0,
    yaw_jitter: float = 0.0,
    check_radius: float = 0.9,
    z_offset: float = 0.6,
) -> None:
    """Reset root state to a randomly selected spawn point with optional jitter."""
    if spawn_points is None:
        spawn_points = [(0.0, 0.0, 0.0)]

    asset = env.scene[asset_cfg.name]
    n = len(env_ids)

    indices = torch.randint(len(spawn_points), (n,))
    pts = torch.tensor(spawn_points, dtype=torch.float32, device=env.device)
    chosen = pts[indices]

    root_state = asset.data.default_root_state[env_ids].clone()

    root_state[:, 0] = chosen[:, 0] + env.scene.env_origins[env_ids, 0]
    root_state[:, 1] = chosen[:, 1] + env.scene.env_origins[env_ids, 1]
    root_state[:, 2] = chosen[:, 2] + z_offset + env.scene.env_origins[env_ids, 2]

    if xy_jitter > 0.0:
        root_state[:, 0] += (torch.rand(n, device=env.device) * 2 - 1) * xy_jitter
        root_state[:, 1] += (torch.rand(n, device=env.device) * 2 - 1) * xy_jitter

    if yaw_jitter > 0.0:
        yaw = (torch.rand(n, device=env.device) * 2 - 1) * yaw_jitter
        half = yaw * 0.5
        root_state[:, 3] = torch.cos(half)
        root_state[:, 4] = 0.0
        root_state[:, 5] = 0.0
        root_state[:, 6] = torch.sin(half)

    root_state[:, 7:] = 0.0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
