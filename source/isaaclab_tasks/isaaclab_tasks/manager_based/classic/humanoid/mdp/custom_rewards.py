# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions to enforce 4-leg gait for the Ant robot."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_contact_count_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_force: float = 1.0,
) -> torch.Tensor:
    """Reward proportional to the number of feet in contact with the ground.

    Encourages the robot to use ALL 4 legs instead of finding a 3-leg shortcut.
    Measured via the incoming wrench (force) on each foot body.
    """
    asset = env.scene[asset_cfg.name]
    # body_incoming_joint_wrench_b: (num_envs, num_bodies, 6) → first 3 cols = force
    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, asset_cfg.body_ids, :3]  # (N, num_feet, 3)
    force_mag = torch.norm(foot_forces, dim=-1)       # (N, num_feet)

    in_contact = (force_mag > min_force).float()       # (N, num_feet)
    # Average fraction of feet in contact: 0.0 (none) → 1.0 (all four)
    return in_contact.mean(dim=-1)


def feet_force_variance_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise uneven force distribution across feet.

    A large variance means some legs bear much more load than others, which is
    a signature of a 3-leg (or fewer) gait.
    """
    asset = env.scene[asset_cfg.name]
    wrench = asset.data.body_incoming_joint_wrench_b
    foot_forces = wrench[:, asset_cfg.body_ids, :3]
    force_mag = torch.norm(foot_forces, dim=-1)        # (N, num_feet)

    # Normalised variance across feet
    variance = torch.var(force_mag, dim=-1)             # (N,)
    mean_force = force_mag.mean(dim=-1).clamp(min=1.0)
    return variance / mean_force                        # coefficient of variation–style
