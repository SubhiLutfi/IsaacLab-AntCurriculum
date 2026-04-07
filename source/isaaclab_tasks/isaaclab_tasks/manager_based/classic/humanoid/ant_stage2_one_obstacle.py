# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2: Corridor with one chicane obstacle.

Activates real obstacle scan (5 rays) and obstacle avoidance rewards.
All other observation channels remain identical to Stage 1.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
import isaaclab_tasks.manager_based.classic.humanoid.custom_mdp as custom_mdp

from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip


FEET_BODY_NAMES = [
    "front_left_foot",
    "front_right_foot",
    "left_back_foot",
    "right_back_foot",
]


@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    wall_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_1",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 0.3, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 2.3, 0.6)),
    )

    wall_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_2",
        spawn=sim_utils.CuboidCfg(
            size=(8.0, 0.3, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, -2.3, 0.6)),
    )

    chicane_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/chicane_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 2.2, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, -0.7, 0.6)),
    )


@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=6.0,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Complete observation space — dimension FIXED across all stages."""

        # --- Proprioception (always active) ---
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(
            func=mdp.base_angle_to_target,
            params={"target_pos": (1000.0, 0.0, 0.0)},
        )
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(
            func=mdp.base_heading_proj,
            params={"target_pos": (1000.0, 0.0, 0.0)},
        )
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench, scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES)},
        )

        # --- Navigation (DUMMY Stage 0-3, LIVE from Stage 4) ---
        waypoint_angle = ObsTerm(
            func=custom_mdp.waypoint_angle_dummy,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        waypoint_heading = ObsTerm(
            func=custom_mdp.waypoint_heading_proj_dummy,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        waypoint_dist = ObsTerm(
            func=custom_mdp.waypoint_distance_dummy,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # --- Obstacle Scan (LIVE from Stage 2) ---
        obstacle_scan = ObsTerm(
            func=custom_mdp.obstacle_scan_5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_range": 5.0,
                "wall_margin": 0.35,
                "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            },
        )

        # --- Scan History (DUMMY Stage 0-8, LIVE from Stage 9) ---
        scan_history = ObsTerm(
            func=custom_mdp.scan_history_dummy,
            params={"num_rays": 5, "num_frames": 2},
        )

        # --- Agent Detection (DUMMY Stage 0-10, LIVE from Stage 11) ---
        agent_scan = ObsTerm(
            func=custom_mdp.agent_scan_dummy,
            params={"num_rays": 5},
        )

        # --- Last Action (always active) ---
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=custom_mdp.reset_root_state_fixed_spawns,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spawn_points": [(-2.0, 0.0, 0.0)],
            "xy_jitter": 0.0,
            "yaw_jitter": 0.0,
            "check_radius": 0.9,
            "z_offset": 0.6,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
    )


@configclass
class RewardsCfg:
    """Locomotion + obstacle avoidance rewards."""

    progress = RewTerm(
        func=mdp.progress_reward, weight=1.5,
        params={"target_pos": (1000.0, 0.0, 0.0)},
    )
    forward_speed = RewTerm(
        func=custom_mdp.grounded_forward_speed_reward, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
            "speed_cap": 0.3, "min_force": 1.0,
        },
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.03)
    upright = RewTerm(
        func=mdp.upright_posture_bonus, weight=0.15,
        params={"threshold": 0.93},
    )
    feet_contact = RewTerm(
        func=custom_mdp.feet_contact_count_reward, weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES), "min_force": 1.0},
    )
    lateral_motion = RewTerm(
        func=custom_mdp.lateral_motion_when_clear,
        weight=-0.35,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "speed_cap": 0.5,
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            "front_threshold": 3.5,
        },
    )
    yaw_spin = RewTerm(
        func=custom_mdp.yaw_rate_penalty, weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot"), "rate_cap": 2.0},
    )
    roll = RewTerm(
        func=custom_mdp.roll_penalty,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    airborne = RewTerm(
        func=custom_mdp.feet_airborne_penalty, weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES), "min_force": 1.0},
    )
    vertical_velocity = RewTerm(
        func=custom_mdp.vertical_velocity_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "cap": 1.0},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.008)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    energy = RewTerm(
        func=mdp.power_consumption, weight=-0.002,
        params={"gear_ratio": {".*": 15.0}},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.01,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )
    obstacle_proximity = RewTerm(
        func=custom_mdp.obstacle_proximity_penalty,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            "threshold": 0.8,
        },
    )
    front_wall = RewTerm(
        func=custom_mdp.front_wall_penalty,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0),
            "threshold": 1.0,
        },
    )
    forward_heading = RewTerm(
        func=custom_mdp.forward_heading_when_clear,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            "min_front_dist": 1.5,
        },
    )
    lateral_clearance = RewTerm(
        func=custom_mdp.lateral_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
        },
    )
    dodge = RewTerm(
        func=custom_mdp.directed_dodge_when_blocked,
        weight=0.6,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_range": 5.0,
            "wall_margin": 0.35,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            "front_threshold": 3.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.31},
    )


@configclass
class AntStage2OneObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 2: corridor with one chicane, real obstacle scan active."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=15.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.viewer = ViewerCfg(eye=(8.0, -8.0, 6.0), lookat=(0.0, 0.0, 0.0))
        self.decimation = 2
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
