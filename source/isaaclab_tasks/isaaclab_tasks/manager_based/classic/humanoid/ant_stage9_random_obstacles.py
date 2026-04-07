# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 9: Open arena with 10 random static obstacles and cycling random waypoints.

Harder version of Stage 8 V3 — obstacle count raised from 6 to 10.
Same open arena, same cycling waypoint logic, same reward structure.
Obstacle density increase forces the ant to navigate more carefully.

Key changes from Stage 8 V3:
- 10 obstacles (was 6)
- min_obs_dist reduced to 1.5 m (was 1.8) to fit 10 obstacles in same arena
- spawn_safe_radius kept at 1.5 m

Start from Stage 8 V3 checkpoint with noise_std patched to 0.3.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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

custom_mdp._obs_wall_cache = None

from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip


FEET_BODY_NAMES = [
    "front_left_foot",
    "front_right_foot",
    "left_back_foot",
    "right_back_foot",
]

TARGET_RANGE = 8.0
REACH_RADIUS = 0.8

# Fifteen obstacle slots — start underground, placed randomly each episode
OBSTACLE_NAMES = (
    "obs_0", "obs_1", "obs_2", "obs_3", "obs_4",
    "obs_5", "obs_6", "obs_7", "obs_8", "obs_9",
    "obs_10", "obs_11", "obs_12", "obs_13", "obs_14",
)
OBSTACLE_SIZE = (0.7, 0.7, 1.2)

SPAWN_X_RANGE = (-1.5, 1.5)
SPAWN_Y_RANGE = (-1.5, 1.5)


def _make_obstacle_cfg(prim_name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0)),
    )


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
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.85), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.854, 0.354, 0.0, 0.354)),
    )

    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target_marker",
        spawn=sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.1, 0.1), emissive_color=(0.8, 0.0, 0.0)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.5)),
    )

    obs_0 = _make_obstacle_cfg("obs_0")
    obs_1 = _make_obstacle_cfg("obs_1")
    obs_2 = _make_obstacle_cfg("obs_2")
    obs_3 = _make_obstacle_cfg("obs_3")
    obs_4 = _make_obstacle_cfg("obs_4")
    obs_5 = _make_obstacle_cfg("obs_5")
    obs_6 = _make_obstacle_cfg("obs_6")
    obs_7 = _make_obstacle_cfg("obs_7")
    obs_8 = _make_obstacle_cfg("obs_8")
    obs_9 = _make_obstacle_cfg("obs_9")
    obs_10 = _make_obstacle_cfg("obs_10")
    obs_11 = _make_obstacle_cfg("obs_11")
    obs_12 = _make_obstacle_cfg("obs_12")
    obs_13 = _make_obstacle_cfg("obs_13")
    obs_14 = _make_obstacle_cfg("obs_14")


@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=6.0,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Fixed 84-dim observation — identical structure to all prior stages."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(
            func=custom_mdp.random_target_angle,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
        )
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(
            func=custom_mdp.random_target_heading_proj,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
        )
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench, scale=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES)},
        )
        waypoint_angle = ObsTerm(
            func=custom_mdp.random_target_angle,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
        )
        waypoint_heading = ObsTerm(
            func=custom_mdp.random_target_heading_proj,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
        )
        waypoint_dist = ObsTerm(
            func=custom_mdp.random_target_distance,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS, "max_dist": 10.0},
        )
        obstacle_scan = ObsTerm(
            func=custom_mdp.kinematic_object_scan,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obstacle_names": OBSTACLE_NAMES,
                "obstacle_half_size": 0.4,
                "max_range": 5.0,
                "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            },
        )
        scan_history = ObsTerm(
            func=custom_mdp.scan_history_dummy,
            params={"num_rays": 5, "num_frames": 2},
        )
        agent_scan = ObsTerm(
            func=custom_mdp.agent_scan_dummy,
            params={"num_rays": 5},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # One-time startup: register obstacles so waypoints spawn in dense zones
    setup_target_bias = EventTerm(
        func=custom_mdp.set_obstacle_biased_targets,
        mode="startup",
        params={"obstacle_names": OBSTACLE_NAMES},
    )

    reset_base = EventTerm(
        func=custom_mdp.reset_root_state_random_corridor,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_range": SPAWN_X_RANGE,
            "y_range": SPAWN_Y_RANGE,
            "exclusion_zones": [],
            "z_offset": 0.6,
            "random_yaw": True,
        },
    )
    reset_obstacles = EventTerm(
        func=custom_mdp.reset_random_obstacles,
        mode="reset",
        params={
            "obstacle_names": OBSTACLE_NAMES,
            "x_range": (-8.0, 12.0),
            "y_range": (-8.0, 12.0),
            "spawn_safe_radius": 1.5,
            "min_obs_dist": 1.8,
            "z_height": 0.6,
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.1, 0.1)},
    )


@configclass
class RewardsCfg:

    progress = RewTerm(
        func=custom_mdp.random_target_progress,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
    )
    speed_toward = RewTerm(
        func=custom_mdp.random_target_speed_toward,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS, "speed_cap": 0.5},
    )
    heading_alignment = RewTerm(
        func=custom_mdp.random_target_heading_alignment,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_range": TARGET_RANGE, "reach_radius": REACH_RADIUS},
    )

    gait_quality = RewTerm(
        func=custom_mdp.grounded_speed_reward, weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES),
            "speed_cap": 0.3,
            "min_force": 1.0,
        },
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.03)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.15, params={"threshold": 0.93})
    feet_contact = RewTerm(
        func=custom_mdp.feet_contact_count_reward, weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES), "min_force": 1.0},
    )

    yaw_spin = RewTerm(
        func=custom_mdp.yaw_rate_penalty, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot"), "rate_cap": 2.0},
    )
    roll = RewTerm(func=custom_mdp.roll_penalty, weight=-0.3, params={"asset_cfg": SceneEntityCfg("robot")})
    airborne = RewTerm(
        func=custom_mdp.feet_airborne_penalty, weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES), "min_force": 1.0},
    )
    vertical_velocity = RewTerm(
        func=custom_mdp.vertical_velocity_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "cap": 1.0},
    )
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    energy = RewTerm(func=mdp.power_consumption, weight=-0.005, params={"gear_ratio": {".*": 15.0}})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.01,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )

    obstacle_proximity = RewTerm(
        func=custom_mdp.kinematic_obstacle_proximity_penalty,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_names": OBSTACLE_NAMES,
            "obstacle_half_size": 0.4,
            "max_range": 5.0,
            "ray_angles_deg": (0.0, 45.0, -45.0, 90.0, -90.0),
            "threshold": 0.4,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})


@configclass
class AntStage9RandomObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """Stage 9: Open arena with 10 random static obstacles and cycling random waypoints."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=30.0, clone_in_fabric=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.viewer = ViewerCfg(eye=(5.0, -20.0, 25.0), lookat=(5.0, 5.0, 0.0))
        self.decimation = 2
        self.episode_length_s = 96.0
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
