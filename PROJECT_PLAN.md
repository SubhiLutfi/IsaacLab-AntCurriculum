# Quadruped Robot Obstacle Avoidance — Complete Project Plan

## 1. Robot Selection

### Real-World Landscape

| Robot | DOF | Weight | Sensors | Isaac Lab Support | Price |
|-------|-----|--------|---------|-------------------|-------|
| MuJoCo Ant | 8 (simplified) | 0.9 kg | None (add custom) | Built-in `ANT_CFG` | Free (sim only) |
| Unitree Go2 | 12 | 15 kg | LiDAR, cameras, IMU | Built-in `UNITREE_GO2_CFG` | ~$1,600 |
| Unitree A1 | 12 | 12 kg | Depth cameras, IMU | Built-in `UNITREE_A1_CFG` | ~$2,500 |
| ANYmal C/D | 12 | 50 kg | LiDAR, cameras, IMU | Built-in `ANYMAL_C_CFG` | ~$200,000 |
| Boston Dynamics Spot | 12 | 32 kg | LiDAR, cameras, IMU | Built-in `SPOT_CFG` | ~$75,000 |

### Recommendation: Keep the MuJoCo Ant

**Reasoning:**
- You already have all reward functions debugged and working for the Ant
- The Ant's 8 DOF is simpler — faster training iterations — more learning per hour
- Switching to Unitree Go2 (12 DOF) would require rewriting action spaces, reward functions, and all joint configs
- The locomotion + navigation + avoidance challenges are identical regardless of robot
- For a learning project, the Ant teaches the same RL concepts as a Go2 but trains 3-5x faster
- If you want to transfer to a real robot later, the **reward structure and curriculum** transfer directly — only the robot config changes

**Future upgrade path:** After completing this project, swap `ANT_CFG` for `UNITREE_GO2_CFG`, adjust joint names and action scale, and re-run the same curriculum. The architecture, rewards, and training pipeline stay identical.

---

## 2. Architecture: Observation Space (Fixed From Step 0)

Design the full observation space upfront. Every channel exists from Step 0 — unused channels receive dummy values.

```
Observation Vector (total: variable depending on robot)
├── Proprioception (always active)
│   ├── base_height           (1)
│   ├── base_lin_vel          (3)
│   ├── base_ang_vel          (3)
│   ├── base_yaw_roll         (2)
│   ├── base_up_proj          (1)
│   ├── joint_pos_norm        (8)   # Ant has 8 joints
│   ├── joint_vel_rel         (8)
│   ├── feet_body_forces      (24)  # 4 feet × 6 wrench
│   └── last_action           (8)
│
├── Navigation (active from Stage 4, dummy before)
│   ├── waypoint_angle        (1)   # angle to current waypoint
│   ├── waypoint_heading_proj (1)   # cos(angle to waypoint)
│   └── waypoint_distance     (1)   # normalized dist to waypoint
│
├── Obstacle Scan — Current Frame (active from Stage 2, dummy before)
│   └── obstacle_scan_5       (5)   # 5 ray distances, normalized [0,1]
│
├── Obstacle Scan — History (active from Stage 9, dummy before)
│   ├── scan_t_minus_1        (5)   # previous frame scan
│   └── scan_t_minus_2        (5)   # two frames ago scan
│
└── Agent Detection (active from Stage 11, dummy before)
    └── agent_scan_5          (5)   # distance to nearest other agent per ray
```

**Total observation dimension: ~81** (exact count depends on wrench dimensions)

**Critical rule:** This dimension NEVER changes. All checkpoints load across all stages.

### Dummy Value Schedule

| Channel | Stages 0-1 | Stages 2-3 | Stage 4-8 | Stage 9-10 | Stage 11+ |
|---------|-----------|-----------|----------|-----------|----------|
| waypoint_angle | 0.0 | 0.0 | **LIVE** | LIVE | LIVE |
| waypoint_heading_proj | 1.0 (facing +x) | 1.0 | **LIVE** | LIVE | LIVE |
| waypoint_distance | 1.0 (far away) | 1.0 | **LIVE** | LIVE | LIVE |
| obstacle_scan_5 | 1.0 (max range) | **LIVE** | LIVE | LIVE | LIVE |
| scan_t_minus_1 | 1.0 | 1.0 | 1.0 | **LIVE** | LIVE |
| scan_t_minus_2 | 1.0 | 1.0 | 1.0 | **LIVE** | LIVE |
| agent_scan_5 | 1.0 | 1.0 | 1.0 | 1.0 | **LIVE** |

---

## 3. Reward Functions (Final Set)

All functions exist in `custom_mdp.py` from day one. Unused rewards have weight=0.0 until their stage.

### Locomotion Rewards (always active)
- `grounded_forward_speed_reward` — x-velocity clamped, zero when airborne
- `is_alive` — light survival bonus
- `upright_posture_bonus` — torso level
- `feet_contact_count_reward` — encourage 4-leg gait

### Locomotion Penalties (always active)
- `lateral_motion_when_clear` — penalize sideways drift only when path is clear
- `yaw_rate_penalty` — penalize spinning
- `feet_airborne_penalty` — penalize jumping
- `vertical_velocity_penalty` — penalize z-axis motion
- `action_l2` — light action regularization
- `power_consumption` — light energy penalty
- `joint_pos_limits_penalty_ratio` — joint limit penalty

### Navigation Rewards (active from Stage 4)
- `waypoint_progress_reward` — reduction in distance to current waypoint

### Obstacle Rewards (active from Stage 2)
- `obstacle_proximity_penalty` — penalize being near walls
- `front_wall_penalty` — stronger penalty for head-on approach
- `forward_heading_when_clear` — face +x only when path ahead is clear
- `lateral_clearance_reward` — reward having open space on one side
- `directed_dodge_when_blocked` — dodge toward open side when blocked

### Frozen Reward Weights by Stage

| Reward | S0 | S1 | S2 | S3 | S4 | S5-8 | S9-10 | S11+ |
|--------|----|----|----|----|----|----|-------|------|
| progress/waypoint | 1.0 | 1.0 | 1.5 | 1.5 | 2.0 | 2.0 | 2.0 | 2.0 |
| forward_speed | 1.5 | 1.5 | 1.0 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 |
| alive | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| upright | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| feet_contact | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| lateral_motion | -0.35 | -0.35 | -0.35* | -0.35* | -0.35* | -0.35* | -0.35* | -0.35* |
| yaw_spin | -0.2 | -0.2 | -0.2 | -0.2 | -0.2 | -0.2 | -0.2 | -0.2 |
| airborne | -0.8 | -0.8 | -0.8 | -0.8 | -0.8 | -0.8 | -0.8 | -0.8 |
| vertical_velocity | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 | -0.5 |
| action_l2 | -0.008 | -0.008 | -0.008 | -0.008 | -0.008 | -0.008 | -0.008 | -0.008 |
| energy | -0.002 | -0.002 | -0.002 | -0.002 | -0.002 | -0.002 | -0.002 | -0.002 |
| joint_pos_limits | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 | -0.01 |
| obstacle_proximity | 0 | 0 | -0.08 | -0.08 | -0.08 | -0.08 | -0.08 | -0.08 |
| front_wall | 0 | 0 | -0.25 | -0.25 | -0.25 | -0.25 | -0.25 | -0.25 |
| forward_heading | 0 | 0 | 0.4 | 0.4 | 0 | 0 | 0 | 0 |
| lateral_clearance | 0 | 0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| dodge | 0 | 0 | 0 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |

*\* = `lateral_motion_when_clear` (conditional version)*

---

## 4. Maze Geometry (Fixed)

```
Corridor: x = -3 to x = 5, y = -2.15 to y = 2.15 (walls at ±2.3)
Chicane 1: x = 0.5, gap on +y side
Chicane 2: x = 4.0, gap on -y side (wider spacing than original 2.8)
End wall: x = 5.0, gap on +y side (for turn entry)
Turn corridor: x = 5.2 to 7.8, running +y direction
```

### Waypoints
```
Waypoint 1: (4.5, 1.5, 0.0)   — end of horizontal corridor
Waypoint 2: (6.5, 4.0, 0.0)   — middle of vertical corridor
Waypoint 3: (6.5, 8.0, 0.0)   — end of vertical corridor
```

---

## 5. Training Stages

### PHASE A: Static Maze (Steps 0-8)

#### Stage 0 — Forward Locomotion on Flat Plane
- **Scene:** Plane only, no walls
- **Obs:** Proprioception + dummy nav + dummy scan + dummy history + dummy agent
- **Rewards:** Locomotion only, all obstacle/nav weights = 0
- **Spawn:** Origin, default reset
- **Iterations:** 2000
- **Success:** Walks forward, 4 legs, no jumping, no flipping

#### Stage 1 — Straight Corridor
- **Scene:** Add wall_1, wall_2
- **Obs:** Same (scan still dummy)
- **Rewards:** Same as Stage 0
- **Spawn:** (-2.0, 0.0, 0.0), z_offset=0.6
- **Load from:** Stage 0 best
- **Iterations:** 500-1000
- **Success:** Walks through corridor without freezing

#### Stage 2 — One Chicane
- **Scene:** Add chicane_1
- **Obs:** Activate `obstacle_scan_5` (real), activate `forward_heading_when_clear`, activate `lateral_clearance`
- **Rewards:** Activate obstacle penalties + clearance + heading
- **Spawn:** (-2.0, 0.0, 0.0)
- **Load from:** Stage 1 best
- **Iterations:** 1000-1500
- **Success:** Steers around chicane, continues forward

#### Stage 3 — Two Chicanes (wider spacing)
- **Scene:** Add chicane_2 at x=4.0 (3.5m gap between chicanes)
- **Obs:** Same as Stage 2
- **Rewards:** Same + activate `directed_dodge_when_blocked` and switch to `lateral_motion_when_clear`
- **Spawn:** (-2.0, 0.0, 0.0)
- **Load from:** Stage 2 best
- **Iterations:** 1000-1500
- **Success:** Zigzags through both chicanes

#### Stage 4 — Waypoint Reward System
- **Scene:** Keep two-chicane corridor
- **Obs:** Activate navigation channels (waypoint_angle, heading_proj, distance)
- **Rewards:** Replace `progress` with `waypoint_progress_reward`, remove `forward_heading_when_clear` (waypoint obs handles direction now)
- **Spawn:** (-2.0, 0.0, 0.0)
- **Load from:** Stage 3 best
- **Iterations:** 1500-2000
- **Success:** Follows waypoints, still avoids obstacles

#### Stage 5 — Add Turn Geometry
- **Scene:** Add end_wall_low, turn_wall_left, turn_wall_right. Optionally remove chicanes first.
- **Obs:** Same
- **Rewards:** Same (waypoint now guides through turn)
- **Spawn:** (-2.0, 0.0, 0.0)
- **Load from:** Stage 4 best
- **Iterations:** 2000-3000
- **Success:** Reaches turn, turns corner, continues +y

#### Stage 6 — Multiple Spawn Points
- **Scene:** Full L-maze
- **Obs:** Same
- **Rewards:** Same
- **Spawn:** Gradually add: (-2.0, 0, 0) → add (1.5, 1.3, 0) → add (4.0, -1.3, 0)
- **Load from:** Stage 5 best
- **Iterations:** 1000-1500
- **Success:** Works from multiple starts

#### Stage 7 — Better Sensing (Config Change Only)
- **Scene:** Full L-maze
- **Obs:** Tune ray angles to (0°, 30°, -30°, 60°, -60°), increase max_range to 8.0
- **Rewards:** Same
- **Spawn:** Multiple
- **Load from:** Stage 6 best
- **Iterations:** 500-1000
- **Success:** Smoother avoidance with better angular coverage
- **NOTE:** Same 5 rays, same obs dimension, checkpoints load fine

#### Stage 8 — Full Static Maze + Polish
- **Scene:** Full L-maze with all chicanes
- **Obs:** Same
- **Rewards:** Same, light domain randomization on friction and mass
- **Spawn:** All spawn points including post-turn spawns
- **Load from:** Stage 7 best
- **Iterations:** 1500-2000
- **Success:** Reliably navigates entire maze from any spawn. PHASE A COMPLETE.

---

### PHASE B: Moving Obstacles (Steps 9-10)

**Prerequisites:**
- Stage 8 checkpoint (static maze solved)
- New code: dynamic AABB updates every step (replace cached `_obs_wall_cache`)
- New code: moving obstacle spawning and trajectory control

#### Stage 9 — Single Slow-Moving Obstacle
- **Scene:** Full L-maze + one moving cuboid (0.1-0.3 m/s lateral oscillation)
- **Obs:** Activate `scan_t_minus_1` and `scan_t_minus_2` (frame history). The 3 stacked frames give implicit velocity information.
- **Rewards:** Same + increase dodge weight slightly
- **Spawn:** (-2.0, 0.0, 0.0) only
- **Load from:** Stage 8 best
- **Iterations:** 2000-3000
- **Sub-steps:**
  - 9a: Moving obstacle in straight section only, slow speed (0.1 m/s)
  - 9b: Increase speed to 0.3 m/s
  - 9c: Randomize speed between 0.1-0.3 m/s
- **Success:** Ant dodges the moving obstacle and continues navigating

#### Stage 10 — Multiple Moving Obstacles
- **Scene:** Full L-maze + 2-3 moving cuboids
- **Obs:** Same (frame stacking already active)
- **Rewards:** Same
- **Spawn:** Multiple
- **Load from:** Stage 9c best
- **Iterations:** 3000-5000
- **Sub-steps:**
  - 10a: 2 slow obstacles in corridor section
  - 10b: 2 obstacles, randomized speed 0.1-0.5 m/s
  - 10c: 3 obstacles, full randomization
- **Success:** Ant navigates maze with multiple moving obstacles. PHASE B COMPLETE.

---

### PHASE C: Multi-Agent (Steps 11-12)

**Prerequisites:**
- Stage 10 checkpoint (moving obstacle avoidance)
- New code: `agent_scan_5` that detects articulated bodies (other ants)
- New code: multi-agent environment with shared scene
- Decision: Independent PPO per agent OR centralized critic (MAPPO)

#### Stage 11 — Two Ants, Same Maze
- **Scene:** Full L-maze, two ants
- **Obs:** Activate `agent_scan_5` (distinguishes walls from other agents)
- **Rewards:** Same + collision penalty between agents
- **Spawn:** Opposite ends of maze
- **Load from:** Stage 10 best
- **Iterations:** 3000-5000
- **Sub-steps:**
  - 11a: Two ants, no moving obstacles, opposite spawns
  - 11b: Two ants, one moving obstacle
  - 11c: Two ants, full moving obstacles
- **Success:** Both ants navigate without colliding with each other

#### Stage 12 — Multiple Ants
- **Scene:** Full L-maze, 3-4 ants
- **Obs:** Same
- **Rewards:** Same
- **Spawn:** Various positions
- **Load from:** Stage 11c best
- **Iterations:** 5000+
- **Success:** Multiple ants coexist and navigate the maze. PHASE C COMPLETE.

---

## 6. Code Infrastructure Needed Before Starting

### Day 1 — Before Any Training
1. `custom_mdp.py` — ALL reward and observation functions, including future ones (with dummy paths)
2. Stage 0 config with FULL observation space (all dummies active)
3. Task registration for all 13 stages
4. Verify obs dimension matches across all configs

### Before Phase B (after Stage 8)
5. Dynamic AABB update system (replace static cache)
6. Moving obstacle spawner + trajectory controller
7. Frame history buffer for scan stacking

### Before Phase C (after Stage 10)
8. Agent detection scan (articulated body awareness)
9. Multi-agent environment wrapper
10. Inter-agent collision penalty

---

## 7. Estimated Timeline

| Phase | Stages | Training Time | Code Time | Total |
|-------|--------|--------------|-----------|-------|
| A: Static Maze | 0-8 | 8-12 hours | 4-6 hours | 2-3 days |
| B: Moving Obstacles | 9-10 | 8-15 hours | 4-8 hours | 2-4 days |
| C: Multi-Agent | 11-12 | 15-25 hours | 8-12 hours | 3-5 days |
| **Total** | **0-12** | **31-52 hours** | **16-26 hours** | **7-12 days** |

*Assumes prior experience from the first project. First-time would be 2-3x longer.*

---

## 8. Key Lessons From First Project (Do Not Repeat)

1. **Never make standing more profitable than walking** — keep alive/upright/contact total < 0.3
2. **Conditional penalties** — lateral_motion and forward_heading must suspend near obstacles
3. **Check geometry before blaming rewards** — the zigzag failure was a spacing problem
4. **Stop early, check video** — don't run 3000 iterations before looking at behavior
5. **Observation dimension must be fixed from Step 0** — include ALL dummy channels upfront
6. **Regularization must be light** — action_l2, energy, joint_limits should never exceed -0.01 each
7. **Chicane spacing must match dodge capability** — 3.5m minimum between opposite-side obstacles
8. **Front detection range must exceed stopping distance** — front_threshold should be 3.5m, not 1.5m
