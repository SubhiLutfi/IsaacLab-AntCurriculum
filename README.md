# IsaacLab Ant Curriculum

Training a MuJoCo Ant to walk, dodge obstacles, and chase moving waypoints — through a 9-stage reinforcement learning curriculum built on top of [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab).

> **Status:** Phase A (static obstacles) complete at Stage 9. Phase B (moving obstacles) in progress.

**Final Stage 9 policy — open arena, 15 random obstacles, cycling waypoint:**

<!-- VIDEO 9: Final stage 9 result. Paste the github user-attachments URL on the line below. -->
<!-- PASTE_URL_VIDEO_9_FINAL_STAGE9 -->

---

## What this is

This is my fork of IsaacLab where I built a curriculum that takes a stock 8-DoF MuJoCo Ant from "lying on the floor twitching" all the way to "navigating an open arena scattered with 15 random obstacles to reach a continuously respawning waypoint." Everything trains with PPO via RSL-RL on 4096 parallel environments at 120 Hz physics / 60 Hz policy.

The interesting part isn't really the final policy — it's the curriculum design and the architectural decisions that let one network checkpoint transfer cleanly across nine very different environments.

## Why I built it

I wanted a real, end-to-end project to learn how RL curricula are actually shaped in practice — not the toy version where each stage is a clean jump, but the messy version where you watch a video, realize the reward is wrong, and back up two stages. The longer-term goal is to transfer the same curriculum to a Unitree Go2, where the locomotion changes but the navigation and avoidance logic should carry over directly.

## The plan vs. what actually happened

I started with a fairly detailed 13-stage plan: an L-shaped maze with two chicanes and a 90-degree turn, fixed waypoints guiding the ant through it, and a clean progression from flat-plane locomotion to moving obstacles to multi-agent. That plan is still in the repo as `PROJECT_PLAN.md`. About a third of it survived contact with reality.

### Stage 0: the three-legged tilted ant

The very first version of Stage 0 had a single reward: velocity in the +x direction. The ant figured out that the cheapest way to maximize +x velocity was to tilt sideways and shuffle on three legs, which is technically forward motion but is also extremely funny to watch and absolutely useless for everything that comes after.

[<!-- VIDEO 1: Stage 0 broken — three-legged tilted ant. -->
<!-- PASTE_URL_VIDEO_1_STAGE0_BROKEN -->](https://github.com/user-attachments/assets/265bd73a-4868-4174-8a72-e026c32ca28a)

The fix was a stack of conditional rewards and penalties: an upright-posture bonus, a feet-contact-count reward, an airborne penalty for lifting more than two feet at once, a roll penalty for tilting sideways, and a vertical-velocity penalty to kill jumping. Together they make four-legged grounded walking the only profitable strategy. The single most important rule from this stage, and one I keep coming back to: standing must never be more profitable than walking. The alive + upright + contact bonuses combined have to stay under 0.3, or the ant just stands there collecting income.

<!-- VIDEO 2: Stage 0 working — clean four-legged forward walk. -->
<!-- PASTE_URL_VIDEO_2_STAGE0_WORKING -->

### Stages 2 and 3: hitting walls before learning to dodge

The first time I added an obstacle, the ant walked straight into it. Of course it did — it had spent 2000 iterations learning that +x velocity is the only thing that matters. The obstacle scan and proximity penalties were live, but the policy needed time to incorporate them into the value function. After a few hundred iterations of head-on collisions, it started learning to steer around.

<!-- VIDEO 3: Stage 2 — ant hits the wall the first time, then passes the barrier. -->
<!-- PASTE_URL_VIDEO_3_STAGE2_HITS_THEN_PASSES -->

By the time Stage 4 was running, it had paired the avoidance with waypoint following on the single-barrier scene:

<!-- VIDEO 4: Stage 4 — ant navigating to a waypoint past one barrier. -->
<!-- PASTE_URL_VIDEO_4_STAGE4_WAYPOINT_ONE_BARRIER -->

### Stages 3, 4 and 5: where fixed waypoints broke down

Stages 3 through 5 added the second chicane, the fixed waypoint sequence, and the L-turn from the original plan. They worked, in the narrow sense that the ant did the thing the rewards asked for. But two videos made it obvious the approach was a dead end.

The first was the zigzag through the two-chicane corridor:

<!-- VIDEO 5a: Stage 3/4 zigzag — ant memorizing the chicane sequence. -->
<!-- PASTE_URL_VIDEO_5A_ZIGZAG -->

The second was the 90-degree turn at the end of the L-maze:

<!-- VIDEO 5b: Stage 5 turn — ant struggling with the fixed waypoint at the corner. -->
<!-- PASTE_URL_VIDEO_5B_TURN -->

In both cases the ant wasn't *navigating*. It was memorizing. The waypoints were fixed, the obstacles were fixed, and the policy was overfitting to the specific geometry. The moment I imagined moving any of it, I knew the ant would fail — and worse, I'd be debugging an overfit policy without knowing whether the navigation logic was correct in the first place. This is what pushed me to Stage 4.5.

### Stage 4.5: the pivot to random targets

I replaced the entire fixed-waypoint system with a single cycling random target. The waypoint respawns 2–8 m away in a random direction the moment the ant reaches it (`reach_radius=0.8 m`). One waypoint, infinite variety. From this point on, every stage uses the same random-target system, and the ant has to actually learn to navigate rather than to memorize.

<!-- VIDEO 6: Stage 4.5 — ant chasing cycling random targets in clean space. -->
<!-- PASTE_URL_VIDEO_6_STAGE4_5_RANDOM_TARGETS -->

This was the moment the project clicked. The ant immediately got harder to train (no more memorization shortcut) but the resulting policy was something I trusted.

### The L-maze stages, then dropping the L-maze

I did keep going with the maze geometry for a while after introducing random targets. The ant could still handle the L-turn:

<!-- VIDEO 7: Stage 5/6 with random targets — ant navigating the L-turn. -->
<!-- PASTE_URL_VIDEO_7_L_TURN_RANDOM -->

And in parallel, all 4096 envs running the full L-maze look surprisingly orderly:

<!-- VIDEO 8: Stage 6/7 — many ants navigating the full L-maze in parallel. -->
<!-- PASTE_URL_VIDEO_8_L_MAZE_PARALLEL -->

But once the random-target system was working, the maze geometry was actively making things harder without teaching anything new. An open arena with random obstacles is both simpler to reason about and harder for the policy, which is what you want from a curriculum. I dropped the L-maze entirely and moved to an open flat arena for Stages 8 and 9.

### The AABB cache went stale the moment anything moved

Around the same time I was rewriting the obstacle scan. My first implementation built a one-time cache of axis-aligned bounding boxes for the kinematic walls in env 0 and reused it forever. Fast, worked perfectly for static walls. The moment I started thinking about Phase B (moving obstacles) I realized the whole approach was a dead end — the cache had no way to know an obstacle had moved. I rewrote the scan as `kinematic_object_scan`, which reads `root_pos_w` from each RigidObject every timestep. Slightly more expensive, but it works identically for static and moving obstacles, which means Phase B will need zero changes to the observation pipeline.

### The orbital behavior bug at Stage 9

After Stage 8 (6 obstacles) converged cleanly at ~8200 iterations, I scaled up to Stage 9 (15 obstacles) and trained until model_8500. The ant looked great in aggregate metrics — episode length pinned at the maximum, reward climbing — but when I actually watched the video it was *circling obstacles* instead of going to the waypoint. It would pick a nearby obstacle and orbit it like a satellite until the episode ended.

The root cause took a while to find. Random targets were spawning uniformly across the arena, which meant they sometimes landed inside or just behind a tight obstacle cluster, often within 1.5 m of an obstacle's center. From the ant's perspective, the progress reward was pulling it toward the target while the proximity penalty was pushing it sideways from the obstacle right next to the target — and the equilibrium of those two forces is, geometrically, an orbit. The fix was two-part: enforce `min_target_obs_dist=2.5 m` so targets never spawn glued to an obstacle, and soften the proximity penalty (weight -5.0 → -2.5, threshold 0.7 → 0.4 m) so it reacts later and less violently. I also added an obstacle-biased target sampler that *prefers* spawning targets near obstacle clusters (but not inside them) — otherwise, since open space is larger than obstructed space, uniform sampling means the ant rarely practices threading through the dense zones.

The Stage 9 video at the top of this README is the post-fix policy.

## The curriculum

| Stage | What changes | Notes |
|-------|--------------|-------|
| 0 | Forward locomotion on flat plane | Baseline gait. No walls, no obstacles. |
| 1 | Add corridor walls | Learn to walk between things without bouncing off. |
| 2 | One static obstacle | Activates the real obstacle scan and obstacle penalties. |
| 3 | Two static obstacles | First taste of having to plan around something. |
| 4 | Fixed waypoint sequence | Original plan — kept for reference, superseded by 4.5. |
| 4.5 | Single cycling random target | The pivot. Target respawns 2–8 m away on reach. |
| 5 | Targets that require turning back | Tests heading control under large angular changes. |
| 6 | Multiple spawn positions | Generalize to different starting states. |
| 7 | Fully random spawn pose | Maximum spawn generalization before obstacles return. |
| 8 | Open arena, 6 random static obstacles | Cycling target, episode 96 s. Converged at ~8200 iterations. |
| 9 | Open arena, 15 random static obstacles | Obstacle-biased target spawning. Best checkpoint: model_8500. |

## Architecture decisions that mattered

**The fixed 84-dimensional observation vector is the single most important decision in the whole project.** RSL-RL ties checkpoint compatibility to network input size, so the moment you change the observation dimension between stages, you can no longer warm-start from the previous stage's policy. I designed the full observation vector on day one — proprioception, waypoint channels, obstacle scan, scan history, agent scan — and made the early stages emit dummy values (zeros or ones) for the channels that weren't yet active. The result is that I can take the Stage 0 checkpoint and load it directly into Stage 9, or back-port a Stage 9 fix into Stage 4 without retraining from scratch. Every stage transition in this project was a `--load_checkpoint` away from instant warm-start, and that saved a huge amount of training time.

**Kinematic scan instead of cached AABBs** — covered above. The lesson is the same one I keep relearning in robotics: caching is an optimization, and optimizations are commitments. The AABB cache was committing me to a static world.

**Obstacle-biased target spawning** is the kind of fix you only think of after watching your agent fail. Uniform random sampling sounds fair, but in an arena where 80% of the area is open and 20% is around obstacles, uniform sampling means 80% of the training signal comes from clear-zone navigation. I now sample 16 candidate targets per respawn, score each by how close it is to obstacle clusters (with a hard floor of 2.5 m to avoid the orbiting trap), and softmax-sample.

**Reward shaping lessons.** Three things from the first iteration that I am never going to relearn the hard way: standing must never be more profitable than walking (alive + upright + contact totals stay under 0.3), penalties like `lateral_motion` and `forward_heading` have to suspend themselves when there's an obstacle ahead (otherwise the ant gets penalized for the dodge it's supposed to do), and regularization terms (`action_l2`, energy, joint limits) all stay under -0.01. Heavy regularization in early stages is the single fastest way to train a policy that just stands still and collects the alive bonus.

## Observation vector

| Term | Dims | Description |
|---|---|---|
| `base_height` | 1 | Robot base Z position |
| `base_lin_vel` | 3 | Linear velocity (world frame) |
| `base_ang_vel` | 3 | Angular velocity |
| `base_yaw_roll` | 2 | Yaw and roll angles |
| `base_angle_to_target` | 1 | Angle to current waypoint |
| `base_up_proj` | 1 | Upright projection (tilt measure) |
| `base_heading_proj` | 1 | Heading alignment to target |
| `joint_pos_norm` | 8 | Normalized joint positions |
| `joint_vel_rel` | 8 | Relative joint velocities (×0.2) |
| `feet_body_forces` | 24 | Incoming wrench at 4 feet (×0.1) |
| `waypoint_angle` | 1 | Angle to waypoint |
| `waypoint_heading` | 1 | Heading alignment to waypoint |
| `waypoint_dist` | 1 | Normalized distance to waypoint |
| `obstacle_scan` | 5 | 5-ray kinematic scan (0°, ±45°, ±90°) |
| `scan_history` | 10 | 2 frames × 5 rays (dummy until Phase B) |
| `agent_scan` | 5 | Multi-agent scan (dummy until Phase C) |
| `actions` | 8 | Last action (8 joint efforts) |
| **Total** | **84** | |

`scan_history` and `agent_scan` are placeholder dummy channels right now. They exist in the vector so that when Phase B and Phase C activate them, the network shape doesn't change and Phase A checkpoints still load.

## Training infrastructure

Two small helper scripts live at the repo root and are doing more work than they look like.

`convergence_monitor.py` wraps the training process and auto-stops at a clean checkpoint boundary. The criterion is "episode length has been at the maximum for 20 consecutive iterations," but the stop only fires when the iteration count is a multiple of 100 — because RSL-RL only saves checkpoints every 100 iterations, and stopping mid-window throws away the model. This sounds trivial but it has saved me hours of "oh no I killed it one iteration before the save."

`patch_noise_std.py` resets `noise_std` directly in a `.pt` checkpoint when starting a new curriculum stage. New stages have different reward scales, which means the action noise the policy learned in the previous stage is usually too aggressive for the new one and produces unstable early training. Patching the noise back down before the resume gives the new stage a clean start without losing the learned weights.

Training Stage 9 from a Stage 8 checkpoint:

```bash
python convergence_monitor.py python scripts/rsl_rl/train.py \
  --task Isaac-Ant-Stage9-RandomObstacles-v0 \
  --num_envs 4096 \
  --headless \
  --load_checkpoint logs/rsl_rl/humanoid/STAGE8_RUN/model_8200.pt \
  --run_name ant_stage9_15obs \
  --max_iterations 15000
```

Recording a video of a trained policy:

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Ant-Stage9-RandomObstacles-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/humanoid/RUN_DIR/model_8500.pt \
  --video --video_length 500
```

## Repository layout

The repo is a full IsaacLab fork, so most of the tree is upstream code. The files that are mine:

```
convergence_monitor.py                          # auto-stop training at checkpoint boundary
patch_noise_std.py                              # reset noise_std for stage transitions
PROJECT_PLAN.md                                 # original 13-stage plan (historical)
source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/humanoid/
├── __init__.py                                 # gym environment registrations
├── custom_mdp.py                               # all custom MDP functions (~1400 lines)
├── ant_stage0_forward_env_cfg.py
├── ant_stage1_corridor.py
├── ant_stage2_one_obstacle.py
├── ant_stage3_two_obstacles.py
├── ant_stage4_waypoints.py
├── ant_stage4_5_random_targets.py
├── ant_stage5_turn.py
├── ant_stage6_multi_spawn.py
├── ant_stage7_random_spawn.py
├── ant_stage8_random_obstacles.py
└── ant_stage9_random_obstacles.py
```

Registered Gym IDs follow the pattern `Isaac-Ant-StageN-<Name>-v0`, e.g. `Isaac-Ant-Stage9-RandomObstacles-v0`.

## Pretrained checkpoints

<!-- TODO: replace this with the actual GitHub Release URL once you tag v1.0-phase-a -->

Phase A checkpoints are attached to the GitHub Release for this repo. The two worth grabbing:

- **Stage 8** — `model_8200.pt` — best behavior on the 6-obstacle arena, ~8200 iterations.
- **Stage 9** — `model_8500.pt` — best behavior on the 15-obstacle arena, after the orbital-behavior fix.

Drop them under `logs/rsl_rl/humanoid/<run_name>/` and pass the path to `--checkpoint`.

## What's next

Phase B is moving obstacles. The good news is that the kinematic scan already handles moving objects — `kinematic_object_scan` reads positions every step, so all I need is an event that updates obstacle positions during the episode rather than only on reset. The `scan_history` channel is already in the observation vector waiting to be activated; once it's live, the policy gets two-frame velocity information for free.

Phase C (multi-agent) is further out and will require activating the `agent_scan` channel and a multi-agent environment wrapper. That's its own project.

Eventually I want to swap `ANT_CFG` for `UNITREE_GO2_CFG`, adjust the action scale and joint names, and re-run the curriculum. The whole point of designing the rewards and observation vector to be robot-agnostic is to make that swap cheap.

## Acknowledgements

This project is built on top of [NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab), which does all of the heavy lifting — physics, rendering, RL pipelines, robot assets. Everything I added is small in comparison. Thanks to the IsaacLab team for making a framework that makes this kind of project possible for one person on a university workstation.

## License

This is a fork of IsaacLab and inherits its licensing: BSD-3-Clause for the framework, Apache-2.0 for `isaaclab_mimic`. My additions (the stage configs, `custom_mdp.py`, the helper scripts) are released under the same BSD-3-Clause license as the rest of the framework.
