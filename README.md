# BLIND: Bipedal Locomotion with Intermittent Navigation Data for Environmental Hazards

BLIND (*Bipedal Locomotion with Intermittent Navigation Data for Environmental Hazards*) is a reinforcement learning framework based on NVIDIA Isaac Gym, designed to train robust locomotion policies for humanoid robots (specifically RobotEra's XBot-L) under external hazards and internal failures. This codebase builds directly upon the foundational architecture established in [*Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer*](https://arxiv.org/abs/2404.05695).

## 1. Project Overview & Motivation
Standard reinforcement learning policies for humanoid locomotion are highly susceptible to out-of-distribution (OOD) disturbances. They typically assume perfect, continuous data from sensors and uninterrupted joint execution. In the real world, physical humanoids experience:
* **Sensor noise and intermittent blackouts** (e.g., thermal sensor failures, camera visual occlusions).
* **Actuator degradation** (e.g., motor saturation, joint limpness or freezing).
* **External environmental hazards** (e.g., collisions, physical obstacles, projectile bombardment).

**BLIND** introduces a framework that models system failures **causally** linked to environmental hazards. By training humanoid agents inside a three-stage curriculum with **"blinking"** (intermittent, randomized masking of sensors, actuators, and radar inputs), we force the policy to learn robust, multi-modal recovery behaviors and a distributed, resilient gait.

## 2. Core Methodologies
Our framework utilizes Proximal Policy Optimization (PPO) in an asymmetric Actor-Critic setup to control a 12-DoF RobotEra XBot-L humanoid robot.

### Asymmetric Actor-Critic
* **Actor Observation (54D $\times$ 15 stacked frames = 810D)**: Proprioceptive joint states, base orientation, command vectors, and a 7D synthetic exteroceptive radar tracking incoming threats.
* **Privileged Critic Observation (73D $\times$ 3 frames = 219D)**: Ground-truth states (e.g., actual velocities, contact forces, domain randomization parameters) to stabilize value estimation.

### Procedural Projectiles
Procedural 3 kg projectiles spawn dynamically on a 2.0 m radius cylinder centered around the robot's Center of Mass (CoM). They are fired at $7.5\text{ m/s}$ directly targeting the CoM, creating uniform impact coverage across all body segments.

### "Blinking" Failure Modes
To teach the policy how to handle intermittent outages, we introduce three masking conditions:
1. **Proprioceptive Sensor Dropout**: Zeroes out the velocity feedback of a randomly selected joint in the actor observation for $50$ steps ($0.5\text{ s}$).
2. **Exteroceptive Radar Blackout**: Zeroes out the 7D radar tracking vector for $50$ steps, leaving the robot blind to incoming projectiles.
3. **Actuator Limpness**: Sets the PD control gains ($K_p, K_d$) of a randomly selected joint to zero for $30$ steps ($0.3\text{ s}$), rendering the joint completely floppy.

Blinking is triggered via two concurrent mechanisms:
* **Stochastic background blinks**: Random activations during training based on configured step probabilities ($p_{\text{sensor}} = 0.01$, $p_{\text{radar}} = 0.01$, $p_{\text{actuator}} = 0.002$).
* **Causal impact-based blinks**: Projectile collisions exceeding $10\text{ N}$ trigger localized failures (left-leg hits mask left-leg joints, right-leg hits mask right-leg joints, torso/head hits trigger radar blackouts).

## 3. Three-Stage Curriculum Training
Because direct training under high-intensity physical and sensory trauma is unstable, we employ a progressive curriculum:

* **Stage 1: Locomotion Baseline**: The policy is trained from scratch on flat ground without projectiles or blinking failures. This run is trained for 300 iterations (73.7M simulation steps) to establish a stable walking gait using the pre-allocated 54D observation size.
* **Stage 2: Projectile Resilience**: Resuming from the Stage 1 baseline, we activate spherical projectile spawning and impact-triggered failures. The model is trained for 100 iterations (24.6M simulation steps) to adapt to constant physical perturbations.
* **Stage 3: Blinking Training**: Resuming from the Stage 2 checkpoint, the robot is trained under different blinking configurations for 100 iterations (24.6M simulation steps) to learn recovery policies. In Stage 3, the training branches into five parallel configurations starting from the same Stage 2 checkpoint:
  * **Branch A (Control)**: Projectiles remain active, but all blinking failure modes are disabled.
  * **Branch B (Sensor Blink)**: Enables stochastic background proprioceptive joint velocity masking.
  * **Branch C (Actuator Blink)**: Enables stochastic background joint actuator limpness.
  * **Branch D (Radar Blink)**: Enables stochastic background exteroceptive radar masking.
  * **Branch E (Combined Blinking)**: All three stochastic background blinking failure modes are simultaneously enabled.

## 4. Repository Structure
The core logic is implemented in the following modules:
```
blind-humanoid-gym/
├── humanoid/
│   ├── algo/                       # RL Algorithms
│   │   └── ppo/                    # PPO implementation (actor_critic.py, ppo.py, on_policy_runner.py)
│   ├── envs/                       # Task and Environment setups
│   │   ├── base/
│   │   │   ├── legged_robot.py     # Base physics and projectile spawn/impact queries
│   │   │   └── legged_robot_config.py
│   │   └── custom/
│   │       ├── humanoid_config_radar_mask.py # Main configuration file (Stage 3 settings)
│   │       └── humanoid_env_radar_mask.py    # Environment implementation for blinking & radar
│   ├── scripts/                    # Entrypoint execution scripts
│   │   ├── train.py                # Policy training entrypoint
│   │   ├── play.py                 # Evaluation visualizer and video renderer
│   │   └── eval_metrics.py         # Diagnostic benchmark and metrics calculator
│   └── utils/                      # Math, registry, and helper utilities
├── setup.py                        # Dependency configuration
└── README_humanoid_gym.md          # Upstream repository documentation and installation details
```

## 5. Installation Guide
1. Create a Python virtual environment with Python 3.8:
   ```bash
   conda create -n blind-gym python=3.8
   conda activate blind-gym
   ```
2. Install PyTorch 1.13 and CUDA 11.7:
   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
   conda install numpy=1.23
   ```
3. Install NVIDIA Isaac Gym (Preview 4):
   * Download from the [NVIDIA Isaac Gym Website](https://developer.nvidia.com/isaac-gym).
   * Install via pip:
     ```bash
     cd isaacgym/python && pip install -e .
     ```
4. Install this repository and its dependencies (requires `numpy==1.23.5` and `mujoco==2.3.6`):
   ```bash
   pip install -e .
   ```

*(For detailed troubleshooting and hardware/driver compatibility, refer to README_humanoid_gym.md.)*

## 6. Execution and Usage

### Training Policies
To train the policy, execute the `train.py` script. The task is registered under `humanoid_ppo_radar_mask` (utilizing the `XBotLRadarMaskEnv` class and the configuration `XBotLCfgRadarMask`). Note that `--task humanoid_ppo` and `--task humanoid_ppo_radar_mask` are identical and fully interchangeable:

* **Train Baseline from Scratch (Stage 1)**:
  ```bash
  python humanoid/scripts/train.py --task humanoid_ppo_radar_mask --run_name <baseline_run_name> --headless --num_envs 4096
  ```
* **Resume/Curriculum Transfer (Stage 2)**:
  To load a saved run and continue training with projectiles enabled (loading a saved baseline checkpoint):
  ```bash
  python humanoid/scripts/train.py --task humanoid_ppo_radar_mask --resume --load_run <baseline_log_dir_name> --checkpoint <checkpoint_number> --run_name <stage2_run_name> --max_iterations 100 --projectiles True --impact_failures True --headless --num_envs 4096
  ```
  *(Note: Replace `<baseline_log_dir_name>` with the directory name under `logs/XBot_ppo/` containing your baseline run, e.g., `<date_time>_<run_name>`. Set `--checkpoint -1` to load the latest saved checkpoint.)*
* **Train Blinking Branches (Stage 3)**:
  To run a Stage 3 blinking branch starting from the Stage 2 checkpoint:
  ```bash
  python humanoid/scripts/train.py --task humanoid_ppo_radar_mask --resume --load_run <stage2_log_dir_name> --checkpoint <checkpoint_number> --run_name <stage3_run_name> --max_iterations 100 --projectiles True --impact_failures True --blink_actuators True --headless --num_envs 4096
  ```
  *(Note: Replace `--blink_actuators True` with other failure mode flags as needed: `--blink_sensors True` or `--blink_radar True`.)*

### Evaluating & Rendering
To render a checkpoint rollout in real-time or export an `.mp4` video (saved in `videos/`):
```bash
python humanoid/scripts/play.py --task humanoid_ppo_radar_mask --run_name <trained_run_name>
```
*(Note: `play.py` automatically sets `resume = True` and loads the latest checkpoint in the specified run directory.)*

### Metrics Diagnostics
To compile performance statistics (Mean Reward, Survival Time, Fall Rate, etc.) over a 100-episode sweep, you **must** specify either `--resume` (to load the latest checkpoint of a run) or a direct `--checkpoint-path`:
* **Using run name**:
  ```bash
  python humanoid/scripts/eval_metrics.py --task humanoid_ppo_radar_mask --run_name <trained_run_name> --resume --episodes 100
  ```
* **Using direct checkpoint path**:
  ```bash
  python humanoid/scripts/eval_metrics.py --task humanoid_ppo_radar_mask --checkpoint-path logs/XBot_ppo/<run_dir_name>/model_<checkpoint_number>.pt --episodes 100
  ```
