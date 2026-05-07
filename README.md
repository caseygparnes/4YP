# ULSS Control

This repository contains the code for the project **Stability Preserving Control for Ultra-large-scale Systems**.

**Author:** Casey Parnes

This project implements a fixed-wing multi-agent orbiting environment with obstacle avoidance, a nonlinear look-ahead base controller, and a learned residual controller trained using graph-based multi-agent reinforcement learning.

> **Note:** This repository builds on the [Cao and Furieri Scaling-Up-Stability codebase](https://github.com/Mudhdhoo/Scaling-Up-Stability), which itself builds on the [InforMARL codebase](https://github.com/nsidn98/InforMARL).

---

## Overview

The aim of this project is to study stability-preserving learned control for a multi-agent fixed-wing system. The implemented controller combines:

- fixed-wing constant-speed dynamics,
- an orbit-following base controller,
- a learned residual policy,
- graph-based local information aggregation,
- disturbance-aware training,
- evaluation across different numbers of agents and initial conditions.

A pretrained model is included so that the main evaluation and plotting scripts can be run without retraining the full policy from scratch.


---

## Repository Structure

```text
.
├── multiagent/          Environment and scenario definitions
├── onpolicy/            Training code, policies, algorithms, and runners
├── plot_scripts/        Evaluation and plotting scripts
├── pretrained/          Included pretrained model checkpoint
│   └── orbit_mad_final/
├── baselines/           Inherited baseline implementations
├── utils/               Utility scripts
├── environment.yml      Conda environment specification
├── requirements.txt     Python package requirements
└── README.md
```

---

## Installation


```powershell
conda env create -f environment.yml
conda activate stable_gnn_policy
```

If additional packages are required, install them using:

```powershell
pip install -r requirements.txt
```

---

## Pretrained Model

A pretrained model is included at:

```text
pretrained_policy/orbit_mad_final/
```

This folder should contain the saved model files and the configuration needed to run evaluation scripts without retraining.

---

## Quick Start: Evaluate the Pretrained Model

The commands below are written for **Windows PowerShell**.

Set the run directory:

```powershell
$RUN_DIR = ".\pretrained_policy\orbit_mad_final"
```

Plot a multi-agent trajectory:

```powershell
python .\plot_scripts\plot_policy_trajectories.py --run_dir $RUN_DIR
```

Evaluate the trained policy with different numbers of agents:

```powershell
python .\plot_scripts\eval_changing_N_post_training.py --run_dir $RUN_DIR --Ns 4 5 6 7 --episodes 20
```

Plot the orbit and obstacle loss heatmap:

```powershell
python .\plot_scripts\plot_loss_heatmap.py --run_dir $RUN_DIR --save_path "orbit_obstacle_loss_heatmap.png" --resolution 400
```

Test start-angle generalization:

```powershell
python .\plot_scripts\test_generalization_start_angle.py --run_dir $RUN_DIR --original_angle_deg 0 --shifted_angle_deg 30 --x0_std_override 0.0
```

---

## Training

To train a new policy from scratch, run the following command.

The command is written for **Windows PowerShell**. If using Linux or macOS, replace the PowerShell line-continuation character `` ` `` with `` \ ``.

```powershell
python -u onpolicy/scripts/train_mpe.py `
  --project_name "orbit_tensorboard" `
  --env_name "GraphMPE" `
  --algorithm_name "rmappo" `
  --experiment_name "orbit_mad_rewardfix_v1_easyobs" `
  --scenario_name "orbit_graph" `
  --seed 0 `
  --num_agents 4 `
  --num_obstacles 40 `
  --ring1_count 10 `
  --ring2_count 15 `
  --ring3_count 15 `
  --episode_length 200 `
  --num_env_steps 230000 `
  --n_training_threads 1 `
  --n_rollout_threads 8 `
  --ppo_epoch 8 `
  --num_mini_batch 8 `
  --target_mini_batch_size 32 `
  --use_cent_obs "True" `
  --graph_feat_type "relative" `
  --use_mad_policy `
  --use_orbit_base_controller `
  --discrete_action "False" `
  --use_disturbance `
  --disturbance_std 0.10 `
  --disturbance_decay_rate 0.018 `
  --max_edge_dist 160 `
  --v_star 10.0 `
  --dt 0.4 `
  --a_n_max 30 `
  --u_to_an_gain 1.0 `
  --u_range 5.0 `
  --guidance_k 0.1 `
  --delta_bl 35.0 `
  --d_shift 0.0 `
  --orbit_radius 50.0 `
  --orbit_dir 1 `
  --x0_rad 220.0 `
  --x0_std 5.0 `
  --obstacle_layout "rings3" `
  --obstacle_safety_margin 0.0 `
  --obstacle_smooth_margin 10.0 `
  --agent_safety_margin 1.5 `
  --obstacle_cost_mode "max" `
  --m_max_start 0.25 `
  --m_max_final 0.25 `
  --m_warmup_episodes 0 `
  --w_orbit 20.0 `
  --w_align 0.2 `
  --w_obstacle 5.0 `
  --w_obstacle_smooth 0.0 `
  --w_agent_coll 25.0 `
  --w_control 0.001 `
  --use_popart `
  --use_valuenorm "False" `
  --entropy_coef 0.001 `
  --gamma 0.99 `
  --use_eval `
  --eval_interval 20 `
  --eval_episodes 5 `
  --log_interval 5 `
  --save_interval 20 `
  --cuda `
  --use_wandb
```


Training outputs are saved under:

```text
onpolicy/results/
```
---

## TensorBoard

If TensorBoard logs are available, they can be viewed using:

```powershell
tensorboard --logdir .\onpolicy\results --port 6006
```

Then open the local TensorBoard URL shown in the terminal 

To see the training curves for the pretrained model, use:

```powershell
tensorboard --logdir .\pretrained\orbit_mad_final\logs --port 6006
```

---




