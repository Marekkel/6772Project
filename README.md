# Reinforcement Learning for TCP Congestion Control

This project implements a reinforcement learning based TCP congestion-control prototype. The goal is to train an RL agent to adjust the TCP congestion window under different network conditions and compare its behavior with traditional TCP congestion-control baselines such as Reno and CUBIC-inspired control.

The project supports two execution modes:

1. A lightweight local bottleneck-link simulator for fast training and validation.
2. An optional `ns3-gym` based environment for experiments connected to ns-3.

The current implementation mainly focuses on PPO-based control, with additional support for a custom Double DQN implementation.

---

## Project Overview

TCP congestion control needs to balance high throughput, low delay, stable queue behavior, and low packet loss. Traditional algorithms such as Reno and CUBIC use hand-designed rules to adjust the congestion window. In this project, the congestion-control process is formulated as a reinforcement learning problem.

At each step, the agent observes compact network feedback, including RTT, loss rate, throughput, queue occupancy, congestion window size, and other TCP state features. The agent then selects one of five discrete congestion-control actions.

| Action ID | Action Name | Description |
|---|---|---|
| 0 | `cubic_backoff` | Multiplicative backoff similar to CUBIC-style reduction |
| 1 | `mild_decrease` | Slight congestion-window decrease |
| 2 | `keep` | Keep the current congestion window |
| 3 | `additive_increase` | Conservative additive probing |
| 4 | `aggressive_probe` | More aggressive bandwidth probing |

The reward function encourages high throughput while penalizing excessive delay, packet loss, queue buildup, cwnd error, and unstable oscillation.

---

## Features

- Reinforcement learning based TCP congestion-control prototype
- PPO training using Stable-Baselines3
- Custom Double DQN implementation with replay buffer and target network
- Local Gymnasium-compatible bottleneck-link simulator
- Optional `ns3-gym` adapter for ns-3 based experiments
- Compact 9-dimensional observation interface
- Five-action CUBIC-oriented discrete control interface
- Evaluation against Reno and CUBIC-inspired baselines
- Multi-seed validation scripts
- Multiple network presets:
  - `clean`
  - `bottleneck`
  - `lossy`
  - `long-rtt`
  - `aqm`
  - `multi-flow`

---

## Repository Structure

```text
6772Project/
├── train_tcp_rl_cubic_tuned.py      # Main training and evaluation script
├── requirements.txt                 # Python dependencies
├── run_validate_local.sh            # Multi-seed validation on local simulator
├── run_validate_presets.sh          # Multi-preset validation script
├── outputs_bottleneck/              # Saved bottleneck-condition results
│   ├── evaluation.csv
│   ├── evaluation_metrics.png
│   └── validation/
├── outputs_multicondition/          # Saved multi-condition results
│   ├── evaluation.csv
│   ├── evaluation_metrics.png
│   └── preset_validation/
└── .gitignore
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/Marekkel/6772Project.git
cd 6772Project
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
gym
gymnasium
matplotlib
numpy
stable-baselines3
torch
```

---

## Quick Start

Train and evaluate PPO on the local simulator:

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode all \
  --timesteps 50000 \
  --episodes 3 \
  --seed 7
```

This command trains a PPO agent and then evaluates it against Reno and CUBIC-inspired baselines.

The default outputs are saved under:

```text
outputs/evaluation.csv
outputs/evaluation_metrics.png
```

---

## Usage

### Train only

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode train \
  --timesteps 100000 \
  --seed 0 \
  --model-path outputs/models/ppo_seed_0.zip
```

### Evaluate only

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode eval \
  --episodes 5 \
  --seed 0 \
  --model-path outputs/models/ppo_seed_0.zip
```

### Train and evaluate DDQN

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ddqn \
  --mode all \
  --timesteps 100000 \
  --episodes 5 \
  --seed 0 \
  --model-path outputs/models/ddqn_seed_0.pt
```

### Resume PPO training

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode train \
  --timesteps 50000 \
  --resume-from outputs/models/ppo_seed_0.zip \
  --model-path outputs/models/ppo_seed_0_resumed.zip
```

---

## Multi-Seed Validation

To run PPO validation over multiple random seeds on the local simulator:

```bash
bash run_validate_local.sh
```

This script runs several seeds, saves trained models, logs, CSV files, and metric plots.

Expected output directory:

```text
outputs/validation/
```

---

## Multi-Condition Validation

To evaluate the agent under multiple network presets:

```bash
bash run_validate_presets.sh
```

This script evaluates different network settings such as clean, bottleneck, long-RTT, lossy, and AQM conditions.

Expected output directory:

```text
outputs/preset_validation/
```

---

## Optional ns3-gym Mode

The script can also run with an `ns3-gym` backed environment.

Before using this mode, make sure that ns-3 and `ns3-gym` are installed and that the active Python environment can import `ns3gym`.

Example installation path:

```text
~/ns3-gym-workspace/ns-allinone-3.40/ns-3.40/contrib/opengym/model/ns3gym
```

Install `ns3-gym` into the active Python environment:

```bash
pip install ~/ns3-gym-workspace/ns-allinone-3.40/ns-3.40/contrib/opengym/model/ns3gym
```

Run PPO with `ns3-gym`:

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env ns3 \
  --algo ppo \
  --mode all \
  --network-preset bottleneck \
  --timesteps 100000 \
  --episodes 5 \
  --seed 0 \
  --model-path outputs/models/ppo_ns3_bottleneck_seed_0.zip
```

If your ns-3 path is different, pass it manually:

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env ns3 \
  --ns3-scenario-dir /path/to/ns-3/contrib/opengym/examples/rl-tcp \
  --algo ppo \
  --mode all
```

---

## Network Presets

| Preset | Description |
|---|---|
| `clean` | High-bandwidth, low-delay, no-loss setting |
| `bottleneck` | Lower bottleneck bandwidth with moderate RTT |
| `lossy` | Bottleneck setting with random packet loss |
| `long-rtt` | Higher-delay network condition |
| `aqm` | Bottleneck setting using CoDel queue discipline |
| `multi-flow` | Multiple leaf flows sharing the bottleneck |

Example:

```bash
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode all \
  --network-preset long-rtt \
  --timesteps 100000 \
  --episodes 5
```

---

## Evaluation

During evaluation, the script compares three policies:

1. The trained RL agent, such as PPO or DDQN.
2. A Reno-style baseline.
3. A CUBIC-inspired baseline.

The generated CSV file records per-step metrics such as:

- selected action
- reward
- accumulated reward
- congestion window
- RTT
- throughput
- loss rate
- queue occupancy

The generated plot visualizes key metrics over time:

- throughput
- RTT
- loss rate
- congestion window

---

## Output Files

After running training and evaluation, the project may generate:

```text
outputs/
├── evaluation.csv
├── evaluation_metrics.png
└── models/
    └── ppo_tcp_congestion_control.zip
```

For batch validation, the scripts save additional files such as:

```text
outputs/validation/
├── log_seed_0.txt
├── evaluation_seed_0.csv
├── evaluation_seed_0.png
└── ...
```

and:

```text
outputs/preset_validation/
├── log_bottleneck_seed_0.txt
├── evaluation_bottleneck_seed_0.csv
├── evaluation_bottleneck_seed_0.png
└── ...
```

---

## Method Summary

The reinforcement learning formulation is defined as follows:

- **State:** compact TCP and network feedback vector
- **Action:** discrete congestion-window control operation
- **Reward:** weighted objective balancing throughput, delay, loss, queue stability, and cwnd behavior
- **Policy:** PPO MLP policy or DDQN Q-network
- **Baselines:** Reno-style and CUBIC-inspired control policies

The local simulator provides a fast environment for debugging and repeated experiments. The `ns3-gym` mode allows the same agent interface to be tested in a more realistic network simulation environment.

---

## Notes and Limitations

- The local simulator is a simplified bottleneck-link model and does not fully replace ns-3.
- The CUBIC baseline is CUBIC-inspired and implemented over the same five-action interface. It is not the native Linux or ns-3 TcpCubic implementation.
- `ns3-gym` mode requires a correctly installed ns-3 and OpenGym environment.
- Training results can vary across random seeds, so multi-seed validation is recommended.
- The reward function is manually designed and may need further tuning for different network conditions.

---

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train and evaluate PPO locally
python3 train_tcp_rl_cubic_tuned.py \
  --env local \
  --algo ppo \
  --mode all \
  --timesteps 100000 \
  --episodes 5 \
  --seed 0

# 3. Check outputs
ls outputs/

# 4. Run multi-seed validation
bash run_validate_local.sh

# 5. Run multi-condition validation
bash run_validate_presets.sh
```

---

## Acknowledgement

This project is developed as a course project for exploring reinforcement learning based TCP congestion control. It uses Gymnasium-compatible environments, Stable-Baselines3 PPO, PyTorch, and optional `ns3-gym` integration.
