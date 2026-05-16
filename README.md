# Reinforcement Learning for TCP Congestion Control

This project implements a reinforcement learning based TCP congestion-control prototype. The main goal is to train an RL agent to control the TCP congestion window under different network conditions and compare its behavior against traditional rule-based TCP control strategies such as Reno and CUBIC-inspired baselines.

The project supports two execution modes:

1. A lightweight local bottleneck-link simulator for fast training and validation.
2. An optional ns3-gym based environment for experiments connected to ns-3.

The current implementation focuses on PPO-based control, with additional support for a custom Double DQN implementation.

---

## Project Overview

TCP congestion control needs to balance high throughput, low delay, stable queue behavior, and low packet loss. Traditional algorithms such as Reno and CUBIC use hand-designed rules to adjust the congestion window. In this project, the congestion-control decision is formulated as a reinforcement learning problem.

At each step, the agent observes compact network feedback, including RTT, loss rate, throughput, queue occupancy, congestion window size, and related TCP state features. The agent then selects one of five discrete congestion-control actions:

| Action ID | Action Name | Meaning |
|---|---|---|
| 0 | `cubic_backoff` | Multiplicative backoff similar to CUBIC-style reduction |
| 1 | `mild_decrease` | Slight congestion-window decrease |
| 2 | `keep` | Keep the current congestion window |
| 3 | `additive_increase` | Conservative additive probing |
| 4 | `aggressive_probe` | More aggressive bandwidth probing |

The reward function encourages high throughput while penalizing excessive delay, packet loss, queue buildup, cwnd error, and unstable oscillation.

---

## Main Features

- Reinforcement learning based TCP congestion-control prototype
- PPO training using Stable-Baselines3
- Custom Double DQN implementation with replay buffer and target network
- Local Gymnasium-compatible bottleneck-link simulator
- Optional ns3-gym adapter for ns-3 based experiments
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
