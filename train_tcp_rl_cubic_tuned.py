"""PPO TCP congestion-control prototype with optional ns3-gym support.

The default environment is a deterministic, lightweight bottleneck-link
simulator. Pass ``--env ns3`` to use an ns3-gym backed environment from WSL or
Linux while keeping the same PPO training and evaluation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


MODEL_PATH = Path("ppo_tcp_congestion_control.zip")
DDQN_MODEL_PATH = Path("ddqn_tcp_congestion_control.pt")
OUTPUT_DIR = Path("outputs")
DEFAULT_NS3_SCENARIO_DIR = Path(
    "/home/donovan/ns3-gym-workspace/ns-allinone-3.40/ns-3.40/contrib/opengym/examples/rl-tcp"
)
UINT32_MAX = 2**32 - 1

# CUBIC-oriented compact interface used by both the local simulator and ns3 adapter.
# 0 = cubic backoff, 1 = mild decrease, 2 = keep, 3 = additive increase, 4 = aggressive probe.
ACTION_COUNT = 5
OBS_DIM = 9
ACTION_NAMES = {
    0: "cubic_backoff",
    1: "mild_decrease",
    2: "keep",
    3: "additive_increase",
    4: "aggressive_probe",
}
NS3_NETWORK_PRESETS: dict[str, dict[str, str | int | float]] = {
    "clean": {
        "--nLeaf": 1,
        "--error_p": 0.0,
        "--bottleneck_bandwidth": "10Mbps",
        "--bottleneck_delay": "1ms",
        "--access_bandwidth": "50Mbps",
        "--access_delay": "5ms",
        "--queue_disc_type": "ns3::PfifoFastQueueDisc",
    },
    "bottleneck": {
        "--nLeaf": 1,
        "--error_p": 0.0,
        "--bottleneck_bandwidth": "2Mbps",
        "--bottleneck_delay": "10ms",
        "--access_bandwidth": "20Mbps",
        "--access_delay": "20ms",
        "--queue_disc_type": "ns3::PfifoFastQueueDisc",
    },
    "lossy": {
        "--nLeaf": 1,
        "--error_p": 0.0001,
        "--bottleneck_bandwidth": "2Mbps",
        "--bottleneck_delay": "20ms",
        "--access_bandwidth": "20Mbps",
        "--access_delay": "20ms",
        "--queue_disc_type": "ns3::PfifoFastQueueDisc",
    },
    "long-rtt": {
        "--nLeaf": 1,
        "--error_p": 0.0,
        "--bottleneck_bandwidth": "5Mbps",
        "--bottleneck_delay": "50ms",
        "--access_bandwidth": "20Mbps",
        "--access_delay": "50ms",
        "--queue_disc_type": "ns3::PfifoFastQueueDisc",
    },
    "aqm": {
        "--nLeaf": 1,
        "--error_p": 0.0,
        "--bottleneck_bandwidth": "2Mbps",
        "--bottleneck_delay": "10ms",
        "--access_bandwidth": "20Mbps",
        "--access_delay": "20ms",
        "--queue_disc_type": "ns3::CoDelQueueDisc",
    },
    "multi-flow": {
        "--nLeaf": 3,
        "--error_p": 0.0,
        "--bottleneck_bandwidth": "5Mbps",
        "--bottleneck_delay": "10ms",
        "--access_bandwidth": "20Mbps",
        "--access_delay": "20ms",
        "--queue_disc_type": "ns3::PfifoFastQueueDisc",
    },
}

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LinkConfig:
    """Parameters for the local, no-ns3 network model."""

    bandwidth_mbps: float = 10.0
    base_rtt_ms: float = 50.0
    buffer_packets: float = 80.0
    packet_size_bytes: int = 1500
    min_cwnd: float = 1.0
    max_cwnd: float = 300.0
    max_steps: int = 240
    noise_std_ms: float = 1.5

    @property
    def bdp_packets(self) -> float:
        bits_per_rtt = self.bandwidth_mbps * 1_000_000 * (self.base_rtt_ms / 1000)
        return bits_per_rtt / (8 * self.packet_size_bytes)


@dataclass(frozen=True)
class Ns3Config:
    """Connection settings for ns3-gym."""

    scenario_dir: Path = DEFAULT_NS3_SCENARIO_DIR
    port: int = 5555
    step_time: float = 0.5
    start_sim: bool = True
    sim_seed: int = 0
    sim_args: dict[str, str | int | float] | None = None
    debug: bool = False
    max_steps: int = LinkConfig().max_steps
    reward_mode: str = "cwnd-control"
    # More CUBIC-like defaults: prioritize utilization, tolerate moderate queues,
    # but still penalize persistent delay and loss.
    throughput_weight: float = 9.0
    delay_weight: float = 0.6
    loss_weight: float = 4.0
    queue_weight: float = 0.25
    cwnd_error_weight: float = 0.15
    oscillation_weight: float = 0.05
    link: LinkConfig = LinkConfig()


class TcpCongestionEnv(gym.Env):
    """Toy TCP congestion-control environment.

    Observation:
        [
            rtt_norm, loss_rate, throughput_norm, ack_interval_norm,
            cwnd_norm, queue_norm, w_max_norm, time_since_loss_norm,
            cwnd_to_pipe_norm
        ]

    Action:
        0 = CUBIC-style backoff
        1 = mild decrease
        2 = keep cwnd
        3 = additive increase
        4 = aggressive probe
    """

    metadata = {"render_modes": []}

    def __init__(self, config: LinkConfig | None = None):
        super().__init__()
        self.config = config or LinkConfig()
        self.current_step = 0
        self.cwnd = 10.0
        self.w_max = 10.0
        self.steps_since_loss = 0
        self.last_cwnd = 10.0
        self.last_metrics: dict[str, float] = {}

        self.observation_space = spaces.Box(
            low=np.zeros(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_COUNT)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cwnd = 10.0 if options is None else float(options.get("initial_cwnd", 10.0))
        self.cwnd = float(np.clip(self.cwnd, self.config.min_cwnd, self.config.max_cwnd))
        self.w_max = self.cwnd
        self.steps_since_loss = 0
        self.last_cwnd = self.cwnd
        self.last_metrics = self._simulate_feedback()
        self._update_cubic_memory(self.last_metrics)
        return self._make_obs(self.last_metrics), self.last_metrics.copy()

    def step(self, action: int):
        self.current_step += 1
        self._apply_action(int(action))

        metrics = self._simulate_feedback()
        reward = self._compute_reward(metrics)
        self._update_cubic_memory(metrics)
        obs = self._make_obs(metrics)

        terminated = False
        truncated = self.current_step >= self.config.max_steps
        self.last_metrics = metrics

        info = metrics.copy()
        info["reward"] = reward
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> None:
        if action == 0:
            # CUBIC-like multiplicative backoff. Less severe than Reno's 0.5 cut.
            self.cwnd *= 0.70
        elif action == 1:
            # Mild queue-draining action. Useful before actual loss occurs.
            self.cwnd *= 0.90
        elif action == 3:
            # Reno-friendly additive increase.
            self.cwnd += max(1.0, 0.01 * self.cwnd)
        elif action == 4:
            # Aggressive probing when the pipe is under-utilized.
            self.cwnd += max(2.0, 0.08 * self.cwnd)

        self.cwnd = float(np.clip(self.cwnd, self.config.min_cwnd, self.config.max_cwnd))

    def _simulate_feedback(self) -> dict[str, float]:
        cfg = self.config
        bdp = cfg.bdp_packets
        queue_packets = min(max(0.0, self.cwnd - bdp), cfg.buffer_packets)
        queue_delay_ms = queue_packets * cfg.base_rtt_ms / max(bdp, 1.0)
        noise_ms = float(self.np_random.normal(0.0, cfg.noise_std_ms))
        rtt_ms = max(1.0, cfg.base_rtt_ms + queue_delay_ms + noise_ms)

        overflow = max(0.0, self.cwnd - bdp - cfg.buffer_packets)
        loss_rate = np.clip((overflow / max(cfg.buffer_packets, 1.0)) ** 1.2, 0.0, 1.0)

        offered_mbps = self.cwnd * cfg.packet_size_bytes * 8 / (rtt_ms / 1000) / 1_000_000
        throughput_mbps = min(cfg.bandwidth_mbps, offered_mbps) * (1.0 - loss_rate)

        if throughput_mbps <= 1e-6:
            ack_interval_ms = 200.0
        else:
            packets_per_second = throughput_mbps * 1_000_000 / (cfg.packet_size_bytes * 8)
            ack_interval_ms = 1000.0 / packets_per_second

        return {
            "cwnd": self.cwnd,
            "rtt_ms": float(rtt_ms),
            "loss_rate": float(loss_rate),
            "throughput_mbps": float(max(0.0, throughput_mbps)),
            "ack_interval_ms": float(min(200.0, ack_interval_ms)),
            "queue_packets": float(queue_packets),
            "bdp_packets": float(bdp),
        }

    def _make_obs(self, metrics: dict[str, float]) -> np.ndarray:
        cfg = self.config
        pipe_packets = cfg.bdp_packets + cfg.buffer_packets
        return np.array(
            [
                min(metrics["rtt_ms"] / 250.0, 1.0),
                min(metrics["loss_rate"], 1.0),
                min(metrics["throughput_mbps"] / cfg.bandwidth_mbps, 1.0),
                min(metrics["ack_interval_ms"] / 200.0, 1.0),
                min(metrics["cwnd"] / cfg.max_cwnd, 1.0),
                min(metrics["queue_packets"] / cfg.buffer_packets, 1.0),
                min(self.w_max / cfg.max_cwnd, 1.0),
                min(self.steps_since_loss / max(cfg.max_steps, 1), 1.0),
                min(metrics["cwnd"] / max(pipe_packets, 1.0), 1.0),
            ],
            dtype=np.float32,
        )

    def _update_cubic_memory(self, metrics: dict[str, float]) -> None:
        congested = metrics["loss_rate"] > 0.0 or metrics["queue_packets"] > 0.85 * self.config.buffer_packets
        if congested:
            self.w_max = max(metrics["cwnd"], self.last_cwnd)
            self.steps_since_loss = 0
        else:
            self.steps_since_loss += 1
            self.w_max = max(self.w_max, metrics["cwnd"])
        self.last_cwnd = metrics["cwnd"]

    def _compute_reward(self, metrics: dict[str, float]) -> float:
        cfg = self.config
        throughput_util = min(metrics["throughput_mbps"] / cfg.bandwidth_mbps, 1.0)
        delay_ratio = metrics["rtt_ms"] / max(cfg.base_rtt_ms, 1.0)
        queue_ratio = metrics["queue_packets"] / max(cfg.buffer_packets, 1.0)
        loss_penalty = metrics["loss_rate"]

        target_cwnd = cfg.bdp_packets + 0.35 * cfg.buffer_packets
        cwnd_error = abs(metrics["cwnd"] - target_cwnd) / max(target_cwnd, 1.0)
        high_delay_penalty = max(0.0, delay_ratio - 2.0) ** 2
        high_queue_penalty = max(0.0, queue_ratio - 0.75) ** 2

        reward = (
            8.0 * throughput_util
            - 0.6 * max(0.0, delay_ratio - 1.0)
            - 4.0 * loss_penalty
            - 0.25 * queue_ratio
            - 1.5 * high_delay_penalty
            - 3.0 * high_queue_penalty
            - 0.15 * min(cwnd_error, 2.0)
        )

        if throughput_util > 0.90 and loss_penalty == 0.0 and queue_ratio < 0.75:
            reward += 0.5
        if throughput_util < 0.50 and loss_penalty == 0.0 and queue_ratio < 0.50:
            reward -= 0.5

        return float(reward)


class Ns3TcpEnv(gym.Env):
    """Gymnasium adapter around ns3-gym's older Gym API.

    The PPO agent always sees the same compact control interface as the local
    toy environment. If the ns-3 scenario already exposes this exact interface,
    actions are passed through. If it exposes the bundled ns3-gym rl-tcp action
    format ``[ssThresh, cWnd]``, this adapter translates the discrete action
    into that pair using the latest raw TCP observation.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Ns3Config | None = None):
        super().__init__()
        self.config = config or Ns3Config()
        self.current_step = 0
        self.w_max = 10.0
        self.steps_since_loss = 0
        self.last_cwnd = 10.0
        self.last_metrics: dict[str, float] = {}
        self._raw_obs: Any = None
        self._legacy_env: Any = None
        self._legacy_action_space: Any = None

        self.observation_space = spaces.Box(
            low=np.zeros(OBS_DIM, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self._legacy_env = self._make_legacy_env()

    @contextmanager
    def _scenario_cwd(self):
        old_cwd = Path.cwd()
        if self.config.start_sim:
            os.chdir(self.config.scenario_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _make_legacy_env(self):
        try:
            from ns3gym import ns3env
        except ImportError as exc:
            raise RuntimeError(
                "ns3gym is not installed in this Python environment. Activate "
                "the WSL venv and run: pip install "
                "~/ns3-gym-workspace/ns-allinone-3.40/ns-3.40/contrib/opengym/model/ns3gym"
            ) from exc

        if self.config.start_sim and not self.config.scenario_dir.exists():
            raise FileNotFoundError(
                f"ns3-gym scenario directory does not exist: {self.config.scenario_dir}"
            )

        sim_args = self.config.sim_args or {}
        with self._scenario_cwd():
            env = ns3env.Ns3Env(
                port=self.config.port,
                stepTime=self.config.step_time,
                startSim=self.config.start_sim,
                simSeed=self.config.sim_seed,
                simArgs=sim_args,
                debug=self.config.debug,
            )

        self._legacy_action_space = env.action_space
        return env

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None and hasattr(self._legacy_env, "simSeed"):
            self._legacy_env.simSeed = int(seed)

        with self._scenario_cwd():
            raw_obs = self._legacy_env.reset()

        self._raw_obs = raw_obs
        metrics = self._metrics_from_obs(raw_obs)
        self.w_max = metrics["cwnd"]
        self.steps_since_loss = 0
        self.last_cwnd = metrics["cwnd"]
        self.last_metrics = metrics
        self._update_cubic_memory(metrics)
        return self._make_obs(metrics), metrics.copy()

    def step(self, action: int):
        self.current_step += 1
        ns3_action = self._format_action_for_ns3(int(action))
        raw_obs, ns3_reward, done, extra_info = self._legacy_env.step(ns3_action)

        self._raw_obs = raw_obs
        previous_metrics = self.last_metrics.copy()
        metrics = self._metrics_from_obs(raw_obs)
        metrics.update(self._parse_extra_info(extra_info))
        reward = self._compute_reward(metrics, previous_metrics, int(action), float(ns3_reward))
        self._update_cubic_memory(metrics)
        obs = self._make_obs(metrics)

        truncated = self.current_step >= self.config.max_steps
        terminated = bool(done)
        self.last_metrics = metrics

        info = metrics.copy()
        info["ns3_reward"] = float(ns3_reward)
        info["reward"] = float(reward)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self._legacy_env is not None:
            self._legacy_env.close()
            self._legacy_env = None

    def _format_action_for_ns3(self, action: int) -> int | list[int]:
        if hasattr(self._legacy_action_space, "n") and int(self._legacy_action_space.n) == ACTION_COUNT:
            return action

        ss_thresh, cwnd, segment_size = self._tcp_window_state()
        if action == 0:
            new_cwnd = max(segment_size, int(cwnd * 0.70))
            new_ss_thresh = max(2 * segment_size, int(new_cwnd))
        elif action == 1:
            new_cwnd = max(segment_size, int(cwnd * 0.90))
            new_ss_thresh = max(2 * segment_size, int(new_cwnd))
        elif action == 3:
            new_cwnd = cwnd + segment_size
            new_ss_thresh = max(ss_thresh, int(new_cwnd * 0.70))
        elif action == 4:
            new_cwnd = int(cwnd * 1.08) + segment_size
            new_ss_thresh = max(ss_thresh, int(new_cwnd * 0.70))
        else:
            new_cwnd = cwnd
            new_ss_thresh = ss_thresh

        return [
            self._clamp_tcp_window(new_ss_thresh, segment_size, allow_infinite=True),
            self._clamp_tcp_window(new_cwnd, segment_size, allow_infinite=False),
        ]

    def _tcp_window_state(self) -> tuple[int, int, int]:
        raw = self._raw_array()
        if raw.size >= 7:
            segment_size = max(1, int(raw[6]))
            ss_thresh = self._clamp_tcp_window(raw[4], segment_size, allow_infinite=True)
            cwnd = self._clamp_tcp_window(raw[5], segment_size, allow_infinite=False)
            return ss_thresh, cwnd, segment_size

        cfg = self.config.link
        segment_size = cfg.packet_size_bytes
        cwnd = int(self.last_metrics.get("cwnd", 10.0) * segment_size)
        ss_thresh = max(2 * segment_size, cwnd)
        return (
            self._clamp_tcp_window(ss_thresh, segment_size, allow_infinite=True),
            self._clamp_tcp_window(cwnd, segment_size, allow_infinite=False),
            segment_size,
        )

    def _clamp_tcp_window(self, value: float, segment_size: int, allow_infinite: bool) -> int:
        if not np.isfinite(value):
            return UINT32_MAX if allow_infinite else self.config.link.max_cwnd * segment_size

        int_value = int(value)
        if allow_infinite and int_value >= UINT32_MAX:
            return UINT32_MAX

        max_window = int(self.config.link.max_cwnd * segment_size)
        upper = UINT32_MAX if allow_infinite else max_window
        upper = max(segment_size, min(UINT32_MAX, upper))
        return int(np.clip(int_value, segment_size, upper))

    def _raw_array(self) -> np.ndarray:
        if isinstance(self._raw_obs, dict):
            return np.array(list(self._raw_obs.values()), dtype=np.float64)
        return np.asarray(self._raw_obs, dtype=np.float64).reshape(-1)

    def _metrics_from_obs(self, raw_obs: Any) -> dict[str, float]:
        raw = np.asarray(list(raw_obs.values()) if isinstance(raw_obs, dict) else raw_obs, dtype=np.float64).reshape(-1)
        cfg = self.config.link

        if raw.size == 6 and np.all((0.0 <= raw) & (raw <= 1.0)):
            rtt_ms = raw[0] * 250.0
            loss_rate = raw[1]
            throughput_mbps = raw[2] * cfg.bandwidth_mbps
            ack_interval_ms = raw[3] * 200.0
            cwnd_packets = raw[4] * cfg.max_cwnd
            queue_packets = raw[5] * cfg.buffer_packets
        elif raw.size >= 16 and int(raw[1]) == 1:
            segment_size = max(1.0, raw[6])
            cwnd_packets = raw[5] / segment_size
            rtt_ms = max(1.0, raw[11] / 1000.0)
            throughput_mbps = self._throughput_to_mbps(raw[15])
            ack_interval_ms = self._ack_interval_ms(throughput_mbps, segment_size)
            queue_packets = max(0.0, cwnd_packets - cfg.bdp_packets)
            loss_rate = 0.0
        elif raw.size >= 15:
            segment_size = max(1.0, raw[6])
            cwnd_packets = raw[5] / segment_size
            rtt_ms = max(1.0, raw[9] / 1000.0)
            throughput_mbps = self._estimate_throughput_mbps(cwnd_packets, segment_size, rtt_ms)
            ack_interval_ms = self._ack_interval_ms(throughput_mbps, segment_size)
            queue_packets = max(0.0, cwnd_packets - cfg.bdp_packets)
            loss_rate = 1.0 if int(raw[11]) == 0 or int(raw[12]) == 4 else 0.0
        else:
            raise ValueError(
                "Unsupported ns3-gym observation. Expected the project 6-value "
                "observation or the bundled rl-tcp observation."
            )

        return {
            "cwnd": float(np.clip(cwnd_packets, cfg.min_cwnd, cfg.max_cwnd)),
            "rtt_ms": float(rtt_ms),
            "loss_rate": float(np.clip(loss_rate, 0.0, 1.0)),
            "throughput_mbps": float(np.clip(throughput_mbps, 0.0, cfg.bandwidth_mbps)),
            "ack_interval_ms": float(np.clip(ack_interval_ms, 0.0, 200.0)),
            "queue_packets": float(np.clip(queue_packets, 0.0, cfg.buffer_packets)),
            "bdp_packets": float(cfg.bdp_packets),
        }

    def _make_obs(self, metrics: dict[str, float]) -> np.ndarray:
        cfg = self.config.link
        pipe_packets = cfg.bdp_packets + cfg.buffer_packets
        return np.array(
            [
                min(metrics["rtt_ms"] / 250.0, 1.0),
                min(metrics["loss_rate"], 1.0),
                min(metrics["throughput_mbps"] / cfg.bandwidth_mbps, 1.0),
                min(metrics["ack_interval_ms"] / 200.0, 1.0),
                min(metrics["cwnd"] / cfg.max_cwnd, 1.0),
                min(metrics["queue_packets"] / cfg.buffer_packets, 1.0),
                min(self.w_max / cfg.max_cwnd, 1.0),
                min(self.steps_since_loss / max(self.config.max_steps, 1), 1.0),
                min(metrics["cwnd"] / max(pipe_packets, 1.0), 1.0),
            ],
            dtype=np.float32,
        )

    def _update_cubic_memory(self, metrics: dict[str, float]) -> None:
        cfg = self.config.link
        congested = metrics["loss_rate"] > 0.0 or metrics["queue_packets"] > 0.85 * cfg.buffer_packets
        if congested:
            self.w_max = max(metrics["cwnd"], self.last_cwnd)
            self.steps_since_loss = 0
        else:
            self.steps_since_loss += 1
            self.w_max = max(self.w_max, metrics["cwnd"])
        self.last_cwnd = metrics["cwnd"]

    def _compute_reward(
        self,
        metrics: dict[str, float],
        previous_metrics: dict[str, float],
        action: int,
        ns3_reward: float,
    ) -> float:
        if self.config.reward_mode == "ns3":
            return float(ns3_reward)

        cfg = self.config.link
        throughput_util = min(metrics["throughput_mbps"] / cfg.bandwidth_mbps, 1.0)
        queueing_delay_ms = max(0.0, metrics["rtt_ms"] - cfg.base_rtt_ms)
        delay_penalty = queueing_delay_ms / max(cfg.base_rtt_ms, 1.0)
        loss_penalty = metrics["loss_rate"]
        queue_penalty = metrics["queue_packets"] / max(cfg.buffer_packets, 1.0)

        # CUBIC keeps the pipe full and tolerates moderate queues, but should
        # react before queue saturation and packet loss become persistent.
        high_delay_penalty = max(0.0, metrics["rtt_ms"] / max(cfg.base_rtt_ms, 1.0) - 2.0) ** 2
        high_queue_penalty = max(0.0, queue_penalty - 0.75) ** 2

        target_cwnd = cfg.bdp_packets + 0.35 * cfg.buffer_packets
        cwnd_error = abs(metrics["cwnd"] - target_cwnd) / max(target_cwnd, 1.0)
        prev_cwnd = previous_metrics.get("cwnd", metrics["cwnd"])
        oscillation_penalty = abs(metrics["cwnd"] - prev_cwnd) / cfg.max_cwnd

        congested_before = (
            previous_metrics.get("loss_rate", 0.0) > 0.0
            or previous_metrics.get("queue_packets", 0.0) > 0.80 * cfg.buffer_packets
            or previous_metrics.get("rtt_ms", cfg.base_rtt_ms) > 2.2 * cfg.base_rtt_ms
        )
        underutilized_before = previous_metrics.get("throughput_mbps", 0.0) < 0.85 * cfg.bandwidth_mbps
        queue_low_before = previous_metrics.get("queue_packets", 0.0) < 0.45 * cfg.buffer_packets

        response_bonus = 0.0
        if congested_before and action in {0, 1}:
            response_bonus += 0.35 if action == 0 else 0.15
        if underutilized_before and not congested_before and action in {3, 4}:
            response_bonus += 0.10
        if underutilized_before and queue_low_before and action == 4:
            response_bonus += 0.15
        if throughput_util > 0.90 and loss_penalty == 0.0 and queue_penalty < 0.75:
            response_bonus += 0.30

        hard_loss_penalty = 3.0 if loss_penalty > 0.0 else 0.0

        reward = (
            self.config.throughput_weight * throughput_util
            - self.config.delay_weight * delay_penalty
            - self.config.loss_weight * loss_penalty
            - self.config.queue_weight * queue_penalty
            - 1.5 * self.config.delay_weight * high_delay_penalty
            - 3.0 * self.config.queue_weight * high_queue_penalty
            - hard_loss_penalty
            - self.config.cwnd_error_weight * min(cwnd_error, 2.0)
            - self.config.oscillation_weight * oscillation_penalty
            + response_bonus
        )
        return float(reward)

    def _parse_extra_info(self, extra_info: Any) -> dict[str, float]:
        if isinstance(extra_info, dict):
            return {str(key): float(value) for key, value in extra_info.items() if self._is_number(value)}

        if not isinstance(extra_info, str) or not extra_info.strip():
            return {}

        text = extra_info.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            return {str(key): float(value) for key, value in parsed.items() if self._is_number(value)}

        values = {}
        for token in text.replace(",", ";").split(";"):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if self._is_number(value.strip()):
                values[key.strip()] = float(value)
        return values

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _estimate_throughput_mbps(cwnd_packets: float, segment_size: float, rtt_ms: float) -> float:
        return cwnd_packets * segment_size * 8 / (rtt_ms / 1000.0) / 1_000_000

    @staticmethod
    def _throughput_to_mbps(value: float) -> float:
        if value > 1_000_000:
            return value / 1_000_000
        if value > 1_000:
            return value * 8 / 1_000_000
        return value

    @staticmethod
    def _ack_interval_ms(throughput_mbps: float, segment_size: float) -> float:
        if throughput_mbps <= 1e-6:
            return 200.0
        packets_per_second = throughput_mbps * 1_000_000 / (segment_size * 8)
        return 1000.0 / packets_per_second


def make_env(env_name: str, ns3_config: Ns3Config | None = None) -> gym.Env:
    if env_name == "local":
        return TcpCongestionEnv()
    if env_name == "ns3":
        return Ns3TcpEnv(ns3_config)
    raise ValueError(f"Unsupported env: {env_name}")


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.buffer.append((obs.copy(), int(action), float(reward), next_obs.copy(), bool(done)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(obs), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DoubleDQNModel:
    def __init__(self, q_net: QNetwork, device: torch.device):
        self.q_net = q_net
        self.device = device

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        del deterministic
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(self.q_net(obs_tensor), dim=1).item())
        return action, None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algo": "ddqn",
                "obs_dim": self.q_net.net[0].in_features,
                "action_dim": self.q_net.net[-1].out_features,
                "state_dict": self.q_net.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(path: Path, device: torch.device | None = None) -> "DoubleDQNModel":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        q_net = QNetwork(int(checkpoint["obs_dim"]), int(checkpoint["action_dim"])).to(device)
        q_net.load_state_dict(checkpoint["state_dict"])
        q_net.eval()
        return DoubleDQNModel(q_net, device)


def make_training_env(
    env_name: str,
    seed: int,
    rank: int,
    ns3_config: Ns3Config | None = None,
) -> Callable[[], gym.Env]:
    def init() -> gym.Env:
        worker_config = ns3_config
        if worker_config is not None:
            worker_config = replace(
                worker_config,
                port=worker_config.port + rank,
                sim_seed=worker_config.sim_seed + rank,
            )

        env = make_env(env_name, worker_config)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return init


def train_ppo(
    total_timesteps: int,
    seed: int,
    env_name: str,
    model_path: Path,
    ns3_config: Ns3Config | None = None,
    resume_from: Path | None = None,
    n_envs: int = 1,
) -> PPO:
    if n_envs < 1:
        raise ValueError("--n-envs must be at least 1")

    if n_envs == 1:
        env = Monitor(make_env(env_name, ns3_config))
    else:
        env = SubprocVecEnv(
            [make_training_env(env_name, seed, rank, ns3_config) for rank in range(n_envs)]
        )

    if env_name == "local":
        check_env(TcpCongestionEnv(), warn=True)

    if resume_from is not None:
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            seed=seed,
            verbose=1,
        )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    return model


def train_ddqn(
    total_timesteps: int,
    seed: int,
    env_name: str,
    model_path: Path,
    ns3_config: Ns3Config | None = None,
    resume_from: Path | None = None,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    gamma: float = 0.99,
    buffer_size: int = 50_000,
    learning_starts: int = 1_000,
    target_update_interval: int = 500,
    train_frequency: int = 1,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    exploration_fraction: float = 0.35,
) -> DoubleDQNModel:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(env_name, ns3_config)
    obs, _info = env.reset(seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    if resume_from is not None:
        model = DoubleDQNModel.load(resume_from, device)
        q_net = model.q_net
    else:
        q_net = QNetwork(obs_dim, action_dim).to(device)

    target_net = QNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    replay = ReplayBuffer(buffer_size)
    exploration_steps = max(1, int(total_timesteps * exploration_fraction))

    try:
        for step in range(1, total_timesteps + 1):
            progress = min(1.0, step / exploration_steps)
            epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

            if random.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(q_net(obs_tensor), dim=1).item())

            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            replay.add(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                obs, _info = env.reset(seed=seed + step)

            if len(replay) >= max(batch_size, learning_starts) and step % train_frequency == 0:
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = replay.sample(batch_size, device)
                q_values = q_net(batch_obs).gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = torch.argmax(q_net(batch_next_obs), dim=1)
                    next_q_values = target_net(batch_next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets = batch_rewards + gamma * (1.0 - batch_dones) * next_q_values

                loss = F.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

            if step % target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

            if step % 1000 == 0:
                print(f"DDQN step={step} epsilon={epsilon:.3f} replay={len(replay)}")
    finally:
        env.close()

    model = DoubleDQNModel(q_net, device)
    model.save(model_path)
    return model


def train(
    algo: str,
    total_timesteps: int,
    seed: int,
    env_name: str,
    model_path: Path,
    ns3_config: Ns3Config | None = None,
    resume_from: Path | None = None,
    n_envs: int = 1,
) -> PPO | DoubleDQNModel:
    if algo == "ppo":
        return train_ppo(
            total_timesteps=total_timesteps,
            seed=seed,
            env_name=env_name,
            model_path=model_path,
            ns3_config=ns3_config,
            resume_from=resume_from,
            n_envs=n_envs,
        )

    if algo == "ddqn":
        if n_envs != 1:
            raise ValueError("DDQN training in this script uses one environment; set --n-envs 1.")
        return train_ddqn(
            total_timesteps=total_timesteps,
            seed=seed,
            env_name=env_name,
            model_path=model_path,
            ns3_config=ns3_config,
            resume_from=resume_from,
        )

    raise ValueError(f"Unsupported algo: {algo}")


def ppo_policy(model: PPO) -> Callable[[np.ndarray, dict[str, float]], int]:
    def choose(obs: np.ndarray, _: dict[str, float]) -> int:
        action, _state = model.predict(obs, deterministic=True)
        return int(action)

    return choose


def ddqn_policy(model: DoubleDQNModel) -> Callable[[np.ndarray, dict[str, float]], int]:
    def choose(obs: np.ndarray, _: dict[str, float]) -> int:
        action, _state = model.predict(obs, deterministic=True)
        return int(action)

    return choose


class CubicPolicy:
    """CUBIC-inspired baseline over the discrete RL-TCP action interface.

    This is not ns-3's native TcpCubic implementation. It keeps the same
    evaluation pipeline as PPO/Reno by choosing among the project's five
    actions: backoff, mild decrease, keep, additive increase, and aggressive probe.
    """

    def __init__(self, c: float = 0.4, beta: float = 0.7):
        self.c = c
        self.beta = beta
        self.reset()

    def reset(self) -> None:
        self.epoch_start_step: int | None = None
        self.w_max = 10.0
        self.last_cwnd = 10.0

    def __call__(self, _: np.ndarray, info: dict[str, float]) -> int:
        step = int(info.get("step", 0))
        cwnd = max(1.0, float(info.get("cwnd", self.last_cwnd)))
        queue = float(info.get("queue_packets", 0.0))
        loss = float(info.get("loss_rate", 0.0))
        rtt_ms = max(1.0, float(info.get("rtt_ms", 1.0)))

        if loss > 0.0 or queue > 68.0:
            self.w_max = max(cwnd, self.last_cwnd)
            self.epoch_start_step = None
            self.last_cwnd = cwnd
            return 0
        if queue > 58.0:
            self.last_cwnd = cwnd
            return 1

        if self.epoch_start_step is None:
            self.epoch_start_step = step

        elapsed_rtts = max(0.0, (step - self.epoch_start_step) * 500.0 / rtt_ms)
        k = ((self.w_max * (1.0 - self.beta)) / self.c) ** (1.0 / 3.0) if self.w_max > 0 else 0.0
        cubic_target = self.c * ((elapsed_rtts - k) ** 3) + self.w_max
        tcp_friendly_target = self.beta * self.w_max + 0.3 * elapsed_rtts
        target_cwnd = max(cubic_target, tcp_friendly_target, cwnd)

        self.last_cwnd = cwnd
        if queue < 40.0 and cwnd + 3.0 < target_cwnd:
            return 4
        if queue < 55.0 and cwnd < target_cwnd + 1.0:
            return 3
        return 2


def reno_policy(_: np.ndarray, info: dict[str, float]) -> int:
    if info.get("loss_rate", 0.0) > 0.01 or info.get("queue_packets", 0.0) > 65.0:
        return 0
    return 3


def evaluate_policy(
    name: str,
    choose_action: Callable[[np.ndarray, dict[str, float]], int],
    episodes: int,
    seed: int,
    env_name: str,
    ns3_config: Ns3Config | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []

    for episode in range(episodes):
        env = make_env(env_name, ns3_config)
        try:
            if hasattr(choose_action, "reset"):
                choose_action.reset()
            obs, info = env.reset(seed=seed + episode)
            total_reward = 0.0

            while True:
                policy_info = info.copy()
                policy_info["step"] = env.current_step
                action = choose_action(obs, policy_info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                row = {
                    "policy": name,
                    "episode": episode,
                    "step": env.current_step,
                    "action": action,
                    "action_name": ACTION_NAMES.get(action, str(action)),
                    "reward": reward,
                    "total_reward": total_reward,
                    **info,
                }
                rows.append(row)

                if terminated or truncated:
                    break
        finally:
            env.close()

    return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["throughput_mbps", "rtt_ms", "loss_rate", "cwnd"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)

    for axis, metric in zip(axes, metrics):
        for policy in sorted({str(row["policy"]) for row in rows}):
            points = [row for row in rows if row["policy"] == policy and row["episode"] == 0]
            axis.plot(
                [int(row["step"]) for row in points],
                [float(row[metric]) for row in points],
                label=policy,
            )
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)

    axes[0].legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    print("\nEvaluation summary")
    print("------------------")
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        final_rows = []
        for episode in sorted({int(row["episode"]) for row in policy_rows}):
            episode_rows = [row for row in policy_rows if int(row["episode"]) == episode]
            final_rows.append(max(episode_rows, key=lambda row: int(row["step"])))

        avg_reward = np.mean([float(row["total_reward"]) for row in final_rows])
        avg_throughput = np.mean([float(row["throughput_mbps"]) for row in policy_rows])
        avg_rtt = np.mean([float(row["rtt_ms"]) for row in policy_rows])
        avg_loss = np.mean([float(row["loss_rate"]) for row in policy_rows])
        avg_cwnd = np.mean([float(row["cwnd"]) for row in policy_rows])
        avg_queue = np.mean([float(row["queue_packets"]) for row in policy_rows])
        print(
            f"{policy:>5} | reward={avg_reward:8.2f} | "
            f"throughput={avg_throughput:5.2f} Mbps | "
            f"rtt={avg_rtt:6.2f} ms | loss={avg_loss:.4f} | "
            f"cwnd={avg_cwnd:6.2f} pkts | queue={avg_queue:6.2f} pkts"
        )


def evaluate(
    algo: str,
    model_path: Path,
    episodes: int,
    seed: int,
    env_name: str,
    ns3_config: Ns3Config | None = None,
) -> None:
    if algo == "ppo":
        model = PPO.load(model_path)
        learned_policy = ppo_policy(model)
    elif algo == "ddqn":
        model = DoubleDQNModel.load(model_path)
        learned_policy = ddqn_policy(model)
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    rows = []
    rows.extend(
        evaluate_policy(
            algo,
            learned_policy,
            episodes=episodes,
            seed=seed,
            env_name=env_name,
            ns3_config=ns3_config,
        )
    )
    rows.extend(
        evaluate_policy(
            "reno",
            reno_policy,
            episodes=episodes,
            seed=seed,
            env_name=env_name,
            ns3_config=ns3_config,
        )
    )
    rows.extend(
        evaluate_policy(
            "cubic",
            CubicPolicy(),
            episodes=episodes,
            seed=seed,
            env_name=env_name,
            ns3_config=ns3_config,
        )
    )

    csv_path = OUTPUT_DIR / "evaluation.csv"
    plot_path = OUTPUT_DIR / "evaluation_metrics.png"
    write_csv(rows, csv_path)
    plot_metrics(rows, plot_path)
    print_summary(rows)
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")


def parse_sim_args(items: Iterable[str] | None) -> dict[str, str]:
    sim_args: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"ns-3 sim arg must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        sim_args[key] = value
    return sim_args


def parse_rate_mbps(value: Any, default: float) -> float:
    text = str(value).strip().lower()
    multipliers = {
        "gbps": 1000.0,
        "mbps": 1.0,
        "kbps": 0.001,
        "bps": 0.000001,
    }
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * multiplier
    try:
        return float(text)
    except ValueError:
        return default


def parse_time_ms(value: Any, default: float) -> float:
    text = str(value).strip().lower()
    multipliers = {
        "ms": 1.0,
        "s": 1000.0,
        "us": 0.001,
        "ns": 0.000001,
    }
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * multiplier
    try:
        return float(text)
    except ValueError:
        return default


def link_config_from_sim_args(sim_args: dict[str, str | int | float]) -> LinkConfig:
    default = LinkConfig()
    bandwidth_mbps = parse_rate_mbps(sim_args.get("--bottleneck_bandwidth"), default.bandwidth_mbps)
    bottleneck_delay_ms = parse_time_ms(sim_args.get("--bottleneck_delay"), default.base_rtt_ms / 4)
    access_delay_ms = parse_time_ms(sim_args.get("--access_delay"), default.base_rtt_ms / 4)

    return LinkConfig(
        bandwidth_mbps=bandwidth_mbps,
        base_rtt_ms=2.0 * (access_delay_ms + bottleneck_delay_ms),
        buffer_packets=default.buffer_packets,
        packet_size_bytes=default.packet_size_bytes,
        min_cwnd=default.min_cwnd,
        max_cwnd=default.max_cwnd,
        max_steps=default.max_steps,
        noise_std_ms=default.noise_std_ms,
    )


def build_ns3_config(args: argparse.Namespace) -> Ns3Config:
    sim_args: dict[str, str | int | float] = dict(NS3_NETWORK_PRESETS[args.network_preset])
    sim_args["--duration"] = args.ns3_duration
    if args.ns3_transport_prot:
        sim_args["--transport_prot"] = args.ns3_transport_prot
    sim_args.update(parse_sim_args(args.ns3_sim_arg))
    link_config = link_config_from_sim_args(sim_args)

    return Ns3Config(
        scenario_dir=args.ns3_scenario_dir,
        port=args.ns3_port,
        step_time=args.ns3_step_time,
        start_sim=args.ns3_start_sim,
        sim_seed=args.seed,
        sim_args=sim_args,
        debug=args.ns3_debug,
        max_steps=args.ns3_max_steps,
        reward_mode=args.reward_mode,
        throughput_weight=args.throughput_weight,
        delay_weight=args.delay_weight,
        loss_weight=args.loss_weight,
        queue_weight=args.queue_weight,
        cwnd_error_weight=args.cwnd_error_weight,
        oscillation_weight=args.oscillation_weight,
        link=link_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate RL TCP control.")
    parser.add_argument("--algo", choices=["ppo", "ddqn"], default="ppo")
    parser.add_argument("--env", choices=["local", "ns3"], default="local")
    parser.add_argument("--reward-mode", choices=["cwnd-control", "ns3"], default="cwnd-control")
    parser.add_argument("--network-preset", choices=sorted(NS3_NETWORK_PRESETS), default="bottleneck")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--ns3-scenario-dir", type=Path, default=DEFAULT_NS3_SCENARIO_DIR)
    parser.add_argument("--ns3-port", type=int, default=5555)
    parser.add_argument("--ns3-step-time", type=float, default=0.5)
    parser.add_argument("--ns3-duration", type=float, default=10.0)
    parser.add_argument("--ns3-transport-prot", default="TcpRl")
    parser.add_argument("--ns3-max-steps", type=int, default=LinkConfig().max_steps)
    parser.add_argument("--ns3-sim-arg", action="append", default=[])
    parser.add_argument("--ns3-debug", action="store_true")
    parser.add_argument("--ns3-start-sim", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--throughput-weight", type=float, default=9.0)
    parser.add_argument("--delay-weight", type=float, default=0.6)
    parser.add_argument("--loss-weight", type=float, default=4.0)
    parser.add_argument("--queue-weight", type=float, default=0.25)
    parser.add_argument("--cwnd-error-weight", type=float, default=0.15)
    parser.add_argument("--oscillation-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.algo == "ddqn" and args.model_path == MODEL_PATH:
        args.model_path = DDQN_MODEL_PATH
    ns3_config = build_ns3_config(args) if args.env == "ns3" else None

    if args.mode in {"train", "all"}:
        train(
            algo=args.algo,
            total_timesteps=args.timesteps,
            seed=args.seed,
            env_name=args.env,
            model_path=args.model_path,
            ns3_config=ns3_config,
            resume_from=args.resume_from,
            n_envs=args.n_envs,
        )

    if args.mode in {"eval", "all"}:
        evaluate(
            algo=args.algo,
            model_path=args.model_path,
            episodes=args.episodes,
            seed=args.seed,
            env_name=args.env,
            ns3_config=ns3_config,
        )


if __name__ == "__main__":
    main()
