"""Configuration helpers for the security incident response environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float_env(name: str, default: float) -> float:
    """Read a float environment variable with a safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a safe default."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


DEFAULT_ENV_NAME = os.getenv("OPENENV_ENV_NAME", "security-incident-response")
SERVICE_HOST = os.getenv("OPENENV_HOST", "0.0.0.0")
SERVICE_PORT = get_int_env("OPENENV_PORT", 7860)


@dataclass(frozen=True)
class EnvironmentConfig:
    """Runtime configuration for seeded scenario generation and observability."""

    seed: int = 17
    max_steps: int = 8
    budget: int = 7
    noise_level: float = 1.0
    num_decoys: int = 1
    observation_window_size: int = 5
    total_logs: int = 12
    difficulty_profile: str = "standard"
    initial_visible_logs: int = 5
    reveal_per_step: int = 1
    randomize_identifiers: bool = True

    @property
    def log_window(self) -> int:
        """Backward-compatible alias for the observation window size."""
        return self.observation_window_size

    @property
    def max_budget(self) -> int:
        """Backward-compatible alias for the configured per-episode budget."""
        return self.budget


def load_environment_config() -> EnvironmentConfig:
    """Load environment configuration from environment variables."""
    difficulty_profile = os.getenv("OPENENV_DIFFICULTY", "standard").strip().lower() or "standard"
    default_noise = {
        "standard": 1.0,
        "hardcore": 1.4,
        "lite": 0.7,
    }.get(difficulty_profile, 1.0)
    default_budget = {
        "standard": 7,
        "hardcore": 6,
        "lite": 8,
    }.get(difficulty_profile, 7)
    default_decoys = {
        "standard": 1,
        "hardcore": 2,
        "lite": 1,
    }.get(difficulty_profile, 1)
    default_logs = {
        "standard": 12,
        "hardcore": 14,
        "lite": 10,
    }.get(difficulty_profile, 12)

    max_steps = max(3, min(10, get_int_env("OPENENV_MAX_STEPS", 8)))
    window_default = get_int_env("OPENENV_OBSERVATION_WINDOW_SIZE", get_int_env("OPENENV_LOG_WINDOW", 5))
    observation_window_size = max(3, min(8, window_default))
    total_logs = max(observation_window_size + 2, min(24, get_int_env("OPENENV_TOTAL_LOGS", default_logs)))
    initial_visible_logs = max(
        3,
        min(observation_window_size, get_int_env("OPENENV_INITIAL_VISIBLE_LOGS", observation_window_size)),
    )

    return EnvironmentConfig(
        seed=get_int_env("OPENENV_SEED", 17),
        max_steps=max_steps,
        budget=max(4, min(10, get_int_env("OPENENV_BUDGET", default_budget))),
        noise_level=max(0.0, min(2.0, get_float_env("OPENENV_NOISE_LEVEL", default_noise))),
        num_decoys=max(1, min(3, get_int_env("OPENENV_NUM_DECOYS", default_decoys))),
        observation_window_size=observation_window_size,
        total_logs=total_logs,
        difficulty_profile=difficulty_profile,
        initial_visible_logs=initial_visible_logs,
        reveal_per_step=max(1, min(3, get_int_env("OPENENV_REVEAL_PER_STEP", 1))),
        randomize_identifiers=get_bool_env("OPENENV_RANDOMIZE_IDENTIFIERS", True),
    )


DEFAULT_CONFIG = load_environment_config()
DEFAULT_MAX_STEPS = DEFAULT_CONFIG.max_steps
