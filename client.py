"""Thin local client wrapper required by the OpenEnv packaging layout."""

from __future__ import annotations

from typing import Optional

from security_incident_env.environment import SecurityIncidentResponseEnv
from security_incident_env.models import Action, Observation, State, TaskName


class SecurityIncidentResponseClient:
    """Minimal in-process client over the packaged environment."""

    def __init__(self, task_name: TaskName = TaskName.EASY):
        self._env = SecurityIncidentResponseEnv()
        self._task_name = task_name

    def reset(self, task_name: Optional[TaskName] = None) -> Observation:
        """Reset the underlying environment and return the initial observation."""
        if task_name is not None:
            self._task_name = task_name
        return self._env.reset(self._task_name)

    def step(self, action: Action):
        """Run a single environment step."""
        return self._env.step(action)

    def state(self) -> State:
        """Return the current internal state."""
        return self._env.state()

    def grade(self, use_llm_judge: bool | None = None):
        """Return the current episode grade."""
        return self._env.grade(use_llm_judge=use_llm_judge)

    def close(self) -> None:
        """Close the underlying environment."""
        self._env.close()
