"""Adaptive curriculum controller for security incident tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .config import EnvironmentConfig
from .models import State, TaskName


@dataclass(frozen=True)
class CurriculumProfile:
    """Difficulty settings derived from recent agent performance."""

    level: int
    name: str
    weak_spots: tuple[str, ...]


@dataclass
class CurriculumController:
    """Track mastery and weak spots across episodes."""

    window: int = 8
    min_episodes: int = 2
    mastery_threshold: float = 0.75
    _success_history: dict[TaskName, list[bool]] = field(default_factory=lambda: {task_name: [] for task_name in TaskName})
    _score_history: dict[TaskName, list[float]] = field(default_factory=lambda: {task_name: [] for task_name in TaskName})
    _weak_spots: dict[TaskName, dict[str, int]] = field(default_factory=lambda: {task_name: {} for task_name in TaskName})

    def profile_for(self, task_name: TaskName) -> CurriculumProfile:
        """Return the current curriculum profile for a task."""
        scores = self._score_history[task_name][-self.window :]
        successes = self._success_history[task_name][-self.window :]
        if len(successes) < self.min_episodes:
            level = 0
        else:
            success_rate = sum(1 for item in successes if item) / len(successes)
            average_score = sum(scores) / len(scores) if scores else 0.0
            if success_rate >= 0.9 and average_score >= 0.85:
                level = 2
            elif success_rate >= self.mastery_threshold and average_score >= 0.7:
                level = 1
            else:
                level = 0

        names = ("warmup", "analyst", "principal")
        weak_spots = tuple(
            spot
            for spot, _ in sorted(
                self._weak_spots[task_name].items(),
                key=lambda item: (-item[1], item[0]),
            )[:3]
        )
        return CurriculumProfile(level=level, name=names[level], weak_spots=weak_spots)

    def effective_config(self, task_name: TaskName, base_config: EnvironmentConfig) -> EnvironmentConfig:
        """Tighten observability and add decoys as mastery improves."""
        profile = self.profile_for(task_name)
        if profile.level == 0:
            return base_config
        return replace(
            base_config,
            noise_level=min(2.0, base_config.noise_level + (0.25 * profile.level)),
            num_decoys=min(3, base_config.num_decoys + profile.level),
            observation_window_size=max(3, base_config.observation_window_size - profile.level),
            initial_visible_logs=max(3, min(base_config.initial_visible_logs, base_config.observation_window_size - profile.level)),
        )

    def record(self, state: State, score: float) -> None:
        """Record episode outcome and update weak-spot counters."""
        task_name = state.task_name
        success = bool(state.incident_resolved and not state.false_positive_blocks and score >= 0.7)
        self._append(self._success_history[task_name], success)
        self._append(self._score_history[task_name], score)
        for weak_spot in self._episode_weak_spots(state):
            self._weak_spots[task_name][weak_spot] = self._weak_spots[task_name].get(weak_spot, 0) + 1

    def snapshot(self, task_name: TaskName) -> dict[str, object]:
        """Expose a compact curriculum snapshot for observations and transcripts."""
        profile = self.profile_for(task_name)
        scores = self._score_history[task_name][-self.window :]
        successes = self._success_history[task_name][-self.window :]
        success_rate = (sum(1 for item in successes if item) / len(successes)) if successes else 0.0
        return {
            "level": profile.level,
            "profile": profile.name,
            "weak_spots": list(profile.weak_spots),
            "recent_success_rate": round(success_rate, 4),
            "recent_average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    def _append(self, values: list[object], value: object) -> None:
        values.append(value)
        if len(values) > self.window:
            del values[0]

    def _episode_weak_spots(self, state: State) -> list[str]:
        weak_spots: list[str] = []
        if state.false_positive_blocks:
            weak_spots.append("false_positive_containment")
        if not set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids)):
            weak_spots.append("incomplete_investigation")
        if not set(state.required_alert_ids).issubset(set(state.flagged_alert_ids)):
            weak_spots.append("missed_alert_correlation")
        if not set(state.required_block_ips).issubset(set(state.blocked_ips)):
            weak_spots.append("missed_containment")
        if state.requires_escalation and not state.escalation_sent:
            weak_spots.append("missed_escalation")
        if state.premature_escalation_count:
            weak_spots.append("premature_escalation")
        if state.report_submitted and state.report_score < 0.5:
            weak_spots.append("weak_case_report")
        if state.useless_step_count or state.redundant_action_count:
            weak_spots.append("inefficient_workflow")
        return weak_spots
