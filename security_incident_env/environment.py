"""Core OpenEnv-compatible security incident response environment."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from .config import DEFAULT_CONFIG, DEFAULT_ENV_NAME, EnvironmentConfig
from .graders import grade_episode
from .models import (
    Action,
    ActionType,
    Alert,
    AttackStage,
    Observation,
    ScenarioDefinition,
    Severity,
    State,
    StepInfo,
    TaskName,
)
from .scenarios import TASK_OFFSETS, build_scenario

STAGE_ORDER = {
    AttackStage.NONE: 0,
    AttackStage.RECONNAISSANCE: 1,
    AttackStage.EXPLOITATION: 2,
    AttackStage.PERSISTENCE: 3,
    AttackStage.CONTAINED: 4,
}

ACTION_COSTS = {
    ActionType.ANALYZE_LOG: 1,
    ActionType.FLAG_ALERT: 1,
    ActionType.BLOCK_IP: 2,
    ActionType.ESCALATE: 2,
    ActionType.IGNORE: 0,
}


class SecurityIncidentResponseEnv:
    """Deterministic incident response environment with a Gym-like API."""

    def __init__(
        self,
        env_name: str = DEFAULT_ENV_NAME,
        max_steps: int | None = None,
        seed: int | None = None,
        config: EnvironmentConfig | None = None,
    ) -> None:
        base_config = config or DEFAULT_CONFIG
        if max_steps is not None:
            base_config = replace(base_config, max_steps=max(3, min(10, max_steps)))
        if seed is not None:
            base_config = replace(base_config, seed=seed)

        self.env_name = env_name
        self._config = base_config
        self._episode_counter = 0
        self._task_success_history: dict[TaskName, list[bool]] = {task_name: [] for task_name in TaskName}
        self._task_mistake_history: dict[TaskName, list[int]] = {task_name: [] for task_name in TaskName}
        self._scenario: ScenarioDefinition | None = None
        self._alerts_by_id: Dict[str, Alert] = {}
        self._state: State | None = None
        self._episode_outcome_recorded = False
        self._feedback = "Episode not started."

    def reset(self, task_name: TaskName = TaskName.EASY) -> Observation:
        """Initialize a new deterministic episode and return the first observation."""
        effective_config = self._effective_config(task_name)
        episode_seed = effective_config.seed + (self._episode_counter * 97) + TASK_OFFSETS[task_name]
        self._episode_counter += 1
        scenario = build_scenario(task_name=task_name, seed=episode_seed, config=effective_config)
        self._scenario = scenario
        self._episode_outcome_recorded = False
        self._alerts_by_id = {alert.alert_id: alert.model_copy(deep=True) for alert in scenario.alerts}

        initial_count = min(max(1, scenario.initial_visible_log_count), len(scenario.logs))
        visible_logs = [log.model_copy(deep=True) for log in scenario.logs[:initial_count]]
        remaining_logs = [log.model_copy(deep=True) for log in scenario.logs[initial_count:]]

        self._state = State(
            env_name=self.env_name,
            task_name=task_name,
            scenario_seed=scenario.scenario_seed,
            attack_path=scenario.attack_path,
            branch_description=scenario.branch_description,
            full_log_history=visible_logs,
            remaining_log_queue=remaining_logs,
            malicious_log_ids=list(scenario.malicious_log_ids),
            malicious_ips=list(scenario.malicious_ips),
            benign_ips=list(scenario.benign_ips),
            decoy_ips=list(scenario.decoy_ips),
            decoy_log_ids=list(scenario.decoy_log_ids),
            decoy_alert_ids=list(scenario.decoy_alert_ids),
            log_stage_map=dict(scenario.log_stage_map),
            attack_progression_stage=AttackStage.NONE,
            steps_taken=0,
            incident_resolved=False,
            analyzed_log_ids=[],
            flagged_alert_ids=[],
            blocked_ips=[],
            false_positive_blocks=[],
            escalation_sent=False,
            action_history=[],
            observation_history=[],
            feedback_history=[],
            reward_history=[],
            raw_reward_history=[],
            cumulative_raw_reward=0.0,
            invalid_action_count=0,
            redundant_action_count=0,
            useless_step_count=0,
            missed_signal_count=0,
            attack_progressions=0,
            timeliness_penalty=0.0,
            score_cap=1.0,
            max_steps=min(self._config.max_steps, scenario.max_steps),
            max_budget=scenario.max_budget,
            remaining_budget=scenario.max_budget,
            budget_exhausted=False,
            window_size=effective_config.log_window,
            optimal_steps=scenario.optimal_steps,
            action_costs={action_type.value: cost for action_type, cost in ACTION_COSTS.items()},
            evidence_groups=[list(group) for group in scenario.evidence_groups],
            required_analysis_log_ids=list(scenario.required_analysis_log_ids),
            required_alert_ids=list(scenario.required_alert_ids),
            required_block_ips=list(scenario.required_block_ips),
            requires_escalation=scenario.requires_escalation,
            resolved_step=None,
        )
        self._update_attack_stage()
        self._feedback = (
            f"Monitoring started for task '{task_name.value}'. "
            f"Only the most recent {effective_config.log_window} logs remain visible and the episode budget is "
            f"{scenario.max_budget}. Adaptive difficulty level {self._adaptive_level(task_name)} is active."
        )
        initial_observation = self._build_observation()
        self._state.observation_history.append(initial_observation.model_copy(deep=True))
        self._state.feedback_history.append(self._feedback)
        return initial_observation

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, object]]:
        """Advance the environment by one action."""
        state = self._require_state()
        self._require_scenario()
        raw_reward = 0.0
        invalid_action = False
        meaningful_action = False
        feedback = "Action recorded."

        if state.incident_resolved or state.steps_taken >= state.max_steps or state.budget_exhausted:
            raw_reward = -0.25
            invalid_action = True
            feedback = "Episode already completed. Call reset() to start a new episode."
        else:
            state.steps_taken += 1
            state.action_history.append(action)

            cost = ACTION_COSTS[action.action_type]
            if cost > state.remaining_budget:
                state.invalid_action_count += 1
                state.budget_exhausted = True
                state.remaining_budget = 0
                raw_reward = -0.65
                invalid_action = True
                feedback = f"Action cost {cost} exceeded the remaining budget. The episode terminated immediately."
            else:
                state.remaining_budget -= cost
                if action.action_type == ActionType.ANALYZE_LOG:
                    raw_reward, feedback, invalid_action, meaningful_action = self._handle_analyze_log(action)
                elif action.action_type == ActionType.FLAG_ALERT:
                    raw_reward, feedback, invalid_action, meaningful_action = self._handle_flag_alert(action)
                elif action.action_type == ActionType.BLOCK_IP:
                    raw_reward, feedback, invalid_action, meaningful_action = self._handle_block_ip(action)
                elif action.action_type == ActionType.ESCALATE:
                    raw_reward, feedback, invalid_action, meaningful_action = self._handle_escalate(action)
                elif action.action_type == ActionType.IGNORE:
                    raw_reward, feedback, invalid_action, meaningful_action = self._handle_ignore()

                timeline_note = self._advance_timeline(
                    action=action,
                    invalid_action=invalid_action,
                    meaningful_action=meaningful_action,
                )
                if timeline_note:
                    feedback = f"{feedback} {timeline_note}".strip()

                state.incident_resolved = self._is_incident_resolved()
                if state.incident_resolved and state.resolved_step is None:
                    state.resolved_step = state.steps_taken

        self._update_attack_stage()
        state.cumulative_raw_reward = round(state.cumulative_raw_reward + raw_reward, 4)
        state.raw_reward_history.append(round(raw_reward, 4))
        state.reward_history.append(round(raw_reward, 4))

        done = state.incident_resolved or state.steps_taken >= state.max_steps or state.budget_exhausted
        self._feedback = feedback
        done = state.incident_resolved or state.steps_taken >= state.max_steps or state.budget_exhausted
        observation = self._build_observation()
        state.observation_history.append(observation.model_copy(deep=True))
        state.feedback_history.append(feedback)
        grade = grade_episode(state, use_llm_judge=done)
        info = StepInfo(
            raw_reward=round(raw_reward, 4),
            cumulative_reward=state.cumulative_raw_reward,
            budget_remaining=state.remaining_budget,
            budget_exhausted=state.budget_exhausted,
            resolved=state.incident_resolved,
            attack_stage=state.attack_progression_stage,
            invalid_action=invalid_action,
            score=grade.score,
            programmatic_score=grade.programmatic_score,
            llm_judge_score=grade.llm_judge_score,
            final_reward=grade.final_reward,
            feedback=feedback,
            score_cap=state.score_cap,
            judge_explanation=grade.judge_explanation,
            judge_phase_classification=grade.judge_phase_classification,
            judge_phase_quality=grade.judge_phase_quality,
        ).model_dump()
        if done and not self._episode_outcome_recorded:
            self._record_episode_outcome(state)
            self._episode_outcome_recorded = True
        return observation, round(raw_reward, 4), done, info

    def state(self) -> State:
        """Return the full internal state."""
        return self._require_state().model_copy(deep=True)

    def grade(self, use_llm_judge: bool | None = None):
        """Grade the current episode deterministically."""
        return grade_episode(self._require_state(), use_llm_judge=use_llm_judge)

    def close(self) -> None:
        """Release episode-local state so the instance can be safely discarded or reused."""
        self._scenario = None
        self._alerts_by_id = {}
        self._state = None
        self._episode_outcome_recorded = False
        self._feedback = "Episode closed."

    def _build_observation(self) -> Observation:
        state = self._require_state()
        current_logs = [log.model_copy(deep=True) for log in state.full_log_history[-state.window_size :]]
        alerts = [alert.model_copy(deep=True) for alert in self._visible_alerts()]
        return Observation(
            current_logs=current_logs,
            active_alerts=alerts,
            blocked_ips=list(state.blocked_ips),
            remaining_budget=state.remaining_budget,
            max_budget=state.max_budget,
            step_count=state.steps_taken,
            previous_action_feedback=self._feedback,
            visible_history_count=len(state.full_log_history),
            log_window_size=state.window_size,
            context_truncated=len(state.full_log_history) > state.window_size,
        )

    def _handle_analyze_log(self, action: Action) -> tuple[float, str, bool, bool]:
        state = self._require_state()
        if not action.log_id:
            state.invalid_action_count += 1
            return -0.25, "analyze_log requires log_id.", True, False

        visible_log_ids = {log.log_id for log in state.full_log_history}
        if action.log_id not in visible_log_ids:
            state.invalid_action_count += 1
            return -0.25, f"Log {action.log_id} is not available in the revealed history.", True, False

        if action.log_id in state.analyzed_log_ids:
            state.redundant_action_count += 1
            return self._useless_penalty(-0.12), f"Log {action.log_id} was already analyzed.", False, False

        state.analyzed_log_ids.append(action.log_id)
        evidence_group = self._group_for_log(action.log_id)
        if action.log_id in state.required_analysis_log_ids:
            if evidence_group and not self._group_is_complete(evidence_group):
                return (
                    self._scale_positive_reward(0.08),
                    f"Log {action.log_id} is suspicious, but it is not decisive until correlated with {self._group_partner_text(evidence_group, action.log_id)}.",
                    False,
                    True,
                )
            if evidence_group:
                return (
                    self._scale_positive_reward(0.28),
                    f"Combined analysis of {', '.join(evidence_group)} confirms the real intrusion chain.",
                    False,
                    True,
                )
            base_reward = 0.18 if self._log_stage(action.log_id) == AttackStage.RECONNAISSANCE else 0.22
            return self._scale_positive_reward(base_reward), f"Log {action.log_id} added decisive attack evidence.", False, True

        if action.log_id in state.malicious_log_ids:
            return self._scale_positive_reward(0.1), f"Log {action.log_id} is suspicious and contributes weak evidence.", False, True

        if action.log_id in state.decoy_log_ids:
            state.timeliness_penalty = round(min(0.7, state.timeliness_penalty + 0.05), 4)
            state.missed_signal_count += 1
            if state.task_name == TaskName.HARD:
                state.score_cap = min(state.score_cap, 0.8)
            return (
                self._useless_penalty(-0.09),
                f"Log {action.log_id} looked hostile, but it belongs to a decoy trail and did not improve containment readiness.",
                False,
                False,
            )

        return self._useless_penalty(-0.05), f"Log {action.log_id} appears benign and did not advance the investigation.", False, False

    def _handle_flag_alert(self, action: Action) -> tuple[float, str, bool, bool]:
        state = self._require_state()
        if not action.alert_id:
            state.invalid_action_count += 1
            return -0.25, "flag_alert requires alert_id.", True, False

        visible_alerts = {alert.alert_id: alert for alert in self._visible_alerts()}
        alert = visible_alerts.get(action.alert_id)
        if alert is None:
            state.invalid_action_count += 1
            return -0.25, f"Alert {action.alert_id} is not currently visible.", True, False
        if action.alert_id in state.flagged_alert_ids:
            state.redundant_action_count += 1
            return self._useless_penalty(-0.12), f"Alert {action.alert_id} was already flagged.", False, False

        state.flagged_alert_ids.append(action.alert_id)
        self._alerts_by_id[action.alert_id].status = "flagged"

        if action.alert_id in state.required_alert_ids:
            if self._required_analysis_complete():
                return self._scale_positive_reward(0.32), f"Alert {action.alert_id} correctly correlates the active attack.", False, True
            return self._scale_positive_reward(0.12), f"Alert {action.alert_id} points in the right direction, but the evidence chain is incomplete.", False, True

        if action.alert_id in state.decoy_alert_ids:
            state.timeliness_penalty = round(min(0.6, state.timeliness_penalty + 0.04), 4)
            return self._useless_penalty(-0.2), f"Alert {action.alert_id} is misleading and consumed analyst attention.", False, False

        return self._useless_penalty(-0.15), f"Alert {action.alert_id} does not correspond to the real incident driver.", False, False

    def _handle_block_ip(self, action: Action) -> tuple[float, str, bool, bool]:
        state = self._require_state()
        if not action.ip_address:
            state.invalid_action_count += 1
            return -0.25, "block_ip requires ip_address.", True, False

        known_ips = {log.source_ip for log in state.full_log_history}
        known_ips.update(alert.source_ip for alert in self._visible_alerts() if alert.source_ip)
        if action.ip_address not in known_ips:
            state.invalid_action_count += 1
            return -0.25, f"IP {action.ip_address} has not appeared in the investigation context.", True, False
        if action.ip_address in state.blocked_ips:
            state.redundant_action_count += 1
            return self._useless_penalty(-0.12), f"IP {action.ip_address} is already blocked.", False, False

        state.blocked_ips.append(action.ip_address)
        if action.ip_address not in state.malicious_ips:
            state.false_positive_blocks.append(action.ip_address)
            state.timeliness_penalty = round(min(0.8, state.timeliness_penalty + 0.06), 4)
            state.score_cap = min(state.score_cap, 0.6)
            return -0.55, f"IP {action.ip_address} is legitimate. This false positive permanently caps the score at 0.6.", False, False

        evidence_ready = self._required_alerts_complete() and self._required_analysis_complete()
        if action.ip_address in state.required_block_ips:
            reward = 0.58 if evidence_ready else 0.2
            response = (
                f"IP {action.ip_address} was contained with supporting evidence."
                if evidence_ready
                else f"IP {action.ip_address} is malicious, but the block happened before the case was fully substantiated."
            )
            return self._scale_positive_reward(reward), response, False, True

        return self._scale_positive_reward(0.08), f"IP {action.ip_address} is suspicious, but not the primary containment target.", False, True

    def _handle_escalate(self, action: Action) -> tuple[float, str, bool, bool]:
        state = self._require_state()
        if state.escalation_sent:
            state.redundant_action_count += 1
            return self._useless_penalty(-0.12), "Incident was already escalated.", False, False

        if not state.requires_escalation:
            return self._useless_penalty(-0.14), "Escalation is unnecessary for this task and adds avoidable operational noise.", False, False

        evidence_ready = self._sufficient_escalation_evidence()
        containment_ready = self._required_blocks_complete()
        if not evidence_ready:
            state.premature_escalation_count += 1
            state.timeliness_penalty = round(min(0.7, state.timeliness_penalty + 0.05), 4)
            return -0.4, "Escalation was premature and lacked the required evidence chain.", False, False
        if not containment_ready:
            state.premature_escalation_count += 1
            state.timeliness_penalty = round(min(0.7, state.timeliness_penalty + 0.03), 4)
            return -0.1, "Escalation is defensible, but mitigation is incomplete. Contain the attacker before closing the case.", False, True

        state.escalation_sent = True
        return self._scale_positive_reward(0.38), "Escalation captured a substantiated, mitigated incident and ended the response cleanly.", False, True

    def _handle_ignore(self) -> tuple[float, str, bool, bool]:
        state = self._require_state()
        state.ignored_attack_count += 1
        severity_penalty = 0.04 * STAGE_ORDER[state.attack_progression_stage]
        state.timeliness_penalty = round(min(0.8, state.timeliness_penalty + 0.04 + severity_penalty), 4)
        return (
            self._useless_penalty(round(-0.12 - severity_penalty, 4)),
            "No mitigation action taken. The attack timeline continues to advance while the analyst waits.",
            False,
            False,
        )

    def _advance_timeline(self, action: Action, invalid_action: bool, meaningful_action: bool) -> str:
        state = self._require_state()
        if state.incident_resolved or state.budget_exhausted:
            return ""

        reveal_target = self._config.reveal_per_step
        if invalid_action or action.action_type == ActionType.IGNORE or not meaningful_action:
            reveal_target += 1

        newly_revealed: list[str] = []
        while reveal_target > 0 and state.remaining_log_queue:
            next_log = state.remaining_log_queue.pop(0)
            if next_log.log_id in state.malicious_log_ids and next_log.source_ip in state.blocked_ips:
                continue
            state.full_log_history.append(next_log)
            newly_revealed.append(next_log.log_id)
            reveal_target -= 1

        highest_before = STAGE_ORDER[state.attack_progression_stage]
        self._update_attack_stage()
        highest_after = STAGE_ORDER[state.attack_progression_stage]

        if highest_after > highest_before and state.attack_progression_stage != AttackStage.CONTAINED:
            state.attack_progressions += 1
            state.missed_signal_count += 1
            state.timeliness_penalty = round(min(0.7, state.timeliness_penalty + 0.1), 4)
            if state.attack_progression_stage == AttackStage.PERSISTENCE:
                state.score_cap = min(state.score_cap, 0.88)
            if action.action_type == ActionType.IGNORE:
                state.score_cap = min(state.score_cap, 0.82)
            if newly_revealed:
                return (
                    f"New telemetry arrived ({', '.join(newly_revealed)}), and the attack progressed to "
                    f"{state.attack_progression_stage.value}."
                )

        if newly_revealed:
            return f"New telemetry arrived ({', '.join(newly_revealed)})."
        if state.remaining_log_queue:
            return ""
        return "No additional telemetry arrived."

    def _visible_alerts(self) -> list[Alert]:
        state = self._require_state()
        revealed_log_ids = {log.log_id for log in state.full_log_history}
        alerts: list[Alert] = []
        for alert in self._alerts_by_id.values():
            if alert.status == "closed":
                continue
            if set(alert.related_log_ids).intersection(revealed_log_ids) or alert.alert_id in state.flagged_alert_ids:
                alerts.append(alert)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
        }
        return sorted(alerts, key=lambda item: (severity_order[item.severity], item.alert_id))

    def _update_attack_stage(self) -> None:
        state = self._require_state()
        if state.incident_resolved:
            state.attack_progression_stage = AttackStage.CONTAINED
            return

        highest_stage = AttackStage.NONE
        for log in state.full_log_history:
            if log.log_id not in state.malicious_log_ids:
                continue
            if log.source_ip in state.blocked_ips:
                continue
            stage = state.log_stage_map.get(log.log_id, AttackStage.NONE)
            if STAGE_ORDER[stage] > STAGE_ORDER[highest_stage]:
                highest_stage = stage
        state.attack_progression_stage = highest_stage

    def _is_incident_resolved(self) -> bool:
        state = self._require_state()
        required_blocked = self._required_blocks_complete()
        required_alerts = self._required_alerts_complete()
        required_analysis = self._required_analysis_complete()
        if state.requires_escalation:
            return required_blocked and required_alerts and required_analysis and state.escalation_sent
        return required_blocked and required_alerts

    def _required_analysis_complete(self) -> bool:
        state = self._require_state()
        return set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids))

    def _required_alerts_complete(self) -> bool:
        state = self._require_state()
        return set(state.required_alert_ids).issubset(set(state.flagged_alert_ids))

    def _required_blocks_complete(self) -> bool:
        state = self._require_state()
        return set(state.required_block_ips).issubset(set(state.blocked_ips))

    def _sufficient_escalation_evidence(self) -> bool:
        return self._required_analysis_complete() and self._required_alerts_complete()

    def _scale_positive_reward(self, base_reward: float) -> float:
        state = self._require_state()
        multiplier = max(0.45, 1.0 - state.timeliness_penalty)
        return round(base_reward * multiplier, 4)

    def _useless_penalty(self, base_penalty: float) -> float:
        state = self._require_state()
        state.useless_step_count += 1
        penalty_scale = 1.0 + (0.18 * max(0, state.useless_step_count - 1))
        return round(base_penalty * penalty_scale, 4)

    def _group_for_log(self, log_id: str) -> list[str] | None:
        state = self._require_state()
        for group in state.evidence_groups:
            if log_id in group:
                return group
        return None

    def _group_is_complete(self, group: list[str]) -> bool:
        state = self._require_state()
        return set(group).issubset(set(state.analyzed_log_ids))

    def _group_partner_text(self, group: list[str], log_id: str) -> str:
        partners = [item for item in group if item != log_id]
        if not partners:
            return "related telemetry"
        if len(partners) == 1:
            return partners[0]
        return ", ".join(partners[:-1]) + f" and {partners[-1]}"

    def _log_stage(self, log_id: str) -> AttackStage:
        state = self._require_state()
        return state.log_stage_map.get(log_id, AttackStage.NONE)

    def _require_state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        return self._state

    def _require_scenario(self) -> ScenarioDefinition:
        if self._scenario is None:
            raise RuntimeError("Scenario not initialized. Call reset() before step().")
        return self._scenario

    def _adaptive_level(self, task_name: TaskName) -> int:
        history = self._task_success_history[task_name][-4:]
        if len(history) < 2:
            return 0
        success_rate = sum(1 for item in history if item) / len(history)
        if success_rate >= 0.9:
            return 2
        if success_rate >= 0.75:
            return 1
        return 0

    def _effective_config(self, task_name: TaskName) -> EnvironmentConfig:
        level = self._adaptive_level(task_name)
        if level == 0:
            return self._config

        return replace(
            self._config,
            num_decoys=min(3, self._config.num_decoys + level),
            observation_window_size=max(3, self._config.observation_window_size - level),
            initial_visible_logs=max(3, min(self._config.initial_visible_logs, self._config.observation_window_size - level)),
        )

    def _record_episode_outcome(self, state: State) -> None:
        history = self._task_success_history[state.task_name]
        history.append(state.incident_resolved and not state.false_positive_blocks)
        if len(history) > 6:
            del history[0]

        mistakes = state.invalid_action_count + state.redundant_action_count + state.useless_step_count + len(state.false_positive_blocks)
        mistake_history = self._task_mistake_history[state.task_name]
        mistake_history.append(mistakes)
        if len(mistake_history) > 6:
            del mistake_history[0]
