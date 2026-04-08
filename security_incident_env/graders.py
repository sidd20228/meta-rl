"""Deterministic graders for OpenEnv task evaluation with optional LLM judging."""

from __future__ import annotations

import math

from .judge import evaluate_judge
from .models import ActionType, EpisodeGrade, State

TIME_DECAY_ALPHA = 0.05


def grade_episode(state: State, use_llm_judge: bool | None = None) -> EpisodeGrade:
    """Grade an episode based on the full action trajectory."""
    programmatic = _programmatic_breakdown(state)
    judge_result = evaluate_judge(state, use_llm=use_llm_judge) if _should_run_judge(state, use_llm_judge) else evaluate_judge(state, use_llm=False)

    score = (0.7 * programmatic["score"]) + (0.3 * judge_result.normalized_score)
    score = max(0.0, min(1.0, score))

    environment_reward = _environment_reward(state)
    final_reward = _final_reward(environment_reward, programmatic["score"], judge_result.score, state)

    return EpisodeGrade(
        task_name=state.task_name,
        score=round(score, 4),
        programmatic_score=round(programmatic["score"], 4),
        llm_judge_score=round(judge_result.score, 4),
        llm_judge_normalized_score=round(judge_result.normalized_score, 4),
        environment_reward=round(environment_reward, 4),
        final_reward=round(final_reward, 4),
        correctness=round(programmatic["correctness"], 4),
        efficiency=round(programmatic["efficiency"], 4),
        timeliness=round(programmatic["timeliness"], 4),
        trajectory_quality=round(programmatic["trajectory_quality"], 4),
        false_positive_penalty=round(programmatic["false_positive_penalty"], 4),
        resolved=state.incident_resolved,
        steps_taken=state.steps_taken,
        score_cap=round(state.score_cap, 4),
        judge_used_llm=judge_result.used_llm,
        judge_explanation=judge_result.explanation,
        judge_phase_classification=judge_result.phase_classification,
        judge_phase_quality=judge_result.phase_quality,
        judge_fallback_reason=judge_result.fallback_reason,
    )


def _should_run_judge(state: State, use_llm_judge: bool | None) -> bool:
    if use_llm_judge is False:
        return False
    if use_llm_judge is True:
        return True
    return state.incident_resolved or state.budget_exhausted or state.steps_taken >= state.max_steps


def _environment_reward(state: State) -> float:
    """Normalize trajectory reward into [-1, 1] for reward-layer reporting."""
    normalized = math.tanh(state.cumulative_raw_reward / 2.4)
    if not state.incident_resolved:
        normalized -= 0.15
    if state.false_positive_blocks:
        normalized -= 0.1 * len(state.false_positive_blocks)
    if state.budget_exhausted:
        normalized -= 0.12
    return max(-1.0, min(1.0, normalized))


def _final_reward(environment_reward: float, programmatic_score: float, judge_score: float, state: State) -> float:
    """Combine reward layers into a final research-style reward in [-1, 1]."""
    combined = (0.35 * environment_reward) + (0.4 * ((2.0 * programmatic_score) - 1.0)) + (0.25 * judge_score)
    if state.incident_resolved and not state.false_positive_blocks:
        combined = min(1.0, combined + 0.08)
    if not state.incident_resolved:
        combined = max(-1.0, combined - 0.12)
    if state.false_positive_blocks:
        combined = max(-1.0, combined - 0.08)
    return max(-1.0, min(1.0, combined))


def _programmatic_breakdown(state: State) -> dict[str, float]:
    analyzed_required = len(set(state.analyzed_log_ids).intersection(state.required_analysis_log_ids))
    required_analysis_total = max(1, len(state.required_analysis_log_ids))
    analysis_score = analyzed_required / required_analysis_total

    dependency_score = _dependency_score(state)

    flagged_required = len(set(state.flagged_alert_ids).intersection(state.required_alert_ids))
    required_alert_total = max(1, len(state.required_alert_ids))
    alert_score = flagged_required / required_alert_total

    blocked_required = len(set(state.blocked_ips).intersection(state.required_block_ips))
    required_block_total = max(1, len(state.required_block_ips))
    block_score = blocked_required / required_block_total

    escalation_score = 1.0 if (not state.requires_escalation or state.escalation_sent) else 0.0
    correctness = min(
        1.0,
        (0.2 * analysis_score)
        + (0.2 * dependency_score)
        + (0.2 * alert_score)
        + (0.25 * block_score)
        + (0.15 * escalation_score),
    )
    if state.incident_resolved:
        correctness = min(1.0, correctness + 0.05)

    extra_steps = max(0, state.steps_taken - state.optimal_steps)
    efficiency = max(
        0.0,
        1.0
        - (0.05 * extra_steps)
        - (0.06 * state.useless_step_count)
        - (0.05 * state.premature_escalation_count)
        - (0.04 if state.budget_exhausted else 0.0),
    )
    timeliness = max(
        0.0,
        1.0 - state.timeliness_penalty - (0.08 * state.missed_signal_count) - (0.04 * state.ignored_attack_count),
    )

    trajectory_penalty = (
        (0.08 * state.invalid_action_count)
        + (0.05 * state.redundant_action_count)
        + (0.07 * max(0, state.attack_progressions - 1))
        + (0.06 * state.premature_escalation_count)
    )
    trajectory_quality = max(0.0, 1.0 - trajectory_penalty)
    if not _ordering_is_valid(state):
        trajectory_quality = max(0.0, trajectory_quality - 0.18)

    false_positive_penalty = min(0.5, 0.25 * len(state.false_positive_blocks))
    score = (
        (0.4 * correctness)
        + (0.18 * efficiency)
        + (0.2 * timeliness)
        + (0.12 * trajectory_quality)
        + (0.1 * dependency_score)
        - false_positive_penalty
    )

    time_decay_multiplier = max(0.0, 1.0 - (TIME_DECAY_ALPHA * extra_steps))
    score *= time_decay_multiplier

    if state.false_positive_blocks:
        score = min(score, 0.6)
    if state.budget_exhausted:
        score = min(score, 0.55)
    if not state.incident_resolved:
        score = min(score, 0.5)
    score = min(score, state.score_cap)

    if _is_optimal_trajectory(state):
        score = 1.0

    return {
        "score": max(0.0, min(1.0, score)),
        "correctness": correctness,
        "efficiency": efficiency,
        "timeliness": timeliness,
        "trajectory_quality": trajectory_quality,
        "false_positive_penalty": false_positive_penalty,
    }


def _dependency_score(state: State) -> float:
    if not state.evidence_groups:
        return 1.0 if set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids)) else 0.0
    analyzed = set(state.analyzed_log_ids)
    completed = sum(1 for group in state.evidence_groups if set(group).issubset(analyzed))
    return completed / max(1, len(state.evidence_groups))


def _ordering_is_valid(state: State) -> bool:
    analyze_indexes = _action_indexes(state, ActionType.ANALYZE_LOG)
    flag_indexes = _action_indexes(state, ActionType.FLAG_ALERT)
    block_indexes = _action_indexes(state, ActionType.BLOCK_IP)
    escalate_indexes = _action_indexes(state, ActionType.ESCALATE)

    if not flag_indexes or not block_indexes:
        return False
    if analyze_indexes and max(analyze_indexes) > min(flag_indexes):
        analyzed_before_flag = [
            action.log_id
            for index, action in enumerate(state.action_history, start=1)
            if action.action_type == ActionType.ANALYZE_LOG and index < min(flag_indexes)
        ]
        if not set(state.required_analysis_log_ids).issubset(set(analyzed_before_flag)):
            return False
    if min(flag_indexes) > min(block_indexes):
        return False
    if state.requires_escalation and (not escalate_indexes or min(block_indexes) > min(escalate_indexes)):
        return False
    return True


def _is_optimal_trajectory(state: State) -> bool:
    return (
        state.incident_resolved
        and not state.false_positive_blocks
        and not state.budget_exhausted
        and state.invalid_action_count == 0
        and state.redundant_action_count == 0
        and state.useless_step_count == 0
        and state.premature_escalation_count == 0
        and state.missed_signal_count <= 1
        and state.steps_taken <= state.optimal_steps
        and set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids))
        and set(state.required_alert_ids).issubset(set(state.flagged_alert_ids))
        and set(state.required_block_ips).issubset(set(state.blocked_ips))
        and _dependency_score(state) == 1.0
        and (not state.requires_escalation or state.escalation_sent)
        and _ordering_is_valid(state)
    )


def _action_indexes(state: State, action_type: ActionType) -> list[int]:
    return [index for index, action in enumerate(state.action_history, start=1) if action.action_type == action_type]
