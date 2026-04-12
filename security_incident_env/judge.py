"""Optional LLM-backed semantic judge with deterministic fallback."""

from __future__ import annotations

import json
import os
from typing import Iterable

from openai import OpenAI

from .models import (
    ActionType,
    AttackStage,
    JudgePersonaResult,
    JudgePhase,
    JudgePhaseQuality,
    JudgeResult,
    State,
)

JUDGE_PERSONAS = (
    (
        "junior",
        0.2,
        "You are a junior SOC analyst reviewer. Be lenient but still note obvious workflow or containment mistakes.",
    ),
    (
        "senior",
        0.5,
        "You are a senior SOC analyst reviewer. Apply standard incident response expectations and weigh tradeoffs realistically.",
    ),
    (
        "principal",
        0.3,
        "You are a principal incident response leader. Be strict about false positives, workflow quality, and evidence quality.",
    ),
)

DEFAULT_JUDGE_BASE_URL = "https://router.huggingface.co/v1"


def evaluate_judge(state: State, use_llm: bool | None = None) -> JudgeResult:
    """Evaluate the final trajectory with a multi-persona judge."""
    should_use_llm = _should_use_llm(use_llm)
    fallback_reason: str | None = None
    persona_reviews: list[JudgePersonaResult] = []
    used_llm = False

    if should_use_llm:
        try:
            persona_reviews = _run_llm_personas(state)
            used_llm = all(review.used_llm for review in persona_reviews)
        except Exception as exc:  # pragma: no cover - network and provider failures are environment-specific
            fallback_reason = f"LLM judge fallback: {type(exc).__name__}: {exc}"
            persona_reviews = []

    if not persona_reviews:
        persona_reviews = [_fallback_persona_review(state, persona=name, weight=weight) for name, weight, _ in JUDGE_PERSONAS]
        used_llm = False
        if fallback_reason is None and should_use_llm:
            fallback_reason = "LLM judge fallback: no persona response."
        if fallback_reason is None and not should_use_llm:
            fallback_reason = "LLM judge disabled or credentials unavailable."

    score = _weighted_average((review.score, review.weight) for review in persona_reviews)
    normalized_score = max(0.0, min(1.0, (score + 1.0) / 2.0))
    phase_quality = JudgePhaseQuality(
        triage=round(_weighted_average((review.phase_quality.triage, review.weight) for review in persona_reviews), 4),
        investigation=round(_weighted_average((review.phase_quality.investigation, review.weight) for review in persona_reviews), 4),
        mitigation=round(_weighted_average((review.phase_quality.mitigation, review.weight) for review in persona_reviews), 4),
        resolution=round(_weighted_average((review.phase_quality.resolution, review.weight) for review in persona_reviews), 4),
    )
    phase_classification = _dominant_phase(phase_quality)
    explanation = " | ".join(f"{review.persona}: {review.reasoning}" for review in persona_reviews)
    return JudgeResult(
        score=round(max(-1.0, min(1.0, score)), 4),
        normalized_score=round(normalized_score, 4),
        explanation=explanation,
        phase_classification=phase_classification,
        phase_quality=phase_quality,
        personas=persona_reviews,
        used_llm=used_llm,
        fallback_reason=fallback_reason,
    )


def _run_llm_personas(state: State) -> list[JudgePersonaResult]:
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_JUDGE_BASE_URL)
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not model_name or not api_key:
        raise RuntimeError("MODEL_NAME and HF_TOKEN or OPENAI_API_KEY are required for the LLM judge.")

    timeout_seconds = max(3.0, float(os.getenv("OPENENV_JUDGE_TIMEOUT", "12")))
    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=timeout_seconds)

    trajectory_summary = _trajectory_summary(state)
    system_prompt = (
        "You are a cybersecurity expert evaluating an AI SOC analyst. "
        "Return strict JSON with keys: score, reasoning, phase_classification, phase_quality. "
        "phase_quality must contain triage, investigation, mitigation, resolution, each a float in [-1, 1]."
    )
    base_user_prompt = (
        "Given:\n"
        f"- trajectory:\n{trajectory_summary}\n"
        f"- final_state:\n{json.dumps(_state_summary(state), indent=2, sort_keys=True)}\n"
        f"- ground_truth:\n{json.dumps(_ground_truth_summary(state), indent=2, sort_keys=True)}\n\n"
        "Evaluate:\n"
        "1. Did the agent identify the correct attacker?\n"
        "2. Did it avoid false positives?\n"
        "3. Was the sequence efficient?\n"
        "4. Did it follow correct workflow?\n\n"
        "Return:\n"
        "score: float (-1 to 1)\n"
        "reasoning: short explanation\n"
        "phase_classification: one of triage, investigation, mitigation, resolution\n"
        "phase_quality: breakdown"
    )

    reviews: list[JudgePersonaResult] = []
    for persona, weight, persona_instruction in JUDGE_PERSONAS:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": persona_instruction},
                {"role": "user", "content": base_user_prompt},
            ],
        )
        raw_content = response.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        reviews.append(
            JudgePersonaResult(
                persona=persona,
                weight=weight,
                score=round(max(-1.0, min(1.0, float(payload.get("score", 0.0)))), 4),
                reasoning=str(payload.get("reasoning", "")).strip() or f"{persona.title()} reviewer provided no reasoning.",
                phase_classification=JudgePhase(str(payload.get("phase_classification", _fallback_phase(state).value)).lower()),
                phase_quality=_parse_phase_quality(payload.get("phase_quality")),
                used_llm=True,
            )
        )
    return reviews


def _fallback_persona_review(state: State, persona: str, weight: float) -> JudgePersonaResult:
    base_score = _fallback_score(state)
    persona_offsets = {
        "junior": 0.12,
        "senior": 0.0,
        "principal": -0.12,
    }
    score = round(max(-1.0, min(1.0, base_score + persona_offsets.get(persona, 0.0))), 4)
    phase_quality = _fallback_phase_quality(state)
    false_positive_note = "false positives were avoided" if not state.false_positive_blocks else "false positives hurt the investigation"
    resolution_note = "incident resolved cleanly" if state.incident_resolved else "incident remained partially unresolved"
    reasoning = (
        f"{persona.title()} review: attacker identification was "
        f"{'correct' if _identified_correct_attacker(state) else 'incomplete'}, {false_positive_note}, and {resolution_note}."
    )
    return JudgePersonaResult(
        persona=persona,
        weight=weight,
        score=score,
        reasoning=reasoning,
        phase_classification=_fallback_phase(state),
        phase_quality=phase_quality,
        used_llm=False,
    )


def _trajectory_summary(state: State) -> str:
    lines = []
    for index, observation in enumerate(state.observation_history, start=0):
        log_ids = [log.log_id for log in observation.current_logs]
        alert_ids = [alert.alert_id for alert in observation.active_alerts]
        if index == 0:
            lines.append(
                f"step=0 observation logs={log_ids} alerts={alert_ids} blocked={observation.blocked_ips} "
                f"budget={observation.remaining_budget}/{observation.max_budget} feedback={state.feedback_history[0]!r}"
            )
            continue
        action = state.action_history[index - 1]
        lines.append(
            f"step={index} action={action.model_dump(mode='json')} logs={log_ids} alerts={alert_ids} "
            f"blocked={observation.blocked_ips} budget={observation.remaining_budget}/{observation.max_budget} "
            f"feedback={state.feedback_history[index]!r}"
        )
    return "\n".join(lines)


def _state_summary(state: State) -> dict[str, object]:
    return {
        "resolved": state.incident_resolved,
        "attack_stage": state.attack_progression_stage.value,
        "steps_taken": state.steps_taken,
        "remaining_budget": state.remaining_budget,
        "blocked_ips": state.blocked_ips,
        "isolated_hosts": state.isolated_hosts,
        "analyzed_log_ids": state.analyzed_log_ids,
        "flagged_alert_ids": state.flagged_alert_ids,
        "queried_terms": state.queried_terms,
        "intel_lookups": state.intel_lookups,
        "report_submitted": state.report_submitted,
        "report_score": state.report_score,
        "false_positive_blocks": state.false_positive_blocks,
        "invalid_action_count": state.invalid_action_count,
        "redundant_action_count": state.redundant_action_count,
        "useless_step_count": state.useless_step_count,
        "premature_escalation_count": state.premature_escalation_count,
        "ignored_attack_count": state.ignored_attack_count,
    }


def _ground_truth_summary(state: State) -> dict[str, object]:
    return {
        "malicious_ips": state.malicious_ips,
        "required_analysis_log_ids": state.required_analysis_log_ids,
        "required_alert_ids": state.required_alert_ids,
        "required_block_ips": state.required_block_ips,
        "requires_escalation": state.requires_escalation,
        "evidence_groups": state.evidence_groups,
        "decoy_ips": state.decoy_ips,
        "decoy_log_ids": state.decoy_log_ids,
        "decoy_alert_ids": state.decoy_alert_ids,
    }


def _parse_phase_quality(payload: object) -> JudgePhaseQuality:
    if not isinstance(payload, dict):
        return JudgePhaseQuality(triage=0.0, investigation=0.0, mitigation=0.0, resolution=0.0)
    return JudgePhaseQuality(
        triage=round(max(-1.0, min(1.0, float(payload.get("triage", 0.0)))), 4),
        investigation=round(max(-1.0, min(1.0, float(payload.get("investigation", 0.0)))), 4),
        mitigation=round(max(-1.0, min(1.0, float(payload.get("mitigation", 0.0)))), 4),
        resolution=round(max(-1.0, min(1.0, float(payload.get("resolution", 0.0)))), 4),
    )


def _weighted_average(pairs: Iterable[tuple[float, float]]) -> float:
    pairs = list(pairs)
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in pairs) / total_weight


def _should_use_llm(use_llm: bool | None) -> bool:
    if use_llm is not None:
        return use_llm
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("MODEL_NAME")
    judge_flag = os.getenv("OPENENV_ENABLE_LLM_JUDGE", "1").strip().lower()
    return bool(api_key and model_name and judge_flag not in {"0", "false", "no"})


def _fallback_score(state: State) -> float:
    score = -0.65
    if _identified_correct_attacker(state):
        score += 0.55
    if set(state.required_alert_ids).issubset(set(state.flagged_alert_ids)):
        score += 0.2
    if set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids)):
        score += 0.18
    if state.incident_resolved:
        score += 0.3
    score -= 0.18 * len(state.false_positive_blocks)
    score -= 0.1 * state.invalid_action_count
    score -= 0.07 * state.premature_escalation_count
    score -= 0.05 * max(0, state.steps_taken - state.optimal_steps)
    if state.budget_exhausted:
        score -= 0.18
    return max(-1.0, min(1.0, score))


def _fallback_phase_quality(state: State) -> JudgePhaseQuality:
    triage = 0.6 if state.analyzed_log_ids else -0.2
    investigation = 0.7 if set(state.required_analysis_log_ids).issubset(set(state.analyzed_log_ids)) else 0.1
    mitigation = 0.75 if set(state.required_block_ips).issubset(set(state.blocked_ips)) else -0.25
    resolution = 0.8 if state.incident_resolved else -0.3
    if state.false_positive_blocks:
        mitigation = max(-1.0, mitigation - 0.45)
        resolution = max(-1.0, resolution - 0.2)
    if state.premature_escalation_count:
        resolution = max(-1.0, resolution - 0.15)
    return JudgePhaseQuality(
        triage=round(max(-1.0, min(1.0, triage)), 4),
        investigation=round(max(-1.0, min(1.0, investigation)), 4),
        mitigation=round(max(-1.0, min(1.0, mitigation)), 4),
        resolution=round(max(-1.0, min(1.0, resolution)), 4),
    )


def _fallback_phase(state: State) -> JudgePhase:
    if state.incident_resolved:
        return JudgePhase.RESOLUTION
    if set(state.required_block_ips).issubset(set(state.blocked_ips)):
        return JudgePhase.MITIGATION
    if state.analyzed_log_ids or state.flagged_alert_ids:
        return JudgePhase.INVESTIGATION
    return JudgePhase.TRIAGE


def _dominant_phase(phase_quality: JudgePhaseQuality) -> JudgePhase:
    mapping = {
        JudgePhase.TRIAGE: phase_quality.triage,
        JudgePhase.INVESTIGATION: phase_quality.investigation,
        JudgePhase.MITIGATION: phase_quality.mitigation,
        JudgePhase.RESOLUTION: phase_quality.resolution,
    }
    return max(mapping, key=mapping.get)


def _identified_correct_attacker(state: State) -> bool:
    return set(state.required_block_ips).issubset(set(state.blocked_ips))
