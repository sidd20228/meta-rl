"""Inference runner that uses the OpenAI client to act in the environment."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable, Optional

from openai import OpenAI

from security_incident_env.environment import SecurityIncidentResponseEnv
from security_incident_env.models import Action, Alert, LogEntry, Observation, TaskName


SYSTEM_PROMPT = """You are a SOC analyst agent acting in a deterministic security incident response environment.
Return only JSON with keys: action_type, log_id, ip_address, alert_id.
Pick exactly one action from: analyze_log, flag_alert, block_ip, escalate, ignore.
Use null for unused fields.

Rules:
- Only use ids and IPs that are visible in the current observation.
- Only populate the one field required by the chosen action; set all others to null.
- Prefer investigation before mitigation: analyze evidence before flagging, and flag before blocking.
- Never escalate until the incident is contained; escalate is mainly for the hard task.
- Never repeat a failed or redundant action.
- Do not chase travel, contractor, QA, synthetic, or maintenance decoys when stronger attack evidence is visible.
- Use ignore only when there is no higher-value visible action.
- Return JSON only, with no commentary."""

DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_MAX_TOKENS = int(os.getenv("OPENENV_MODEL_MAX_TOKENS", "128"))


def observation_to_prompt(task_name: TaskName, observation) -> str:
    """Render the observation for the model policy."""
    visible_alert_ips = sorted({alert.source_ip for alert in observation.active_alerts if alert.source_ip})
    return json.dumps(
        {
            "task_name": task_name.value,
            "step_count": observation.step_count,
            "previous_action_feedback": observation.previous_action_feedback,
            "current_logs": [log.model_dump(mode="json") for log in observation.current_logs],
            "active_alerts": [alert.model_dump(mode="json") for alert in observation.active_alerts],
            "blocked_ips": observation.blocked_ips,
            "allowed_log_ids": [log.log_id for log in observation.current_logs],
            "allowed_alert_ids": [alert.alert_id for alert in observation.active_alerts],
            "candidate_block_ips": visible_alert_ips,
            "workflow_hint": {
                "easy": "analyze the real privileged-auth attack logs first, then flag the real alert, then block the attacker",
                "medium": "analyze the true investigation chain before flagging and blocking; avoid QA and travel decoys",
                "hard": "analyze the real chain, flag the critical alert, block the attacker, then escalate last",
            }[task_name.value],
        },
        indent=2,
    )


def _severity_rank(severity: str) -> int:
    return {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
    }.get(severity.lower(), 0)


def _signal_score(text: str) -> int:
    text = text.lower()
    positive = (
        "administrator",
        "privileged",
        "failed",
        "attack",
        "exploit",
        "credential",
        "sql",
        "persistence",
        "malicious",
        "burst",
        "replay",
        "success",
    )
    negative = (
        "travel",
        "contractor",
        "maintenance",
        "approved",
        "qa",
        "synthetic",
        "vendor",
        "drill",
        "staging",
        "benign",
    )
    return sum(2 for token in positive if token in text) - sum(2 for token in negative if token in text)


def _best_log(observation: Observation) -> Optional[LogEntry]:
    if not observation.current_logs:
        return None
    return max(
        observation.current_logs,
        key=lambda log: (
            _severity_rank(log.severity.value),
            _signal_score(f"{log.event_type} {log.message}"),
            log.log_id,
        ),
    )


def _best_alert(observation: Observation) -> Optional[Alert]:
    if not observation.active_alerts:
        return None
    return max(
        observation.active_alerts,
        key=lambda alert: (
            _severity_rank(alert.severity.value),
            _signal_score(f"{alert.name} {alert.summary}"),
            alert.alert_id,
        ),
    )


def _best_block_ip(observation: Observation) -> Optional[str]:
    candidate_alert = _best_alert(observation)
    if candidate_alert and candidate_alert.source_ip and candidate_alert.source_ip not in observation.blocked_ips:
        return candidate_alert.source_ip
    candidate_log = _best_log(observation)
    if candidate_log and candidate_log.source_ip not in observation.blocked_ips:
        return candidate_log.source_ip
    return None


def sanitize_action(action: Action, observation: Observation) -> Action:
    """Enforce action shape, visible identifiers, and a safer default workflow."""
    visible_log_ids = {log.log_id for log in observation.current_logs}
    visible_alert_ids = {alert.alert_id for alert in observation.active_alerts}
    visible_block_ips = {alert.source_ip for alert in observation.active_alerts if alert.source_ip}
    visible_block_ips.update(log.source_ip for log in observation.current_logs)
    best_log = _best_log(observation)
    best_alert = _best_alert(observation)
    best_ip = _best_block_ip(observation)

    normalized = Action(action_type=action.action_type)
    if action.action_type.value == "analyze_log":
        normalized.log_id = action.log_id if action.log_id in visible_log_ids else (best_log.log_id if best_log else None)
        if normalized.log_id is None:
            return Action(action_type="ignore")
    elif action.action_type.value == "flag_alert":
        normalized.alert_id = action.alert_id if action.alert_id in visible_alert_ids else (best_alert.alert_id if best_alert else None)
        if normalized.alert_id is None:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value == "block_ip":
        normalized.ip_address = action.ip_address if action.ip_address in visible_block_ips else best_ip
        if normalized.ip_address is None:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value not in {"escalate", "ignore"}:
        return Action(action_type="ignore")

    # Workflow guardrails to reduce premature mitigation and wasted ignore actions.
    if normalized.action_type.value in {"block_ip", "escalate"} and observation.step_count == 0 and best_log:
        return Action(action_type="analyze_log", log_id=best_log.log_id)
    if normalized.action_type.value == "flag_alert" and observation.step_count == 0 and best_log:
        return Action(action_type="analyze_log", log_id=best_log.log_id)
    if normalized.action_type.value == "block_ip" and not observation.active_alerts and best_log:
        return Action(action_type="analyze_log", log_id=best_log.log_id)
    if normalized.action_type.value == "ignore":
        if best_log:
            return Action(action_type="analyze_log", log_id=best_log.log_id)
        if observation.active_alerts and best_alert:
            return Action(action_type="flag_alert", alert_id=best_alert.alert_id)
    return normalized


def is_fatal_provider_error(exc: Exception) -> bool:
    """Detect provider-side failures that should abort the episode immediately."""
    message = str(exc).lower()
    fatal_markers = (
        "error code: 401",
        "error code: 402",
        "error code: 403",
        "error code: 429",
        "depleted your monthly included credits",
        "insufficient_quota",
        "authentication",
        "invalid api key",
        "rate-limited upstream",
        "rate limited upstream",
        "too many requests",
    )
    return any(marker in message for marker in fatal_markers)


def parse_action_payload(raw_content: Optional[str]) -> dict[str, object]:
    """Parse provider output into a JSON object, tolerating fenced or prefixed text."""
    text = (raw_content or "").strip()
    if not text:
        raise RuntimeError("Empty model response.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Model did not return valid JSON: {text[:200]}")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise RuntimeError("Model response JSON was not an object.")
    return payload


def request_model_action(client: OpenAI, model_name: str, task_name: TaskName, observation: Observation) -> Action:
    """Query the provider and normalize the returned action."""
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        max_tokens=DEFAULT_MODEL_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_to_prompt(task_name, observation)},
        ],
    )
    payload = parse_action_payload(response.choices[0].message.content)
    return sanitize_action(Action.model_validate(payload), observation)


def build_openai_policy() -> Callable[[TaskName, object], Action]:
    """Create a model policy backed by the OpenAI client."""
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not model_name:
        raise RuntimeError("MODEL_NAME is required.")
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY is required.")

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    def policy(task_name: TaskName, observation) -> Action:
        return request_model_action(client, model_name, task_name, observation)

    return policy


def build_heuristic_policy() -> Callable[[TaskName, object], Action]:
    """Deterministic fallback policy useful for local smoke tests."""

    def policy(task_name: TaskName, observation) -> Action:
        blocked = set(observation.blocked_ips)
        step = observation.step_count

        if task_name == TaskName.EASY:
            if step == 0:
                return Action(action_type="analyze_log", log_id="L100")
            if step == 1:
                return Action(action_type="flag_alert", alert_id="A100")
            if "198.51.100.24" not in blocked:
                return Action(action_type="block_ip", ip_address="198.51.100.24")
            return Action(action_type="ignore")

        if task_name == TaskName.MEDIUM:
            if step == 0:
                return Action(action_type="analyze_log", log_id="L200")
            if step == 1:
                return Action(action_type="analyze_log", log_id="L201")
            if step == 2:
                return Action(action_type="flag_alert", alert_id="A200")
            if "203.0.113.77" not in blocked:
                return Action(action_type="block_ip", ip_address="203.0.113.77")
            return Action(action_type="ignore")

        if step == 0:
            return Action(action_type="analyze_log", log_id="L300")
        if step == 1:
            return Action(action_type="analyze_log", log_id="L301")
        if step == 2:
            return Action(action_type="analyze_log", log_id="L302")
        if step == 3:
            return Action(action_type="flag_alert", alert_id="A301")
        if "203.0.113.200" not in blocked:
            return Action(action_type="block_ip", ip_address="203.0.113.200")
        return Action(action_type="escalate")

    return policy


def stringify_action(action: Action) -> str:
    """Compact action formatter for exact step logs."""
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)


def format_error(error: Optional[str]) -> str:
    """Render the error field exactly as required by the submission contract."""
    if not error:
        return "null"
    return " ".join(str(error).splitlines()).strip() or "null"


def run_episode(task_name: TaskName, use_heuristic: bool = False) -> int:
    """Run one task episode and print the exact evaluation logs."""
    env = SecurityIncidentResponseEnv()
    model_name = os.getenv("MODEL_NAME", "heuristic-policy" if use_heuristic else "unknown-model")
    rewards: list[str] = []
    steps_taken = 0
    score = 0.0

    print(f"[START] task={task_name.value} env={env.env_name} model={model_name}")

    success = False
    exit_code = 1
    try:
        observation = env.reset(task_name)
        policy = build_heuristic_policy() if use_heuristic else build_openai_policy()

        while True:
            step_number = observation.step_count + 1
            try:
                action = policy(task_name, observation)
                observation, reward, done, info = env.step(action)
                error_value = None
            except Exception as exc:  # pragma: no cover - defensive logging path
                if is_fatal_provider_error(exc):
                    error_value = str(exc)
                    break
                action = Action(action_type="ignore")
                observation, reward, done, info = env.step(action)
                error_value = str(exc)

            rewards.append(f"{reward:.2f}")
            steps_taken = env.state().steps_taken
            print(
                f"[STEP] step={step_number} action={stringify_action(action)} "
                f"reward={reward:.2f} done={str(done).lower()} error={format_error(error_value)}"
            )
            if done:
                success = bool(info["resolved"])
                break

        score = env.grade().score
        steps_taken = env.state().steps_taken
        exit_code = 0 if success else 1
    except Exception:
        exit_code = 1
    finally:
        try:
            env.close()
        except Exception:
            pass
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.2f} rewards={','.join(rewards)}"
        )
    return exit_code


def main() -> int:
    """CLI entrypoint for single-task inference."""
    parser = argparse.ArgumentParser(description="Run inference against the OpenEnv security incident environment.")
    parser.add_argument("--task", choices=[task.value for task in TaskName], default=TaskName.EASY.value)
    parser.add_argument(
        "--policy",
        choices=["openai", "heuristic"],
        default="openai",
        help="Use the OpenAI client or a local heuristic smoke-test policy.",
    )
    args = parser.parse_args()
    return run_episode(TaskName(args.task), use_heuristic=args.policy == "heuristic")


if __name__ == "__main__":
    raise SystemExit(main())
