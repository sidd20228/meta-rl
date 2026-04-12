"""Inference runner that uses the OpenAI client to act in the environment."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Callable, Optional

from openai import OpenAI

from security_incident_env.environment import SecurityIncidentResponseEnv
from security_incident_env.models import Action, ActionType, Alert, LogEntry, Observation, TaskName


SYSTEM_PROMPT = """You are a SOC analyst agent acting in a deterministic security incident response environment.
Return only JSON with keys: action_type, log_id, ip_address, alert_id, query, user_id, report.
Pick exactly one action from: analyze_log, query_logs, flag_alert, inspect_user, lookup_threat_intel, block_ip, isolate_host, escalate, create_incident_report, ignore.
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
ALL_TASKS = "all"


def observation_to_prompt(task_name: TaskName, observation) -> str:
    """Render the observation for the model policy."""
    visible_alert_ips = sorted({alert.source_ip for alert in observation.active_alerts if alert.source_ip})
    visible_log_ips = sorted({log.source_ip for log in observation.current_logs if log.source_ip})
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
            "candidate_ips": sorted(set(visible_alert_ips).union(visible_log_ips)),
            "analyst_notes": observation.analyst_notes,
            "curriculum_profile": observation.curriculum_profile,
            "weak_spot_hints": observation.weak_spot_hints,
            "soc_tools": {
                "query_logs": "search telemetry by a concise term before acting when evidence is hidden",
                "lookup_threat_intel": "check a visible IP before containment when the source may be a vendor decoy",
                "inspect_user": "review visible IP ownership or user context",
                "create_incident_report": "summarize attacker, evidence, alert, containment, and escalation near the end",
            },
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
        "enumerated",
        "probed",
        "payload",
        "malformed",
        "privileged_api",
        "follow-on",
        "staged",
        "export",
        "identity",
        "gateway",
        "recovery",
        "service account",
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
        "validation",
        "scanner",
        "replayed",
        "archived",
        "ticketed",
        "exercise",
        "vendor appliance",
        "mirrors",
        "mirror",
    )
    return sum(2 for token in positive if token in text) - sum(2 for token in negative if token in text)


def _best_log(observation: Observation) -> Optional[LogEntry]:
    if not observation.current_logs:
        return None
    return max(
        observation.current_logs,
        key=lambda log: (
            _signal_score(f"{log.event_type} {log.message}"),
            _severity_rank(log.severity.value),
            log.log_id,
        ),
    )


def _best_alert(observation: Observation) -> Optional[Alert]:
    if not observation.active_alerts:
        return None
    return max(
        observation.active_alerts,
        key=lambda alert: (
            _signal_score(f"{alert.name} {alert.summary}"),
            _severity_rank(alert.severity.value),
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


def _best_unanalyzed_log(observation: Observation, analyzed_log_ids: set[str]) -> Optional[LogEntry]:
    candidates = [log for log in observation.current_logs if log.log_id not in analyzed_log_ids]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda log: (
            _signal_score(f"{log.event_type} {log.message}"),
            _severity_rank(log.severity.value),
            log.log_id,
        ),
    )


def _best_analyzed_signal_log(observation: Observation, analyzed_log_ids: set[str]) -> Optional[LogEntry]:
    candidates = [log for log in observation.current_logs if log.log_id in analyzed_log_ids]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda log: (
            _signal_score(f"{log.event_type} {log.message}"),
            _severity_rank(log.severity.value),
            log.log_id,
        ),
    )
    return best if _signal_score(f"{best.event_type} {best.message}") > 0 else None


def _best_same_source_unanalyzed_log(
    observation: Observation,
    analyzed_log_ids: set[str],
    source_ip: str,
) -> Optional[LogEntry]:
    candidates = [
        log
        for log in observation.current_logs
        if log.source_ip == source_ip and log.log_id not in analyzed_log_ids
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda log: (
            _signal_score(f"{log.event_type} {log.message}"),
            _severity_rank(log.severity.value),
            log.log_id,
        ),
    )


def _build_report(task_name: TaskName, observation: Observation, analyzed_log_ids: set[str]) -> str:
    alert = _best_alert(observation)
    ip_address = observation.blocked_ips[-1] if observation.blocked_ips else (_best_block_ip(observation) or (alert.source_ip if alert else "unknown"))
    evidence = sorted(analyzed_log_ids) or [log.log_id for log in observation.current_logs[:3]]
    alert_id = alert.alert_id if alert else "unflagged"
    escalation = " Escalate because this is a multi-stage hard incident." if task_name == TaskName.HARD else ""
    return (
        f"Attacker {ip_address} is tied to evidence {', '.join(evidence)}. "
        f"Alert {alert_id} was correlated and the attacker should be contained by block or isolation.{escalation}"
    )


def sanitize_action(
    action: Action,
    observation: Observation,
    task_name: TaskName | None = None,
    analyzed_log_ids: set[str] | None = None,
) -> Action:
    """Enforce action shape, visible identifiers, and a safer default workflow."""
    visible_log_ids = {log.log_id for log in observation.current_logs}
    visible_alert_ids = {alert.alert_id for alert in observation.active_alerts}
    alerts_by_id = {alert.alert_id: alert for alert in observation.active_alerts}
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
    elif action.action_type.value == "query_logs":
        normalized.query = (action.query or action.log_id or action.ip_address or "").strip() or (
            best_log.event_type if best_log else None
        )
        if normalized.query is None:
            return Action(action_type="ignore")
    elif action.action_type.value == "flag_alert":
        normalized.alert_id = action.alert_id if action.alert_id in visible_alert_ids else (best_alert.alert_id if best_alert else None)
        if normalized.alert_id is None:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value in {"inspect_user", "lookup_threat_intel"}:
        normalized.ip_address = action.ip_address if action.ip_address in visible_block_ips else best_ip
        normalized.user_id = action.user_id
        if normalized.ip_address is None and not normalized.user_id:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value == "block_ip":
        normalized.ip_address = action.ip_address if action.ip_address in visible_block_ips else best_ip
        if normalized.ip_address is None:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value == "isolate_host":
        normalized.ip_address = action.ip_address if action.ip_address in visible_block_ips else best_ip
        if normalized.ip_address is None:
            return Action(action_type="analyze_log", log_id=best_log.log_id) if best_log else Action(action_type="ignore")
    elif action.action_type.value == "create_incident_report":
        normalized.report = " ".join((action.report or "").split()) or None
        if normalized.report is None:
            if best_alert and best_alert.source_ip:
                normalized.report = (
                    f"Incident report: attacker {best_alert.source_ip}, alert {best_alert.alert_id}, "
                    f"evidence {','.join(sorted(visible_log_ids))}, containment required."
                )
            else:
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
    if task_name is not None:
        guarded = _task_workflow_guardrail(task_name, observation, normalized, alerts_by_id, analyzed_log_ids or set())
        if guarded is not None:
            return guarded
    return normalized


def _task_workflow_guardrail(
    task_name: TaskName,
    observation: Observation,
    action: Action,
    alerts_by_id: dict[str, Alert],
    analyzed_log_ids: set[str],
) -> Action | None:
    """Keep local/provider policies on a semantic workflow without fixed fixture IDs."""
    feedback = observation.previous_action_feedback.lower()
    if task_name == TaskName.HARD and "escalation captured" in feedback and "incident report" not in feedback:
        return Action(action_type="create_incident_report", report=_build_report(task_name, observation, analyzed_log_ids))

    best_unanalyzed = _best_unanalyzed_log(observation, analyzed_log_ids)
    best_alert = _best_alert(observation)
    if best_alert and best_alert.status == "flagged" and best_alert.source_ip and best_alert.source_ip not in observation.blocked_ips:
        if action.action_type.value not in {"block_ip", "isolate_host"}:
            return Action(action_type="block_ip", ip_address=best_alert.source_ip)
    if task_name == TaskName.HARD and best_alert and best_alert.source_ip in observation.blocked_ips and "escalation captured" not in feedback:
        if action.action_type.value != "escalate":
            return Action(action_type="escalate")

    analyzed_signal = _best_analyzed_signal_log(observation, analyzed_log_ids)
    if analyzed_signal is not None:
        same_source = _best_same_source_unanalyzed_log(observation, analyzed_log_ids, analyzed_signal.source_ip)
        if same_source is not None and _signal_score(f"{same_source.event_type} {same_source.message}") > 0:
            if action.action_type.value != "analyze_log" or action.log_id != same_source.log_id:
                return Action(action_type="analyze_log", log_id=same_source.log_id)
        source_query = f"source_ip={analyzed_signal.source_ip}"
        query_seen = any(source_query in note for note in observation.analyst_notes)
        if "not decisive until correlated" in feedback and not query_seen:
            if action.action_type.value != "query_logs" or action.query != source_query:
                return Action(action_type="query_logs", query=source_query)

    if best_unanalyzed and observation.step_count < 2:
        if action.action_type.value != "analyze_log" or action.log_id != best_unanalyzed.log_id:
            return Action(action_type="analyze_log", log_id=best_unanalyzed.log_id)
    if best_unanalyzed and "found attack telemetry" in feedback and _signal_score(f"{best_unanalyzed.event_type} {best_unanalyzed.message}") > 0:
        if action.action_type.value != "analyze_log" or action.log_id != best_unanalyzed.log_id:
            return Action(action_type="analyze_log", log_id=best_unanalyzed.log_id)

    if best_alert is None or observation.step_count < 2:
        return None
    if best_alert.status != "flagged" and action.action_type.value != "flag_alert":
        return Action(action_type="flag_alert", alert_id=best_alert.alert_id)
    return None


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


def request_model_action(
    client: OpenAI,
    model_name: str,
    task_name: TaskName,
    observation: Observation,
    analyzed_log_ids: set[str] | None = None,
) -> Action:
    """Query the provider and normalize the returned action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_to_prompt(task_name, observation)},
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=DEFAULT_MODEL_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        if "response_format.type" not in str(exc):
            raise
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=DEFAULT_MODEL_MAX_TOKENS,
            response_format={"type": "text"},
            messages=messages,
        )
    payload = parse_action_payload(response.choices[0].message.content)
    return sanitize_action(Action.model_validate(payload), observation, task_name=task_name, analyzed_log_ids=analyzed_log_ids)


def build_openai_policy() -> Callable[[TaskName, object], Action]:
    """Create a model policy backed by the OpenAI client."""
    api_base_url = os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL
    model_name = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-70B-Instruct"
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"

    client = OpenAI(base_url=api_base_url, api_key=api_key)
    analyzed_by_task: dict[TaskName, set[str]] = {task_name: set() for task_name in TaskName}

    def policy(task_name: TaskName, observation) -> Action:
        if observation.step_count == 0:
            analyzed_by_task[task_name].clear()
        action = request_model_action(client, model_name, task_name, observation, analyzed_by_task[task_name])
        if "found attack telemetry" in observation.previous_action_feedback.lower():
            follow_up_log = _best_unanalyzed_log(observation, analyzed_by_task[task_name])
            if follow_up_log and _signal_score(f"{follow_up_log.event_type} {follow_up_log.message}") > 0:
                action = Action(action_type="analyze_log", log_id=follow_up_log.log_id)
        if action.action_type == ActionType.ANALYZE_LOG and action.log_id:
            analyzed_by_task[task_name].add(action.log_id)
        return action

    return policy


def build_heuristic_policy() -> Callable[[TaskName, object], Action]:
    """Deterministic fallback policy useful for local smoke tests."""
    analyzed_by_task: dict[TaskName, set[str]] = {task_name: set() for task_name in TaskName}
    queried_by_task: dict[TaskName, set[str]] = {task_name: set() for task_name in TaskName}
    escalated_tasks: set[TaskName] = set()

    def policy(task_name: TaskName, observation) -> Action:
        blocked = set(observation.blocked_ips)
        best_alert = _best_alert(observation)
        best_unanalyzed = _best_unanalyzed_log(observation, analyzed_by_task[task_name])
        step = observation.step_count
        if step == 0:
            analyzed_by_task[task_name].clear()
            queried_by_task[task_name].clear()
            escalated_tasks.discard(task_name)

        def analyze(log_id: str) -> Action:
            analyzed_by_task[task_name].add(log_id)
            return Action(action_type="analyze_log", log_id=log_id)

        if task_name == TaskName.HARD and "escalation captured" in observation.previous_action_feedback.lower():
            return Action(action_type="create_incident_report", report=_build_report(task_name, observation, analyzed_by_task[task_name]))

        if best_alert and best_alert.status == "flagged" and best_alert.source_ip and best_alert.source_ip not in blocked:
            return Action(action_type="block_ip", ip_address=best_alert.source_ip)
        if (
            task_name == TaskName.HARD
            and best_alert
            and best_alert.status == "flagged"
            and best_alert.source_ip in blocked
            and task_name not in escalated_tasks
        ):
            escalated_tasks.add(task_name)
            return Action(action_type="escalate")

        analyzed_signal = _best_analyzed_signal_log(observation, analyzed_by_task[task_name])
        if analyzed_signal is not None:
            same_source = _best_same_source_unanalyzed_log(observation, analyzed_by_task[task_name], analyzed_signal.source_ip)
            if same_source is not None and _signal_score(f"{same_source.event_type} {same_source.message}") > 0:
                return analyze(same_source.log_id)
            source_query = f"source_ip={analyzed_signal.source_ip}"
            if "not decisive until correlated" in observation.previous_action_feedback.lower() and source_query not in queried_by_task[task_name]:
                queried_by_task[task_name].add(source_query)
                return Action(action_type="query_logs", query=source_query)

        if best_unanalyzed and "found attack telemetry" in observation.previous_action_feedback.lower():
            if _signal_score(f"{best_unanalyzed.event_type} {best_unanalyzed.message}") > 0:
                return analyze(best_unanalyzed.log_id)

        if best_alert and best_alert.status != "flagged" and step >= 2:
            return Action(action_type="flag_alert", alert_id=best_alert.alert_id)
        if best_unanalyzed and (step < 2 or _signal_score(f"{best_unanalyzed.event_type} {best_unanalyzed.message}") > 0):
            return analyze(best_unanalyzed.log_id)
        query = "severity>=high"
        if query not in queried_by_task[task_name]:
            queried_by_task[task_name].add(query)
            return Action(action_type="query_logs", query=query)
        return Action(action_type="ignore")

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

    print(f"[START] task={task_name.value} env={env.env_name} model={model_name}", flush=True)

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
                action = Action(action_type="ignore")
                observation, reward, done, info = env.step(action)
                error_value = str(exc)

            rewards.append(f"{reward:.2f}")
            steps_taken = env.state().steps_taken
            print(
                f"[STEP] step={step_number} action={stringify_action(action)} "
                f"reward={reward:.2f} done={str(done).lower()} error={format_error(error_value)}",
                flush=True
            )
            if done:
                success = bool(info["resolved"])
                break

        score = env.grade().score
        steps_taken = env.state().steps_taken
        exit_code = 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            env.close()
        except Exception:
            pass
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.3f} rewards={','.join(rewards)}",
            flush=True
        )
    return exit_code


def resolve_tasks(task_selection: str) -> list[TaskName]:
    """Resolve the CLI task selection into one or more task episodes."""
    if task_selection == ALL_TASKS:
        return list(TaskName)
    return [TaskName(task_selection)]


def _default_task_selection() -> str:
    """Read the optional task override while keeping all tasks as the default."""
    selection = os.getenv("OPENENV_TASK", ALL_TASKS).strip().lower() or ALL_TASKS
    choices = {ALL_TASKS, *(task.value for task in TaskName)}
    return selection if selection in choices else ALL_TASKS


def main() -> int:
    """CLI entrypoint for submission inference."""
    parser = argparse.ArgumentParser(description="Run inference against the OpenEnv security incident environment.")
    parser.add_argument(
        "--task",
        choices=[ALL_TASKS, *(task.value for task in TaskName)],
        default=_default_task_selection(),
        help="Run one task or all declared tasks. Defaults to all.",
    )
    parser.add_argument(
        "--policy",
        choices=["openai", "heuristic"],
        default="openai",
        help="Use the OpenAI client or a local heuristic smoke-test policy.",
    )
    args = parser.parse_args()

    exit_code = 0
    for task_name in resolve_tasks(args.task):
        episode_exit_code = run_episode(task_name, use_heuristic=args.policy == "heuristic")
        if episode_exit_code != 0:
            exit_code = episode_exit_code
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
