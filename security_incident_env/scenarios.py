"""Seeded scenario generation for deterministic but branched security incidents."""

from __future__ import annotations

from datetime import datetime, timedelta
from random import Random
from typing import Iterable

from .config import DEFAULT_CONFIG, EnvironmentConfig
from .models import Alert, AttackStage, LogEntry, ScenarioDefinition, Severity, TaskName


TASK_OFFSETS = {
    TaskName.EASY: 101,
    TaskName.MEDIUM: 211,
    TaskName.HARD: 307,
}

BENIGN_IP_POOL = [
    "198.51.100.10",
    "198.51.100.52",
    "198.51.100.81",
    "203.0.113.10",
    "203.0.113.44",
    "192.0.2.16",
    "192.0.2.45",
    "10.0.8.25",
    "10.0.12.17",
    "10.0.14.9",
]


def build_scenarios(seed: int | None = None, config: EnvironmentConfig | None = None) -> dict[TaskName, ScenarioDefinition]:
    """Build one deterministic scenario per task for the provided seed."""
    config = config or DEFAULT_CONFIG
    base_seed = config.seed if seed is None else seed
    return {
        task_name: build_scenario(task_name, seed=base_seed + TASK_OFFSETS[task_name], config=config)
        for task_name in TaskName
    }


def build_scenario(task_name: TaskName, seed: int, config: EnvironmentConfig | None = None) -> ScenarioDefinition:
    """Build a concrete scenario for a single task."""
    config = config or DEFAULT_CONFIG
    rng = Random(seed)
    if task_name == TaskName.EASY:
        scenario = _build_easy_scenario(rng, seed, config)
    elif task_name == TaskName.MEDIUM:
        scenario = _build_medium_scenario(rng, seed, config)
    else:
        scenario = _build_hard_scenario(rng, seed, config)
    if not config.randomize_identifiers:
        return scenario
    return _randomize_identifiers(scenario, seed)


def _pick_benign_ips(rng: Random, count: int, blocked: Iterable[str] | None = None) -> list[str]:
    blocked_set = set(blocked or [])
    pool = [ip for ip in BENIGN_IP_POOL if ip not in blocked_set]
    rng.shuffle(pool)
    return pool[:count]


def _randomize_identifiers(scenario: ScenarioDefinition, seed: int) -> ScenarioDefinition:
    """Remap visible IDs and IPs so agents cannot memorize canonical fixtures."""
    rng = Random(seed + 7919)
    sorted_logs = sorted(scenario.logs, key=lambda item: (item.timestamp, item.log_id))
    log_id_map: dict[str, str] = {}
    used_log_ids: set[str] = set()
    for log in sorted_logs:
        log_id_map[log.log_id] = _next_id(rng, "L", used_log_ids)

    alert_id_map: dict[str, str] = {}
    used_alert_ids: set[str] = set()
    for alert in sorted(scenario.alerts, key=lambda item: item.alert_id):
        alert_id_map[alert.alert_id] = _next_id(rng, "A", used_alert_ids)

    ip_map: dict[str, str] = {}
    used_ips: set[str] = set()
    all_ips = sorted(
        {
            *(log.source_ip for log in scenario.logs),
            *(alert.source_ip for alert in scenario.alerts if alert.source_ip),
            *scenario.malicious_ips,
            *scenario.benign_ips,
            *scenario.decoy_ips,
            *scenario.required_block_ips,
        }
    )
    for ip_address in all_ips:
        ip_map[ip_address] = _next_ip(rng, used_ips)

    remapped_logs = [
        log.model_copy(update={"log_id": log_id_map[log.log_id], "source_ip": ip_map[log.source_ip]})
        for log in scenario.logs
    ]
    remapped_alerts = [
        alert.model_copy(
            update={
                "alert_id": alert_id_map[alert.alert_id],
                "related_log_ids": [log_id_map[log_id] for log_id in alert.related_log_ids],
                "source_ip": ip_map.get(alert.source_ip) if alert.source_ip else None,
            }
        )
        for alert in scenario.alerts
    ]
    return scenario.model_copy(
        update={
            "logs": remapped_logs,
            "alerts": remapped_alerts,
            "malicious_log_ids": [log_id_map[log_id] for log_id in scenario.malicious_log_ids],
            "malicious_ips": [ip_map[ip_address] for ip_address in scenario.malicious_ips],
            "benign_ips": [ip_map[ip_address] for ip_address in scenario.benign_ips],
            "decoy_ips": [ip_map[ip_address] for ip_address in scenario.decoy_ips],
            "decoy_log_ids": [log_id_map[log_id] for log_id in scenario.decoy_log_ids],
            "decoy_alert_ids": [alert_id_map[alert_id] for alert_id in scenario.decoy_alert_ids],
            "log_stage_map": {log_id_map[log_id]: stage for log_id, stage in scenario.log_stage_map.items()},
            "evidence_groups": [[log_id_map[log_id] for log_id in group] for group in scenario.evidence_groups],
            "required_analysis_log_ids": [log_id_map[log_id] for log_id in scenario.required_analysis_log_ids],
            "required_alert_ids": [alert_id_map[alert_id] for alert_id in scenario.required_alert_ids],
            "required_block_ips": [ip_map[ip_address] for ip_address in scenario.required_block_ips],
        }
    )


def _next_id(rng: Random, prefix: str, used: set[str]) -> str:
    while True:
        candidate = f"{prefix}{rng.randrange(1000, 9999)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _next_ip(rng: Random, used: set[str]) -> str:
    ranges = ("198.51.100", "203.0.113", "192.0.2")
    while True:
        candidate = f"{ranges[rng.randrange(len(ranges))]}.{rng.randrange(10, 240)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _ts(base_time: datetime, seconds_offset: int) -> str:
    return (base_time + timedelta(seconds=seconds_offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(
    log_id: str,
    base_time: datetime,
    seconds_offset: int,
    source_ip: str,
    event_type: str,
    severity: Severity,
    message: str,
) -> LogEntry:
    return LogEntry(
        log_id=log_id,
        timestamp=_ts(base_time, seconds_offset),
        source_ip=source_ip,
        event_type=event_type,
        severity=severity,
        message=message,
    )


def _scenario_budget(config: EnvironmentConfig, minimum_budget: int) -> int:
    """Keep scenarios solvable while still respecting the configured budget pressure."""
    return max(minimum_budget, config.max_budget)


def _target_log_count(config: EnvironmentConfig, base_target: int) -> int:
    noise_padding = max(0, int(round(config.noise_level)))
    return max(config.total_logs, base_target + noise_padding)


def _append_decoy_noise(
    logs: list[LogEntry],
    rng: Random,
    decoy_ips: list[str],
    base_time: datetime,
    next_id: int,
    start_offset: int,
    theme: str,
) -> tuple[list[LogEntry], list[str], int]:
    """Inject additional suspicious but benign telemetry for configured decoys."""
    templates = {
        "identity": [
            "Identity provider replayed cached assertions during a certificate rollover, creating suspicious retries.",
            "Privileged SSO checks retried from a vendor NAT after a maintenance freeze lifted.",
            "Passwordless bootstrap flow emitted repeated challenge failures for a contractor pilot tenant.",
        ],
        "app": [
            "Approved external test harness replayed archived payload signatures against the staging edge.",
            "Vendor load-test node mirrored malformed API requests that resemble exploit traffic but match a ticketed exercise.",
            "Synthetic browser session retried bulk export endpoints during a resilience drill.",
        ],
    }
    messages = templates[theme]
    decoy_log_ids: list[str] = []
    offset = start_offset
    for index, ip in enumerate(decoy_ips):
        log_id = f"L{next_id}"
        logs.append(
            _log(
                log_id,
                base_time,
                offset,
                ip,
                "decoy_noise",
                Severity.MEDIUM if index % 2 == 0 else Severity.HIGH,
                messages[index % len(messages)],
            )
        )
        decoy_log_ids.append(log_id)
        next_id += 1
        offset += rng.randint(29, 61)
    return sorted(logs, key=lambda item: item.timestamp), decoy_log_ids, next_id


def _append_benign_fillers(
    logs: list[LogEntry],
    rng: Random,
    benign_ips: list[str],
    base_time: datetime,
    target_count: int,
    next_id: int,
    start_offset: int,
) -> tuple[list[LogEntry], int]:
    """Pad a scenario with benign activity until it reaches the requested size."""
    filler_messages = [
        ("browser_request", Severity.LOW, "Authenticated employee requested the dashboard landing page."),
        ("api_call", Severity.LOW, "Internal inventory API request returned HTTP 200."),
        ("mfa_success", Severity.LOW, "User completed MFA challenge from a managed endpoint."),
        ("health_probe", Severity.LOW, "Application health probe succeeded from monitoring host."),
        ("cdn_fetch", Severity.LOW, "Static asset request served from the edge cache."),
    ]
    offset = start_offset
    while len(logs) < target_count:
        event_type, severity, message = filler_messages[(next_id + len(logs)) % len(filler_messages)]
        ip = benign_ips[(next_id + len(logs)) % len(benign_ips)]
        logs.append(_log(f"L{next_id}", base_time, offset, ip, event_type, severity, message))
        next_id += 1
        offset += rng.randint(25, 65)
    return sorted(logs, key=lambda item: item.timestamp), next_id


def _build_easy_scenario(rng: Random, seed: int, config: EnvironmentConfig) -> ScenarioDefinition:
    attacker_ip = "198.51.100.24"
    benign_ips = _pick_benign_ips(rng, 7 + max(0, config.num_decoys - 1), blocked=[attacker_ip])
    decoy_ip = benign_ips[0]
    traveler_ip = benign_ips[1]
    base_time = datetime(2026, 4, 8, 9, 0, 0)
    attack_path = ["password_spray", "vpn_token_reuse"][rng.randrange(2)]

    logs = [
        _log("L103", base_time, 0, benign_ips[2], "browser_request", Severity.LOW, "Customer support agent opened the internal dashboard."),
        _log("L104", base_time, 46, benign_ips[3], "api_call", Severity.LOW, "Scheduled CRM synchronization completed successfully."),
        _log(
            "L100",
            base_time,
            95,
            attacker_ip,
            "login_failure",
            Severity.HIGH,
            "Ten failed administrator login attempts arrived within one minute from a public host.",
        ),
        _log(
            "L105",
            base_time,
            150,
            decoy_ip,
            "admin_retry_noise",
            Severity.MEDIUM,
            "Privileged sign-in retries came from a contractor VPN range during an approved identity provider maintenance window.",
        ),
        _log(
            "L101",
            base_time,
            212,
            attacker_ip,
            "credential_attack" if attack_path == "password_spray" else "vpn_token_reuse",
            Severity.HIGH,
            "Repeated privileged authentication attempts continued after the initial burst."
            if attack_path == "password_spray"
            else "Old VPN session token was replayed after a burst of failed password attempts.",
        ),
        _log(
            "L102",
            base_time,
            258,
            traveler_ip,
            "geo_mismatch",
            Severity.MEDIUM,
            "Sales employee authenticated from a new country using an enrolled laptop and approved travel itinerary.",
        ),
        _log(
            "L108",
            base_time,
            320,
            attacker_ip,
            "login_success",
            Severity.CRITICAL,
            "Administrator login succeeded after repeated failures and immediately requested privileged resources.",
        ),
        _log(
            "L107",
            base_time,
            378,
            decoy_ip,
            "token_refresh_noise",
            Severity.MEDIUM,
            "A stale privileged SSO assertion retried from the same contractor NAT after certificate rotation completed.",
        ),
        _log("L109", base_time, 432, benign_ips[4], "service_health", Severity.LOW, "Background health-check reported nominal application latency."),
    ]

    extra_decoy_logs: list[str] = []
    if config.num_decoys > 1:
        extra_ips = benign_ips[5 : 5 + (config.num_decoys - 1)]
        logs, extra_decoy_logs, next_id = _append_decoy_noise(
            logs=logs,
            rng=rng,
            decoy_ips=extra_ips,
            base_time=base_time,
            next_id=180,
            start_offset=470,
            theme="identity",
        )
    else:
        next_id = 180

    logs, _ = _append_benign_fillers(
        logs,
        rng,
        benign_ips,
        base_time,
        _target_log_count(config, 10),
        next_id,
        520,
    )

    alerts = [
        Alert(
            alert_id="A100",
            name="Privileged authentication attack",
            severity=Severity.HIGH,
            status="open",
            related_log_ids=["L100", "L101", "L108"],
            source_ip=attacker_ip,
            summary="A single external IP is repeatedly targeting privileged authentication paths with escalating confidence.",
        ),
        Alert(
            alert_id="A101",
            name="Privileged contractor retry review",
            severity=Severity.MEDIUM,
            status="open",
            related_log_ids=["L105", "L107"],
            source_ip=decoy_ip,
            summary="This contractor VPN range often looks hostile during approved identity maintenance windows.",
        ),
        Alert(
            alert_id="A102",
            name="Travel login anomaly",
            severity=Severity.MEDIUM,
            status="open",
            related_log_ids=["L102"],
            source_ip=traveler_ip,
            summary="A legitimate user appears to be traveling and triggered a location anomaly.",
        ),
    ]

    return ScenarioDefinition(
        task_name=TaskName.EASY,
        scenario_seed=seed,
        attack_path=attack_path,
        branch_description="Privileged auth abuse is mixed with benign contractor retry noise from a second suspicious IP.",
        description="Identify the real attacker while avoiding a convincing privileged-auth decoy and a benign travel anomaly.",
        logs=logs,
        alerts=alerts,
        malicious_log_ids=["L100", "L101", "L108"],
        malicious_ips=[attacker_ip],
        benign_ips=benign_ips,
        decoy_ips=[decoy_ip],
        decoy_log_ids=["L102", "L105", "L107", *extra_decoy_logs],
        decoy_alert_ids=["A101", "A102"],
        log_stage_map={
            "L100": AttackStage.RECONNAISSANCE,
            "L101": AttackStage.EXPLOITATION,
            "L108": AttackStage.PERSISTENCE,
        },
        evidence_groups=[["L100", "L101"]],
        required_analysis_log_ids=["L100", "L101"],
        required_alert_ids=["A100"],
        required_block_ips=[attacker_ip],
        requires_escalation=False,
        optimal_steps=4,
        initial_visible_log_count=min(config.initial_visible_logs, 4),
        max_steps=min(config.max_steps, 6),
        max_budget=_scenario_budget(config, 5),
    )


def _build_medium_scenario(rng: Random, seed: int, config: EnvironmentConfig) -> ScenarioDefinition:
    attacker_ip = "203.0.113.77"
    benign_ips = _pick_benign_ips(rng, 7 + max(0, config.num_decoys - 1), blocked=[attacker_ip])
    decoy_ip = benign_ips[0]
    traveler_ip = benign_ips[1]
    qa_ip = benign_ips[2]
    base_time = datetime(2026, 4, 8, 11, 10, 0)
    attack_path = ["sql_injection", "credential_stuffing"][rng.randrange(2)]

    recon_message = (
        "External host enumerated application ports and probed the customer support subnet."
        if attack_path == "sql_injection"
        else "External host enumerated login endpoints and cycled through common usernames."
    )
    exploit_message = (
        "Repeated SQL injection payloads targeted the /tickets/search endpoint."
        if attack_path == "sql_injection"
        else "Burst of failed OAuth password grants suggests credential stuffing against the login service."
    )
    impact_message = (
        "Database error rate spiked immediately after malformed search payloads reached the application tier."
        if attack_path == "sql_injection"
        else "One employee account authenticated successfully after repeated failures and requested sensitive ticket exports."
    )

    logs = [
        _log("L203", base_time, 0, benign_ips[3], "service_health", Severity.LOW, "Application health-check completed from the monitoring subnet."),
        _log(
            "L205",
            base_time,
            44,
            qa_ip,
            "waf_test",
            Severity.MEDIUM,
            "QA automation submitted malformed payloads during a scheduled regression window.",
        ),
        _log("L200", base_time, 96, attacker_ip, "recon_activity", Severity.MEDIUM, recon_message),
        _log(
            "L206",
            base_time,
            143,
            decoy_ip,
            "external_probe_noise",
            Severity.HIGH,
            "Approved partner scanner replayed archived exploit payloads from a public NAT range after a routing failover.",
        ),
        _log("L204", base_time, 187, benign_ips[4], "api_call", Severity.LOW, "Customer billing API returned a normal response."),
        _log(
            "L207",
            base_time,
            236,
            traveler_ip,
            "geo_mismatch",
            Severity.MEDIUM,
            "Support engineer authenticated from a new location after a documented travel approval.",
        ),
        _log("L201", base_time, 284, attacker_ip, "web_attack", Severity.HIGH, exploit_message),
        _log("L208", base_time, 338, benign_ips[5], "cdn_fetch", Severity.LOW, "Static asset request served from the CDN edge."),
        _log("L202", base_time, 392, attacker_ip, "impact_signal", Severity.HIGH, impact_message),
        _log(
            "L209",
            base_time,
            448,
            decoy_ip,
            "bulk_export_noise",
            Severity.MEDIUM,
            "Partner validation suite requested bulk exports that resemble staging abuse but match a change ticket.",
        ),
    ]

    extra_decoy_logs: list[str] = []
    if config.num_decoys > 1:
        extra_ips = benign_ips[6 : 6 + (config.num_decoys - 1)]
        logs, extra_decoy_logs, next_id = _append_decoy_noise(
            logs=logs,
            rng=rng,
            decoy_ips=extra_ips,
            base_time=base_time,
            next_id=280,
            start_offset=500,
            theme="app",
        )
    else:
        next_id = 280

    logs, _ = _append_benign_fillers(
        logs,
        rng,
        benign_ips,
        base_time,
        _target_log_count(config, 11),
        next_id,
        560,
    )

    alerts = [
        Alert(
            alert_id="A200",
            name="Correlated external intrusion",
            severity=Severity.HIGH,
            status="open",
            related_log_ids=["L201", "L202"],
            source_ip=attacker_ip,
            summary="External source is combining exploit activity with impact signals against the production application.",
        ),
        Alert(
            alert_id="A201",
            name="Partner scanner review",
            severity=Severity.HIGH,
            status="open",
            related_log_ids=["L206", "L209"],
            source_ip=decoy_ip,
            summary="An approved partner scanner looks hostile from the edge and often generates false-positive reviews.",
        ),
        Alert(
            alert_id="A202",
            name="Travel anomaly alert",
            severity=Severity.MEDIUM,
            status="open",
            related_log_ids=["L207"],
            source_ip=traveler_ip,
            summary="Legitimate travel can trigger this alert and it often creates false positives.",
        ),
    ]

    return ScenarioDefinition(
        task_name=TaskName.MEDIUM,
        scenario_seed=seed,
        attack_path=attack_path,
        branch_description="Real exploit traffic competes with a second public IP that generates ticketed but hostile-looking scanner noise.",
        description="Correlate the real attacker across recon and exploit evidence while ignoring a believable external scanner decoy.",
        logs=logs,
        alerts=alerts,
        malicious_log_ids=["L200", "L201", "L202"],
        malicious_ips=[attacker_ip],
        benign_ips=benign_ips,
        decoy_ips=[decoy_ip],
        decoy_log_ids=["L205", "L206", "L207", "L209", *extra_decoy_logs],
        decoy_alert_ids=["A201", "A202"],
        log_stage_map={
            "L200": AttackStage.RECONNAISSANCE,
            "L201": AttackStage.EXPLOITATION,
            "L202": AttackStage.PERSISTENCE,
        },
        evidence_groups=[["L200", "L201"]],
        required_analysis_log_ids=["L200", "L201"],
        required_alert_ids=["A200"],
        required_block_ips=[attacker_ip],
        requires_escalation=False,
        optimal_steps=4,
        initial_visible_log_count=min(config.initial_visible_logs, 4),
        max_steps=min(config.max_steps, 8),
        max_budget=_scenario_budget(config, 6),
    )


def _build_hard_scenario(rng: Random, seed: int, config: EnvironmentConfig) -> ScenarioDefinition:
    attacker_ip = "203.0.113.200"
    decoy_ip = "198.51.100.88"
    benign_ips = _pick_benign_ips(rng, 7 + max(0, config.num_decoys - 1), blocked=[attacker_ip, decoy_ip])
    traveler_ip = benign_ips[0]
    finance_ip = benign_ips[1]
    base_time = datetime(2026, 4, 8, 14, 0, 0)
    attack_path = ["api_worker_compromise", "identity_gateway_takeover"][rng.randrange(2)]

    l300_message = (
        "External host enumerated API worker aliases and staged low-volume credential probes against privileged service accounts."
        if attack_path == "api_worker_compromise"
        else "External host enumerated identity gateway recovery paths and tested low-volume password reset probes."
    )
    l302_message = (
        "The same external host pivoted into privileged API calls immediately after the earlier recon, confirming interactive exploitation."
        if attack_path == "api_worker_compromise"
        else "The same external host reused a recovered session and began calling privileged identity APIs after the earlier recon."
    )
    l306_message = (
        "Follow-on task creation and data staging requests indicate persistence on the production worker."
        if attack_path == "api_worker_compromise"
        else "Mailbox rule changes and staged export requests indicate persistence after identity takeover."
    )

    logs = [
        _log(
            "L303",
            base_time,
            0,
            decoy_ip,
            "vendor_validation",
            Severity.HIGH,
            "An approved external validation vendor replayed archived exploit signatures from a public range during a ticketed exercise.",
        ),
        _log("L304", base_time, 47, finance_ip, "browser_request", Severity.LOW, "Finance analyst opened the reporting dashboard from a managed workstation."),
        _log("L300", base_time, 94, attacker_ip, "recon_activity", Severity.MEDIUM, l300_message),
        _log(
            "L305",
            base_time,
            146,
            traveler_ip,
            "geo_mismatch",
            Severity.MEDIUM,
            "Support lead logged in from a hotel network while traveling for a customer workshop.",
        ),
        _log(
            "L301",
            base_time,
            206,
            decoy_ip,
            "exploit_like_noise",
            Severity.CRITICAL,
            "The validation vendor replayed serialized exploit payloads that mirror real attack signatures but align with the approved test window.",
        ),
        _log("L302", base_time, 262, attacker_ip, "privileged_api_abuse", Severity.HIGH, l302_message),
        _log("L307", base_time, 320, benign_ips[2], "api_call", Severity.LOW, "Inventory reconciliation API completed without errors."),
        _log("L308", base_time, 378, benign_ips[3], "service_restart", Severity.LOW, "Routine container restart completed after the patch rollout."),
        _log(
            "L309",
            base_time,
            438,
            finance_ip,
            "report_export",
            Severity.MEDIUM,
            "Finance analyst exported a quarterly revenue workbook during the close process.",
        ),
        _log("L306", base_time, 501, attacker_ip, "follow_on_activity", Severity.CRITICAL, l306_message),
        _log(
            "L310",
            base_time,
            556,
            decoy_ip,
            "beacon_like_noise",
            Severity.MEDIUM,
            "The vendor appliance checked back to its control plane from the same public NAT, resembling outbound beaconing.",
        ),
    ]

    extra_decoy_logs: list[str] = []
    if config.num_decoys > 1:
        extra_ips = benign_ips[4 : 4 + (config.num_decoys - 1)]
        logs, extra_decoy_logs, next_id = _append_decoy_noise(
            logs=logs,
            rng=rng,
            decoy_ips=extra_ips,
            base_time=base_time,
            next_id=380,
            start_offset=610,
            theme="app",
        )
    else:
        next_id = 380

    logs, _ = _append_benign_fillers(
        logs,
        rng,
        benign_ips,
        base_time,
        _target_log_count(config, 13),
        next_id,
        680,
    )

    alerts = [
        Alert(
            alert_id="A300",
            name="Vendor replay intrusion review",
            severity=Severity.CRITICAL,
            status="open",
            related_log_ids=["L301", "L303", "L310"],
            source_ip=decoy_ip,
            summary="A ticketed external validation range looks indistinguishable from a real exploit chain unless the analyst correlates the follow-on logs.",
        ),
        Alert(
            alert_id="A301",
            name="Confirmed intrusion chain",
            severity=Severity.CRITICAL,
            status="open",
            related_log_ids=["L302", "L306"],
            source_ip=attacker_ip,
            summary="The true attacker only becomes obvious after recon is paired with later privileged activity and persistence signals.",
        ),
        Alert(
            alert_id="A302",
            name="Bulk export review",
            severity=Severity.MEDIUM,
            status="open",
            related_log_ids=["L309"],
            source_ip=finance_ip,
            summary="Legitimate finance exports look noisy and are included as a decoy signal.",
        ),
    ]

    return ScenarioDefinition(
        task_name=TaskName.HARD,
        scenario_seed=seed,
        attack_path=attack_path,
        branch_description="A real attacker and a convincing external decoy compete for analyst attention across a multi-stage progression.",
        description="Correlate hidden dependencies across a limited log window, ignore the vendor replay decoy, contain the real attacker, and escalate only after evidence is sufficient.",
        logs=logs,
        alerts=alerts,
        malicious_log_ids=["L300", "L302", "L306"],
        malicious_ips=[attacker_ip],
        benign_ips=benign_ips,
        decoy_ips=[decoy_ip],
        decoy_log_ids=["L301", "L303", "L305", "L309", "L310", *extra_decoy_logs],
        decoy_alert_ids=["A300", "A302"],
        log_stage_map={
            "L300": AttackStage.RECONNAISSANCE,
            "L302": AttackStage.EXPLOITATION,
            "L306": AttackStage.PERSISTENCE,
        },
        evidence_groups=[["L300", "L302"]],
        required_analysis_log_ids=["L300", "L302"],
        required_alert_ids=["A301"],
        required_block_ips=[attacker_ip],
        requires_escalation=True,
        requires_report=True,
        optimal_steps=6,
        initial_visible_log_count=min(config.initial_visible_logs, 5),
        max_steps=min(config.max_steps, 8),
        max_budget=_scenario_budget(config, 8),
    )
