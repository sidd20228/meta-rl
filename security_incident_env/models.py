"""Typed models shared by the environment, API, and inference runner."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TaskName(str, Enum):
    """Supported task names."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    """Agent actions supported by the environment."""

    ANALYZE_LOG = "analyze_log"
    FLAG_ALERT = "flag_alert"
    BLOCK_IP = "block_ip"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class Severity(str, Enum):
    """Severity labels for logs and alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackStage(str, Enum):
    """Canonical attack stages used in the environment state."""

    NONE = "none"
    RECONNAISSANCE = "reconnaissance"
    EXPLOITATION = "exploitation"
    PERSISTENCE = "persistence"
    CONTAINED = "contained"


class JudgePhase(str, Enum):
    """Workflow phases used by the semantic judge."""

    TRIAGE = "triage"
    INVESTIGATION = "investigation"
    MITIGATION = "mitigation"
    RESOLUTION = "resolution"


class Action(BaseModel):
    """Structured action object following the OpenEnv action schema."""

    action_type: ActionType
    log_id: Optional[str] = None
    ip_address: Optional[str] = None
    alert_id: Optional[str] = None


class LogEntry(BaseModel):
    """A structured SOC log entry."""

    log_id: str
    timestamp: str
    source_ip: str
    event_type: str
    severity: Severity
    message: str


class Alert(BaseModel):
    """Security alert visible to the agent."""

    alert_id: str
    name: str
    severity: Severity
    status: str
    related_log_ids: List[str] = Field(default_factory=list)
    source_ip: Optional[str] = None
    summary: str


class Observation(BaseModel):
    """Agent-visible observation at each time step."""

    current_logs: List[LogEntry]
    active_alerts: List[Alert]
    blocked_ips: List[str]
    remaining_budget: int
    max_budget: int
    step_count: int
    previous_action_feedback: str
    visible_history_count: int = 0
    log_window_size: int = 5
    context_truncated: bool = False


class State(BaseModel):
    """Full internal state, including hidden labels and trajectory metrics."""

    env_name: str
    task_name: TaskName
    scenario_seed: int
    attack_path: str
    branch_description: str
    full_log_history: List[LogEntry]
    remaining_log_queue: List[LogEntry]
    malicious_log_ids: List[str]
    malicious_ips: List[str]
    benign_ips: List[str]
    decoy_log_ids: List[str]
    decoy_alert_ids: List[str]
    log_stage_map: Dict[str, AttackStage]
    attack_progression_stage: AttackStage
    steps_taken: int
    incident_resolved: bool
    analyzed_log_ids: List[str]
    flagged_alert_ids: List[str]
    blocked_ips: List[str]
    false_positive_blocks: List[str]
    decoy_ips: List[str]
    escalation_sent: bool
    action_history: List[Action]
    observation_history: List["Observation"]
    feedback_history: List[str]
    reward_history: List[float]
    raw_reward_history: List[float]
    cumulative_raw_reward: float
    invalid_action_count: int
    redundant_action_count: int
    useless_step_count: int
    missed_signal_count: int
    attack_progressions: int
    timeliness_penalty: float
    score_cap: float
    max_steps: int
    max_budget: int
    remaining_budget: int
    budget_exhausted: bool
    window_size: int
    optimal_steps: int
    action_costs: Dict[str, int]
    evidence_groups: List[List[str]]
    required_analysis_log_ids: List[str]
    required_alert_ids: List[str]
    required_block_ips: List[str]
    requires_escalation: bool
    premature_escalation_count: int = 0
    ignored_attack_count: int = 0
    resolved_step: Optional[int] = None


class JudgePhaseQuality(BaseModel):
    """Phase-by-phase semantic quality scores in the judge domain."""

    triage: float
    investigation: float
    mitigation: float
    resolution: float


class JudgePersonaResult(BaseModel):
    """Single-persona judge review."""

    persona: str
    weight: float
    score: float
    reasoning: str
    phase_classification: JudgePhase
    phase_quality: JudgePhaseQuality
    used_llm: bool = False


class JudgeResult(BaseModel):
    """Aggregated multi-persona judge output."""

    score: float
    normalized_score: float
    explanation: str
    phase_classification: JudgePhase
    phase_quality: JudgePhaseQuality
    personas: List[JudgePersonaResult] = Field(default_factory=list)
    used_llm: bool = False
    fallback_reason: Optional[str] = None


class ScenarioDefinition(BaseModel):
    """Concrete seeded scenario definition used to initialize an episode."""

    task_name: TaskName
    scenario_seed: int
    attack_path: str
    branch_description: str
    description: str
    logs: List[LogEntry]
    alerts: List[Alert]
    malicious_log_ids: List[str]
    malicious_ips: List[str]
    benign_ips: List[str]
    decoy_ips: List[str] = Field(default_factory=list)
    decoy_log_ids: List[str]
    decoy_alert_ids: List[str]
    log_stage_map: Dict[str, AttackStage]
    evidence_groups: List[List[str]] = Field(default_factory=list)
    required_analysis_log_ids: List[str]
    required_alert_ids: List[str]
    required_block_ips: List[str]
    requires_escalation: bool = False
    optimal_steps: int = 5
    initial_visible_log_count: int = 5
    max_steps: int = 8
    max_budget: int = 7

    @field_validator(
        "required_analysis_log_ids",
        "required_alert_ids",
        "required_block_ips",
        "malicious_log_ids",
        "malicious_ips",
        "benign_ips",
        "decoy_ips",
        "decoy_log_ids",
        "decoy_alert_ids",
    )
    @classmethod
    def sort_required_fields(cls, value: List[str]) -> List[str]:
        return sorted(value)


class StepInfo(BaseModel):
    """Additional per-step details exposed alongside the main reward."""

    raw_reward: float
    cumulative_reward: float
    budget_remaining: int
    budget_exhausted: bool
    resolved: bool
    attack_stage: AttackStage
    invalid_action: bool
    score: float
    programmatic_score: float
    llm_judge_score: float
    final_reward: float
    feedback: str
    score_cap: float
    judge_explanation: Optional[str] = None
    judge_phase_classification: Optional[JudgePhase] = None
    judge_phase_quality: Optional[JudgePhaseQuality] = None


class EpisodeGrade(BaseModel):
    """Combined grading output for an episode."""

    task_name: TaskName
    score: float
    programmatic_score: float
    llm_judge_score: float
    llm_judge_normalized_score: float
    environment_reward: float
    final_reward: float
    correctness: float
    efficiency: float
    timeliness: float
    trajectory_quality: float
    false_positive_penalty: float
    resolved: bool
    steps_taken: int
    score_cap: float
    judge_used_llm: bool = False
    judge_explanation: Optional[str] = None
    judge_phase_classification: Optional[JudgePhase] = None
    judge_phase_quality: Optional[JudgePhaseQuality] = None
    judge_fallback_reason: Optional[str] = None


class ResetRequest(BaseModel):
    """Request payload for the API reset endpoint."""

    task_name: TaskName = TaskName.EASY


class ResetResponse(BaseModel):
    """Response payload for the API reset endpoint."""

    session_id: str
    observation: Observation
    env_name: str
    task_name: TaskName
    max_steps: int


class StepRequest(BaseModel):
    """Request payload for the API step endpoint."""

    session_id: str
    action: Action


class StepResponse(BaseModel):
    """Response payload for the API step endpoint."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, object]


class PolicyActionResponse(BaseModel):
    """Expected JSON output from the model-driven policy in inference.py."""

    action_type: ActionType
    log_id: Optional[str] = None
    ip_address: Optional[str] = None
    alert_id: Optional[str] = None
