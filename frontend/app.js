const apiBase = window.location.origin.startsWith("file:")
  ? "http://localhost:7860"
  : window.location.origin;

const state = {
  sessionId: null,
  taskName: null,
  maxSteps: null,
  lastReward: null,
  lastScore: null,
  done: false,
  history: [],
};

const els = {
  taskSelect: document.getElementById("task-select"),
  resetButton: document.getElementById("reset-button"),
  stepButton: document.getElementById("step-button"),
  actionType: document.getElementById("action-type"),
  logId: document.getElementById("log-id"),
  alertId: document.getElementById("alert-id"),
  ipAddress: document.getElementById("ip-address"),
  sessionStatus: document.getElementById("session-status"),
  sessionMeta: document.getElementById("session-meta"),
  windowMeta: document.getElementById("window-meta"),
  blockedMeta: document.getElementById("blocked-meta"),
  budgetMeta: document.getElementById("budget-meta"),
  rewardMeta: document.getElementById("reward-meta"),
  doneMeta: document.getElementById("done-meta"),
  scoreMeta: document.getElementById("score-meta"),
  feedbackBox: document.getElementById("feedback-box"),
  judgeSummary: document.getElementById("judge-summary"),
  judgePhaseBreakdown: document.getElementById("judge-phase-breakdown"),
  finalSummary: document.getElementById("final-summary"),
  logsList: document.getElementById("logs-list"),
  alertsList: document.getElementById("alerts-list"),
  historyList: document.getElementById("history-list"),
};

els.resetButton.addEventListener("click", resetEnvironment);
els.stepButton.addEventListener("click", stepEnvironment);

renderEmptyState();

async function resetEnvironment() {
  setBusy(true);
  try {
    const taskName = els.taskSelect.value;
    const response = await fetch(`${apiBase}/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_name: taskName }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Reset failed.");
    }

    state.sessionId = payload.session_id;
    state.taskName = payload.task_name;
    state.maxSteps = payload.max_steps;
    state.lastReward = null;
    state.lastScore = null;
    state.done = false;
    state.history = [];

    updateSessionMeta();
    renderObservation(payload.observation, {
      reward: null,
      done: false,
      info: { feedback: payload.observation.previous_action_feedback, score: null, judge_explanation: null, judge_phase_quality: null },
    });
    renderHistory();
    clearInputs();
    els.stepButton.disabled = false;
  } catch (error) {
    showFeedback(error.message, true);
  } finally {
    setBusy(false);
  }
}

async function stepEnvironment() {
  if (!state.sessionId || state.done) {
    return;
  }

  setBusy(true);
  try {
    const action = buildActionPayload();
    const response = await fetch(`${apiBase}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        action,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Step failed.");
    }

    state.lastReward = payload.reward;
    state.lastScore = payload.info.score;
    state.done = payload.done;
    state.history.unshift({
      action,
      reward: payload.reward,
      feedback: payload.info.feedback,
      score: payload.info.score,
      finalReward: payload.info.final_reward,
      judgeExplanation: payload.info.judge_explanation,
      done: payload.done,
      budgetRemaining: payload.observation.remaining_budget,
    });

    renderObservation(payload.observation, payload);
    renderHistory();
    clearInputs();

    if (payload.done) {
      els.stepButton.disabled = true;
      renderFinalSummary(payload.info.score, payload.info.resolved, payload.observation.remaining_budget);
    }
  } catch (error) {
    showFeedback(error.message, true);
  } finally {
    setBusy(false);
  }
}

function buildActionPayload() {
  return {
    action_type: els.actionType.value,
    log_id: normalizeValue(els.logId.value),
    alert_id: normalizeValue(els.alertId.value),
    ip_address: normalizeValue(els.ipAddress.value),
  };
}

function renderObservation(observation, payload) {
  renderLogs(observation.current_logs || []);
  renderAlerts(observation.active_alerts || []);
  els.windowMeta.textContent = `Window: ${observation.current_logs.length}/${observation.log_window_size} visible, history ${observation.visible_history_count}`;
  els.blockedMeta.textContent = `Blocked IPs: ${(observation.blocked_ips || []).join(", ") || "none"}`;
  els.budgetMeta.textContent = `${observation.remaining_budget} / ${observation.max_budget}`;
  els.rewardMeta.textContent = payload.reward == null ? "-" : Number(payload.reward).toFixed(2);
  els.doneMeta.textContent = String(Boolean(payload.done));
  els.scoreMeta.textContent = payload.info.score == null ? "-" : Number(payload.info.score).toFixed(2);
  showFeedback(payload.info.feedback || observation.previous_action_feedback, false);
  renderJudge(payload.info);
  updateSessionMeta();

  if (!payload.done) {
    els.finalSummary.classList.add("hidden");
    els.finalSummary.textContent = "";
  }
}

function renderLogs(logs) {
  if (!logs.length) {
    els.logsList.innerHTML = `<div class="empty-state">No logs are currently visible.</div>`;
    return;
  }
  els.logsList.innerHTML = logs
    .map(
      (log) => `
        <div class="log-card severity-${log.severity}">
          <div class="meta-row">
            <span class="tag">${log.log_id}</span>
            <span>${log.timestamp}</span>
            <span>${log.source_ip}</span>
            <span>${log.event_type}</span>
          </div>
          <div>${escapeHtml(log.message)}</div>
        </div>
      `
    )
    .join("");
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    els.alertsList.innerHTML = `<div class="empty-state">No alerts are currently visible.</div>`;
    return;
  }
  els.alertsList.innerHTML = alerts
    .map(
      (alert) => `
        <div class="alert-card severity-${alert.severity}">
          <div class="meta-row">
            <span class="tag">${alert.alert_id}</span>
            <span>${escapeHtml(alert.name)}</span>
            <span>${alert.source_ip || "unknown source"}</span>
            <span>Status: ${escapeHtml(alert.status)}</span>
          </div>
          <div>${escapeHtml(alert.summary)}</div>
        </div>
      `
    )
    .join("");
}

function renderHistory() {
  if (!state.history.length) {
    els.historyList.innerHTML = `<div class="empty-state">No actions taken yet.</div>`;
    return;
  }
  els.historyList.innerHTML = state.history
    .map(
      (entry, index) => `
        <div class="history-card">
          <div class="meta-row">
            <span class="tag">Step ${state.history.length - index}</span>
            <span>reward ${Number(entry.reward).toFixed(2)}</span>
            <span>score ${Number(entry.score).toFixed(2)}</span>
            <span>final ${Number(entry.finalReward).toFixed(2)}</span>
            <span>budget ${entry.budgetRemaining}</span>
          </div>
          <div>${escapeHtml(JSON.stringify(entry.action))}</div>
          <div class="meta-row">
            <span>${escapeHtml(entry.feedback)}</span>
          </div>
        </div>
      `
    )
    .join("");
}

function renderFinalSummary(score, resolved, budgetRemaining) {
  els.finalSummary.classList.remove("hidden");
  els.finalSummary.textContent = resolved
    ? `Episode complete. Final score ${Number(score).toFixed(2)}. Incident resolved with ${budgetRemaining} budget remaining.`
    : `Episode complete. Final score ${Number(score).toFixed(2)}. Incident not fully resolved.`;
}

function updateSessionMeta() {
  if (!state.sessionId) {
    els.sessionStatus.textContent = "No active session";
    els.sessionMeta.textContent = "Reset an environment to begin.";
    return;
  }

  els.sessionStatus.textContent = `Session ${state.sessionId.slice(0, 8)} · task ${state.taskName}`;
  els.sessionMeta.textContent = `Max steps ${state.maxSteps}${state.done ? " · completed" : ""}`;
}

function renderJudge(info) {
  if (!info || !info.judge_explanation) {
    els.judgeSummary.textContent = "No semantic judgment yet.";
    els.judgePhaseBreakdown.innerHTML = "";
    return;
  }

  const phase = info.judge_phase_classification || "unknown";
  els.judgeSummary.textContent = `${phase} · judge ${Number(info.llm_judge_score || 0).toFixed(2)} · ${info.judge_explanation}`;

  const breakdown = info.judge_phase_quality || {};
  els.judgePhaseBreakdown.innerHTML = ["triage", "investigation", "mitigation", "resolution"]
    .map((name) => `<div><strong>${name}</strong>: ${Number(breakdown[name] || 0).toFixed(2)}</div>`)
    .join("");
}

function setBusy(isBusy) {
  els.resetButton.disabled = isBusy;
  els.stepButton.disabled = isBusy || !state.sessionId || state.done;
}

function showFeedback(message, isError) {
  els.feedbackBox.textContent = message;
  els.feedbackBox.style.background = isError ? "rgba(161, 29, 51, 0.12)" : "rgba(17, 75, 95, 0.08)";
  els.feedbackBox.style.borderColor = isError ? "rgba(161, 29, 51, 0.16)" : "rgba(17, 75, 95, 0.12)";
}

function clearInputs() {
  els.logId.value = "";
  els.alertId.value = "";
  els.ipAddress.value = "";
}

function renderEmptyState() {
  renderLogs([]);
  renderAlerts([]);
  renderHistory();
}

function normalizeValue(value) {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
