---
title: Security Incident Response
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Security Incident Response Environment

This project implements a deterministic but seeded OpenEnv environment where an agent acts as a SOC analyst responding to security incidents. It models realistic analyst workflows such as inspecting noisy log streams, correlating alerts, blocking malicious infrastructure, and escalating confirmed incidents. The environment is designed for reproducible benchmarking, Docker deployment, and Hugging Face Spaces hosting.

## Why this environment

Security teams repeatedly solve the same structured problems under time pressure: sort noise from signal, correlate telemetry across systems, and contain only the malicious actor without disrupting legitimate users. This environment now emphasizes that tradeoff with partially observable log windows, decoy alerts, seeded branching attack paths, and irreversible mistakes that cap the achievable score.

## Project structure

```text
.
├── Dockerfile
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── requirements.txt
├── server
│   └── app.py
├── validate-submission.sh
├── uv.lock
└── security_incident_env
    ├── __init__.py
    ├── config.py
    ├── curriculum.py
    ├── environment.py
    ├── graders.py
    ├── models.py
    ├── scenarios.py
    └── service.py
```

## OpenEnv API

The environment follows a Gymnasium-style API:

- `reset(task_name)` initializes an episode and returns an `Observation`.
- `step(action)` returns `(observation, reward, done, info)`.
- `state()` returns the full hidden `State`.

Pydantic models are used for `Action`, `Observation`, `State`, alerts, logs, and grading outputs.

## Action space

The agent can submit these actions:

- `analyze_log`
- `query_logs`
- `flag_alert`
- `inspect_user`
- `lookup_threat_intel`
- `block_ip`
- `isolate_host`
- `escalate`
- `create_incident_report`
- `ignore`

Optional parameters:

- `log_id`
- `ip_address`
- `alert_id`
- `query`
- `user_id`
- `report`

The original containment workflow remains valid, while the additional SOC tools let stronger agents search hidden telemetry, check ownership and threat intelligence, isolate a host, and write a final case report.

## Observation space

Each observation contains:

- `current_logs`: only the most recent sliding window of structured log entries, not the full historical archive
- `active_alerts`
- `blocked_ips`
- `step_count`
- `previous_action_feedback`
- `visible_history_count`
- `log_window_size`
- `context_truncated`
- `analyst_notes`
- `curriculum_level`
- `curriculum_profile`
- `weak_spot_hints`

## Internal state

The hidden state tracks:

- the full revealed log history and remaining hidden queue
- malicious log labels
- benign and decoy IPs
- branch metadata and attack path
- the current attack progression stage
- steps taken
- blocked IPs
- flagged alerts
- escalation status
- false-positive blocks
- timeliness penalties
- trajectory metrics used by the grader
- whether the incident is resolved
- curriculum profile and weak spots from recent episodes
- query, intel, isolation, and case-report artifacts

## Curriculum and adversarial pressure

The environment includes a lightweight curriculum controller inspired by incident-response training platforms:

- recent success rate and score move each task through `warmup`, `analyst`, and `principal` profiles
- higher profiles add more decoys, increase noise, and shrink the visible log window
- failed episodes record weak spots such as false-positive containment, missed alert correlation, missed escalation, and weak case reports
- observations expose compact curriculum hints without exposing hidden ground truth

Completed episodes can be written to `outputs/episode_transcripts.jsonl` for post-run analysis. Set `OPENENV_WRITE_TRANSCRIPTS=0` to disable transcript writing or `OPENENV_TRANSCRIPT_PATH` to choose a different path.

## Task levels

### Easy

Detect an obvious privileged-authentication attacker while ignoring a benign travel anomaly.

Expected strong sequence:
`analyze_log(L100) -> flag_alert(A100) -> block_ip(198.51.100.24)`

### Medium

Correlate reconnaissance with either SQL injection or credential stuffing and mitigate the attacking IP only after the pattern is clear.

Expected strong sequence:
`analyze_log(L200) -> analyze_log(L201) -> flag_alert(A200) -> block_ip(203.0.113.77)`

### Hard

Identify a multi-stage branched attack progressing from reconnaissance to exploitation to persistence. The agent must reason across misleading logs and alerts, flag the critical alert, block the attacker, and escalate the incident.

Expected strong sequence:
`analyze_log(L300) -> analyze_log(L301) -> analyze_log(L302) -> flag_alert(A301) -> block_ip(203.0.113.200) -> escalate`

## Reward design

The step reward shaping is deterministic and no longer uses naive linear normalization:

- small positive rewards for correctly analyzing relevant logs
- positive rewards for useful log queries, threat-intel lookups, ownership inspection, and accurate incident reports
- larger rewards for correct alert correlation, mitigation, and justified escalation
- penalties for incorrect or invalid actions
- `-0.5` for blocking a legitimate IP
- penalties for redundant or useless steps

Final scoring is handled by the grader using the full trajectory.

## Grading

Programmatic graders score episodes in `[0.0, 1.0]` using:

- correctness
- efficiency
- timeliness
- trajectory quality
- false positive penalties

The grader is deterministic, task-aware, and trajectory-aware. False positive blocks cap the score at `0.6`, and unresolved incidents are capped at `0.5`. See `security_incident_env/graders.py`.

Strong case reports receive a small score bonus when they name the attacker, core evidence, correlated alert, containment action, and escalation requirement. Vague or wrong reports are penalized as low-quality analyst work.

## Local setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install openenv-core
huggingface-cli login
uv run server
```

The API will be available on `http://localhost:7860`.

## Submission flow

The expected submission path is:

1. Test locally with `uv run server` and the local validation commands.
2. Deploy the environment to a Hugging Face Docker Space with `openenv push`.
3. Submit the public Space URL after the Space and validator both pass.

Recommended local checks:

```bash
python -m unittest
pytest
openenv validate
docker build .
```

Round 1 submission detail from the event docs: only the team leader can make the final submission, and the final artifact submitted on the platform is the public HF Space URL.

## API endpoints

- `POST /reset`
- `POST /step`
- `GET /health`

Example reset request:

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy"}'
```

Example step request:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<session-id>","action":{"action_type":"analyze_log","log_id":"L100"}}'
```

## Inference runner

`inference.py` uses the OpenAI Python client and reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

For Hugging Face router deployments, `API_BASE_URL` should usually be `https://router.huggingface.co/v1`.

Run with an OpenAI-compatible endpoint:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
python inference.py --task hard
```

For a local smoke test without network access:

```bash
python inference.py --task hard --policy heuristic
```

The log format is exactly:

```text
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
```

## Docker

Build and run:

```bash
docker build -t security-incident-openenv .
docker run --rm -p 7860:7860 security-incident-openenv
```

## Hugging Face Spaces

This project is compatible with Docker Spaces. Set the Space SDK to Docker, keep port `7860`, and the container will expose `/reset` and `/step` for OpenEnv clients.

Deploy with:

```bash
huggingface-cli login
uv run server
openenv push --repo-id <username>/<space-name>
```

After the Space is live, verify it and then validate the public URL:

```bash
curl -X POST https://<your-space>.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}'

./validate-submission.sh https://<your-space>.hf.space
```

Submit the public Space URL only after the local checks, `openenv push`, and the public validator all succeed.

## Baseline scores

Using the deterministic heuristic policy against the default seed:

- `easy`: `0.851`
- `medium`: `0.802`
- `hard`: `0.935`

Using random or overly aggressive blocking policies, scores drop quickly because false positives are penalized strongly and delayed containment reduces both reward and final score.
