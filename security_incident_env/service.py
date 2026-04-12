"""FastAPI app exposing the environment over HTTP for Hugging Face Spaces."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_ENV_NAME
from .environment import SecurityIncidentResponseEnv
from .models import ResetRequest, ResetResponse, StepRequest, StepResponse


app = FastAPI(title="Security Incident Response OpenEnv", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, SecurityIncidentResponseEnv] = {}
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health check for container orchestration."""
    return {"status": "healthy", "env_name": DEFAULT_ENV_NAME}


@app.post("/reset", response_model=ResetResponse)
def reset_environment(request: Optional[ResetRequest] = None) -> ResetResponse:
    """Create a new environment session and return the initial observation."""
    if request is None:
        request = ResetRequest()
    env = SecurityIncidentResponseEnv()
    observation = env.reset(request.task_name)
    session_id = str(uuid4())
    _sessions[session_id] = env
    return ResetResponse(
        session_id=session_id,
        observation=observation,
        env_name=env.env_name,
        task_name=request.task_name,
        max_steps=env.state().max_steps,
    )


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest) -> StepResponse:
    """Execute a step within an existing environment session."""
    env = _sessions.get(request.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id.")
    observation, reward, done, info = env.step(request.action)
    if done:
        env.close()
        _sessions.pop(request.session_id, None)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
