"""Deployment entrypoint expected by OpenEnv validators."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security_incident_env.config import SERVICE_HOST, SERVICE_PORT
from security_incident_env.service import app


def main() -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)


if __name__ == "__main__":
    main()
