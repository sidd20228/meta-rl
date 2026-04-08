"""Application entry point for local runs and Docker containers."""

from security_incident_env.config import SERVICE_HOST, SERVICE_PORT
from security_incident_env.service import app


def main() -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)


if __name__ == "__main__":
    main()
