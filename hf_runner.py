"""Run the environment against Hugging Face-hosted OpenAI-compatible models."""

from __future__ import annotations

import argparse
import json
import os

from openai import OpenAI

from inference import format_error, is_fatal_provider_error, request_model_action
from security_incident_env.environment import SecurityIncidentResponseEnv
from security_incident_env.models import TaskName

DEFAULT_HF_ROUTER = "https://router.huggingface.co/v1"


def build_hf_client() -> tuple[OpenAI, str]:
    """Build an OpenAI-compatible client for the Hugging Face router."""
    api_base_url = os.getenv("API_BASE_URL", DEFAULT_HF_ROUTER)
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not model_name:
        raise RuntimeError("MODEL_NAME is required.")
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY is required.")
    client = OpenAI(base_url=api_base_url, api_key=api_key, timeout=max(5.0, float(os.getenv("HF_RUNNER_TIMEOUT", "20"))))
    return client, model_name


def stringify_action(action) -> str:
    """Compact action formatter for exact step logs."""
    return json.dumps(action.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)


def run_episode(env: SecurityIncidentResponseEnv, client: OpenAI, model_name: str, task_name: TaskName, episode_index: int) -> dict[str, object]:
    """Run one episode and return summary metrics."""
    observation = env.reset(task_name)
    rewards: list[str] = []
    print(f"[START] episode={episode_index} task={task_name.value} env={env.env_name} model={model_name}")

    success = False
    while True:
        step_number = observation.step_count + 1
        try:
            action = request_model_action(client, model_name, task_name, observation)
            observation, reward, done, info = env.step(action)
            rewards.append(f"{reward:.2f}")
            print(
                f"[STEP] episode={episode_index} step={step_number} action={stringify_action(action)} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )
            if done:
                success = bool(info["resolved"])
                break
        except Exception as exc:  # pragma: no cover - provider and parsing errors are external
            if is_fatal_provider_error(exc):
                print(f"[ERROR] episode={episode_index} step={step_number} fatal={json.dumps(format_error(str(exc)))}")
                break
            raise

    grade = env.grade(use_llm_judge=True)
    print(
        f"[END] episode={episode_index} success={str(success).lower()} steps={env.state().steps_taken} "
        f"score={grade.score:.2f} programmatic={grade.programmatic_score:.2f} "
        f"judge={grade.llm_judge_score:.2f} final_reward={grade.final_reward:.2f} rewards={','.join(rewards)}"
    )
    print(
        f"[JUDGE] episode={episode_index} phase={grade.judge_phase_classification.value if grade.judge_phase_classification else 'unknown'} "
        f"used_llm={str(grade.judge_used_llm).lower()} reasoning={json.dumps(grade.judge_explanation or '')}"
    )
    return {
        "success": success,
        "score": grade.score,
        "programmatic_score": grade.programmatic_score,
        "judge_score": grade.llm_judge_score,
        "final_reward": grade.final_reward,
    }


def main() -> int:
    """CLI entrypoint for repeated Hugging Face runner episodes."""
    parser = argparse.ArgumentParser(description="Run Hugging Face LLM agents against the security incident environment.")
    parser.add_argument("--task", choices=[task.value for task in TaskName], default=TaskName.HARD.value)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")

    client, model_name = build_hf_client()
    env = SecurityIncidentResponseEnv()

    episode_results = []
    for episode_index in range(1, args.episodes + 1):
        episode_results.append(run_episode(env, client, model_name, TaskName(args.task), episode_index))

    successes = sum(1 for result in episode_results if result["success"])
    avg_score = sum(float(result["score"]) for result in episode_results) / len(episode_results)
    avg_programmatic = sum(float(result["programmatic_score"]) for result in episode_results) / len(episode_results)
    avg_judge = sum(float(result["judge_score"]) for result in episode_results) / len(episode_results)
    avg_final_reward = sum(float(result["final_reward"]) for result in episode_results) / len(episode_results)
    print(
        f"[SUMMARY] task={args.task} episodes={args.episodes} successes={successes} "
        f"avg_score={avg_score:.2f} avg_programmatic={avg_programmatic:.2f} "
        f"avg_judge={avg_judge:.2f} avg_final_reward={avg_final_reward:.2f}"
    )
    return 0 if successes == args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
