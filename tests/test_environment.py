"""Deterministic regression tests for the security incident response environment."""

from __future__ import annotations

import unittest

from inference import sanitize_action
from security_incident_env.environment import SecurityIncidentResponseEnv
from security_incident_env.graders import grade_episode
from security_incident_env.judge import evaluate_judge
from security_incident_env.models import Action, TaskName
from security_incident_env.scenarios import build_scenario


class EnvironmentDeterminismTests(unittest.TestCase):
    """Scenario generation and grading must be reproducible."""

    def test_same_seed_produces_same_medium_scenario(self) -> None:
        scenario_a = build_scenario(TaskName.MEDIUM, seed=812)
        scenario_b = build_scenario(TaskName.MEDIUM, seed=812)

        self.assertEqual(scenario_a.attack_path, scenario_b.attack_path)
        self.assertEqual([log.model_dump(mode="json") for log in scenario_a.logs], [log.model_dump(mode="json") for log in scenario_b.logs])
        self.assertEqual([alert.model_dump(mode="json") for alert in scenario_a.alerts], [alert.model_dump(mode="json") for alert in scenario_b.alerts])

    def test_hard_task_exposes_multiple_branches_across_seed_range(self) -> None:
        branches = {build_scenario(TaskName.HARD, seed=seed).attack_path for seed in range(900, 908)}
        self.assertGreaterEqual(len(branches), 2)

    def test_observation_uses_sliding_window_and_budget(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        observation = env.reset(TaskName.HARD)

        self.assertLessEqual(len(observation.current_logs), 5)
        self.assertEqual(observation.visible_history_count, 5)
        self.assertEqual(observation.remaining_budget, 8)
        self.assertFalse(observation.context_truncated)

        for action in [
            Action(action_type="analyze_log", log_id="L300"),
            Action(action_type="analyze_log", log_id="L301"),
            Action(action_type="analyze_log", log_id="L302"),
        ]:
            observation, _, _, _ = env.step(action)

        self.assertLessEqual(len(observation.current_logs), 5)
        self.assertGreater(observation.visible_history_count, len(observation.current_logs))
        self.assertEqual(observation.remaining_budget, 5)
        self.assertTrue(observation.context_truncated)

    def test_budget_exhaustion_ends_episode(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.EASY)

        actions = [
            Action(action_type="analyze_log", log_id="L100"),
            Action(action_type="analyze_log", log_id="L101"),
            Action(action_type="flag_alert", alert_id="A101"),
            Action(action_type="block_ip", ip_address="198.51.100.24"),
            Action(action_type="escalate"),
            Action(action_type="analyze_log", log_id="L108"),
        ]

        done = False
        last_info = {}
        for action in actions:
            _, _, done, last_info = env.step(action)

        self.assertTrue(done)
        self.assertTrue(last_info["budget_exhausted"])
        self.assertEqual(env.state().remaining_budget, 0)
        self.assertLess(last_info["raw_reward"], 0)

    def test_premature_escalation_is_penalized(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.HARD)

        env.step(Action(action_type="analyze_log", log_id="L300"))
        _, reward, done, info = env.step(Action(action_type="escalate"))

        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.4, places=4)
        self.assertFalse(env.state().escalation_sent)
        self.assertEqual(env.state().premature_escalation_count, 1)
        self.assertIn("premature", info["feedback"])

    def test_escalation_after_mitigation_resolves_incident(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.HARD)

        actions = [
            Action(action_type="analyze_log", log_id="L300"),
            Action(action_type="analyze_log", log_id="L302"),
            Action(action_type="flag_alert", alert_id="A301"),
            Action(action_type="block_ip", ip_address="203.0.113.200"),
            Action(action_type="escalate"),
        ]

        done = False
        reward = 0.0
        info = {}
        for action in actions:
            _, reward, done, info = env.step(action)

        self.assertTrue(done)
        self.assertTrue(info["resolved"])
        self.assertTrue(env.state().escalation_sent)
        self.assertGreater(reward, 0.0)

    def test_hard_heuristic_is_no_longer_near_perfect(self) -> None:
        actions = [
            Action(action_type="analyze_log", log_id="L300"),
            Action(action_type="analyze_log", log_id="L301"),
            Action(action_type="analyze_log", log_id="L302"),
            Action(action_type="flag_alert", alert_id="A301"),
            Action(action_type="block_ip", ip_address="203.0.113.200"),
            Action(action_type="escalate"),
        ]

        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.HARD)
        for action in actions:
            env.step(action)

        grade = grade_episode(env.state())
        self.assertGreaterEqual(grade.score, 0.6)
        self.assertLessEqual(grade.score, 0.8)

    def test_judge_fallback_is_deterministic(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.HARD)
        for action in [
            Action(action_type="analyze_log", log_id="L300"),
            Action(action_type="analyze_log", log_id="L302"),
            Action(action_type="flag_alert", alert_id="A301"),
            Action(action_type="block_ip", ip_address="203.0.113.200"),
            Action(action_type="escalate"),
        ]:
            env.step(action)

        judge_a = evaluate_judge(env.state(), use_llm=False)
        judge_b = evaluate_judge(env.state(), use_llm=False)

        self.assertEqual(judge_a.model_dump(mode="json"), judge_b.model_dump(mode="json"))
        self.assertFalse(judge_a.used_llm)
        self.assertGreater(judge_a.score, 0.0)

    def test_grade_exposes_multi_layer_scores(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.HARD)
        for action in [
            Action(action_type="analyze_log", log_id="L300"),
            Action(action_type="analyze_log", log_id="L301"),
            Action(action_type="analyze_log", log_id="L302"),
            Action(action_type="flag_alert", alert_id="A301"),
            Action(action_type="block_ip", ip_address="203.0.113.200"),
            Action(action_type="escalate"),
        ]:
            env.step(action)

        grade = grade_episode(env.state(), use_llm_judge=False)
        self.assertGreaterEqual(grade.programmatic_score, 0.0)
        self.assertLessEqual(grade.programmatic_score, 1.0)
        self.assertGreaterEqual(grade.llm_judge_score, -1.0)
        self.assertLessEqual(grade.llm_judge_score, 1.0)
        self.assertGreaterEqual(grade.final_reward, -1.0)
        self.assertLessEqual(grade.final_reward, 1.0)

    def test_adaptive_difficulty_increases_after_repeated_success(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        winning_actions = [
            Action(action_type="analyze_log", log_id="L100"),
            Action(action_type="analyze_log", log_id="L101"),
            Action(action_type="flag_alert", alert_id="A100"),
            Action(action_type="block_ip", ip_address="198.51.100.24"),
        ]

        for _ in range(4):
            env.reset(TaskName.EASY)
            for action in winning_actions:
                _, _, done, _ = env.step(action)
                if done:
                    break

        adaptive_observation = env.reset(TaskName.EASY)
        self.assertLessEqual(adaptive_observation.log_window_size, 4)
        self.assertLessEqual(len(adaptive_observation.current_logs), adaptive_observation.log_window_size)

    def test_close_is_idempotent_and_requires_reset_before_reuse(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        env.reset(TaskName.EASY)
        env.close()
        env.close()

        with self.assertRaises(RuntimeError):
            env.state()

    def test_sanitize_action_strips_irrelevant_fields(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        observation = env.reset(TaskName.EASY)
        action = Action(
            action_type="block_ip",
            ip_address="198.51.100.24",
            log_id="L100",
            alert_id="A100",
        )

        sanitized = sanitize_action(action, observation)

        self.assertEqual(sanitized.action_type.value, "analyze_log")
        self.assertEqual(sanitized.log_id, "L100")
        self.assertIsNone(sanitized.alert_id)
        self.assertIsNone(sanitized.ip_address)

    def test_sanitize_action_replaces_ignore_with_visible_work(self) -> None:
        env = SecurityIncidentResponseEnv(seed=17)
        observation = env.reset(TaskName.EASY)

        sanitized = sanitize_action(Action(action_type="ignore"), observation)

        self.assertEqual(sanitized.action_type.value, "analyze_log")
        self.assertEqual(sanitized.log_id, "L100")


if __name__ == "__main__":
    unittest.main()
