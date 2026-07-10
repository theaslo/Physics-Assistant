import unittest

from fastapi.testclient import TestClient

from main import app


class GuidedTutoringApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_piecewise_runner_guided_workflow_then_full_solution(self):
        problem = (
            "A runner moves at 4 m/s for 5 s and then stops for 3 s. "
            "Find the total distance traveled and draw the velocity-time graph."
        )
        conversation = []

        def send(message, expected):
            user_message = {"role": "user", "content": message}
            context = {
                "active_problem": problem,
                "recent_conversation": [*conversation, user_message][-20:],
                "total_messages": len(conversation) + 1,
            }
            response = self.client.post(
                "/agent/kinematics_agent/solve",
                json={
                    "problem": message,
                    "user_id": "integration_test",
                    "context": context,
                },
            )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertTrue(body["success"], body)
            self.assertIn(expected, body["solution"].lower())

            conversation.append(user_message)
            conversation.append({"role": "assistant", "content": body["solution"]})
            return body

        send(problem, "piecewise motion graph")
        send("yes confirm", "area under a velocity-time graph")
        send("distance traveled", "area of the first rectangle")
        send("20 m", "distance during interval 2")
        send("0 m", "add the two distances")
        send("20 m", "final graph checkpoint")
        send(
            "The velocity-time graph is horizontal at 4 m/s from 0-5 s and horizontal at 0 m/s from 5-8 s.",
            "completed the guided setup",
        )

        final_user_message = {"role": "user", "content": "show me the full solution"}
        response = self.client.post(
            "/agent/kinematics_agent/solve",
            json={
                "problem": "show me the full solution",
                "user_id": "integration_test",
                "context": {
                    "active_problem": problem,
                    "recent_conversation": [*conversation[-4:], final_user_message],
                    "total_messages": len(conversation) + 1,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"], body)
        self.assertIsNone(body["error"])
        self.assertIn("Total distance = 20 m", body["solution"])
        self.assertIn("horizontal at 4 m/s from 0 s to 5 s", body["solution"])
        self.assertIn("horizontal at 0 m/s from 5 s to 8 s", body["solution"])
        self.assertEqual(body["tools_used"], ["kinematics_graph_builder"])

    def test_full_solution_recovers_piecewise_problem_from_guided_interval_history(self):
        response = self.client.post(
            "/agent/kinematics_agent/solve",
            json={
                "problem": "show me the full solution",
                "user_id": "integration_test",
                "context": {
                    "recent_conversation": [
                        {
                            "role": "assistant",
                            "content": (
                                "**Tutor checkpoint**\n\n"
                                "Correct. The velocity-time graph is made of these horizontal segments:\n"
                                "- Interval 1: 0-5 s, v = 4 m/s\n"
                                "- Interval 2: 5-8 s, v = 0 m/s\n\n"
                                "You have completed the guided setup."
                            ),
                        },
                        {"role": "user", "content": "show me the full solution"},
                    ],
                    "total_messages": 24,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"], body)
        self.assertIsNone(body["error"])
        self.assertIn("Total distance = 20 m", body["solution"])
        self.assertEqual(body["tools_used"], ["kinematics_graph_builder"])

    def test_api_prefixed_solve_route_supports_frontend_base_url(self):
        response = self.client.post(
            "/api/agent/kinematics_agent/solve",
            json={
                "problem": "show me the full solution",
                "user_id": "integration_test",
                "context": {
                    "recent_conversation": [
                        {
                            "role": "assistant",
                            "content": (
                                "**Tutor checkpoint**\n\n"
                                "Correct. The velocity-time graph is made of these horizontal segments:\n"
                                "- Interval 1: 0-5 s, v = 4 m/s\n"
                                "- Interval 2: 5-8 s, v = 0 m/s\n\n"
                                "You have completed the guided setup."
                            ),
                        },
                        {"role": "user", "content": "show me the full solution"},
                    ],
                    "total_messages": 24,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"], body)
        self.assertIsNone(body["error"])
        self.assertIn("Total distance = 20 m", body["solution"])
        self.assertEqual(body["tools_used"], ["kinematics_graph_builder"])


if __name__ == "__main__":
    unittest.main()
