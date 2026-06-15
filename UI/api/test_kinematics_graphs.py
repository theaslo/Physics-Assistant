import unittest

from physics_graphs import (
    KINEMATICS_EXAMPLE_PROBLEMS,
    build_kinematics_graph_response,
)


class KinematicsGraphTests(unittest.TestCase):
    def test_eight_example_problems_generate_graphs(self):
        self.assertGreaterEqual(len(KINEMATICS_EXAMPLE_PROBLEMS), 8)

        for example in KINEMATICS_EXAMPLE_PROBLEMS:
            with self.subTest(example=example["name"]):
                result = build_kinematics_graph_response(
                    problem=example["problem"],
                    structured=example.get("structured"),
                )

                self.assertEqual(result["errors"], [])
                self.assertEqual(len(result["graphs"]), 1)

                payload = result["graphs"][0]
                self.assertEqual(payload["type"], "kinematics_1d")
                self.assertEqual(payload["motionType"], example["expected_motion_type"])
                self.assertEqual(len(payload["graphs"]), 3)

                for graph in payload["graphs"]:
                    self.assertIn("xAxis", graph)
                    self.assertIn("yAxis", graph)
                    self.assertGreaterEqual(len(graph["series"][0]["points"]), 10)

    def test_missing_units_are_assumed_with_warning(self):
        result = build_kinematics_graph_response(
            "Graph a car that accelerates from rest at 3 for 5 seconds."
        )

        self.assertEqual(result["errors"], [])
        self.assertTrue(any("acceleration" in warning for warning in result["warnings"]))

    def test_negative_time_is_rejected_gracefully(self):
        result = build_kinematics_graph_response(
            structured={"v0": 0, "a": 2, "t": -5}
        )

        self.assertEqual(result["graphs"], [])
        self.assertTrue(any("Time must be greater than 0" in error for error in result["errors"]))

    def test_structured_time_range_is_respected(self):
        result = build_kinematics_graph_response(
            structured={
                "initial_position": 0,
                "initial_velocity": 1,
                "acceleration": 1,
                "duration": 4,
                "time_range": {"start": 0, "end": 4, "step": 1},
            }
        )

        points = result["graphs"][0]["graphs"][0]["series"][0]["points"]
        self.assertEqual([point["x"] for point in points], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
