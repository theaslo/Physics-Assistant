import unittest

from main import _build_kinematics_fallback_solution
from physics_graphs import (
    KINEMATICS_EXAMPLE_PROBLEMS,
    PROJECTILE_EXAMPLE_PROBLEMS,
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

    def test_projectile_example_problems_generate_full_graph_set(self):
        self.assertGreaterEqual(len(PROJECTILE_EXAMPLE_PROBLEMS), 8)

        for example in PROJECTILE_EXAMPLE_PROBLEMS:
            with self.subTest(example=example["name"]):
                result = build_kinematics_graph_response(
                    problem=example["problem"],
                    structured=example.get("structured"),
                )

                self.assertEqual(result["errors"], [])
                self.assertEqual(len(result["graphs"]), 1)

                payload = result["graphs"][0]
                self.assertEqual(payload["type"], "projectile_motion")
                self.assertEqual(payload["motionType"], example["expected_motion_type"])
                self.assertEqual(len(payload["graphs"]), 3)
                self.assertEqual(payload["graphs"][0]["id"], "trajectory")
                self.assertEqual(payload["graphs"][1]["id"], "height-time")
                self.assertEqual(payload["graphs"][2]["id"], "velocity-components-time")
                self.assertGreaterEqual(len(payload["graphs"][2]["series"]), 2)

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

    def test_velocity_change_wording_generates_constant_acceleration_graph(self):
        result = build_kinematics_graph_response(
            "Graph a scooter that speeds up from 5 m/s to 25 m/s in 4 seconds."
        )

        self.assertEqual(result["errors"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["type"], "kinematics_1d")
        self.assertEqual(payload["motionType"], "constant_acceleration")

    def test_braking_to_stop_wording_generates_graph(self):
        result = build_kinematics_graph_response(
            "Graph a car traveling at 20 m/s that brakes to a stop in 5 seconds."
        )

        self.assertEqual(result["errors"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["type"], "kinematics_1d")
        final_velocity = payload["parameters"][4]["value"]
        self.assertEqual(final_velocity, 0)

    def test_piecewise_motion_generates_graph(self):
        result = build_kinematics_graph_response(
            "Graph a car that accelerates from rest at 2 m/s^2 for 5 s, then coasts for 4 s, then brakes at 4 m/s^2 until stopping."
        )

        self.assertEqual(result["errors"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["type"], "kinematics_piecewise")
        self.assertEqual(payload["motionType"], "piecewise_kinematics")
        self.assertEqual(len(payload["graphs"]), 3)

    def test_piecewise_runner_rest_interval_and_fallback_solution(self):
        result = build_kinematics_graph_response(
            "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        )

        self.assertEqual(result["errors"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["motionType"], "piecewise_kinematics")
        parameters = {parameter["symbol"]: parameter["value"] for parameter in payload["parameters"]}
        self.assertEqual(parameters["T"], 8)
        self.assertEqual(parameters["x final"], 20)
        self.assertEqual(parameters["v final"], 0)

        segments = payload["source"]["segments"]
        self.assertEqual(segments[1]["initial_velocity"], 0)
        self.assertEqual(segments[1]["final_velocity"], 0)

        solution = _build_kinematics_fallback_solution(result)
        self.assertIn("Total distance = 20 m", solution)
        self.assertIn("horizontal at 4 m/s from 0 s to 5 s", solution)
        self.assertIn("horizontal at 0 m/s from 5 s to 8 s", solution)

    def test_structured_projectile_components(self):
        result = build_kinematics_graph_response(
            structured={
                "motion_type": "projectile",
                "v0x": 10,
                "v0y": 10,
                "h0": 0,
            }
        )

        self.assertEqual(result["errors"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["type"], "projectile_motion")
        self.assertAlmostEqual(
            payload["graphs"][0]["series"][0]["points"][-1]["y"],
            0,
            places=5,
        )

    def test_projectile_from_ground_without_upward_speed_needs_more_information(self):
        result = build_kinematics_graph_response(
            "Graph a projectile launched horizontally at 10 m/s from ground level."
        )

        self.assertEqual(result["graphs"], [])
        self.assertTrue(any("no positive flight time" in error for error in result["errors"]))

    def test_projectile_missing_angle_is_reported(self):
        result = build_kinematics_graph_response(
            "Graph a projectile launched at 20 m/s."
        )

        self.assertEqual(result["graphs"], [])
        self.assertTrue(any("launch angle" in error for error in result["errors"]))

    def test_soccer_ball_from_hill_uses_initial_height(self):
        result = build_kinematics_graph_response(
            "A soccer ball is kicked from a 20m hill with an initial speed of 25m/s at an angle of 40 degree. Graph the projectile's trajectory, height vs. time, horizontal velocity vs. time, and vertical velocity vs. time, then determine the time it takes to hit the ground and the horizontal distance traveled."
        )

        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"], [])
        payload = result["graphs"][0]
        self.assertEqual(payload["type"], "projectile_motion")

        parameters = {parameter["symbol"]: parameter["value"] for parameter in payload["parameters"]}
        self.assertEqual(parameters["h0"], 20)
        self.assertAlmostEqual(parameters["T"], 4.239, places=2)
        self.assertAlmostEqual(parameters["R"], 81.18, places=1)
        self.assertAlmostEqual(parameters["hmax"], 33.16, places=1)

    def test_projectile_height_phrasing_variants_are_not_ignored(self):
        cases = [
            ("Graph a ball kicked from a 20-meter hill at 25 m/s at 40 degrees.", 20),
            ("Graph a rock thrown from a bridge 30 meters high at 12 m/s at 35 degrees.", 30),
            ("Graph a ball launched from a 15 m roof with initial speed 18 m/s at 25 degrees.", 15),
            ("Graph a projectile launched 12 m above the ground at 10 m/s at 45 degrees.", 12),
        ]

        for problem, expected_height in cases:
            with self.subTest(problem=problem):
                result = build_kinematics_graph_response(problem)

                self.assertEqual(result["errors"], [])
                parameters = {parameter["symbol"]: parameter["value"] for parameter in result["graphs"][0]["parameters"]}
                self.assertAlmostEqual(parameters["h0"], expected_height, places=5)

    def test_elevated_projectile_without_height_refuses_ground_default(self):
        result = build_kinematics_graph_response(
            "Graph a ball kicked from a balcony with initial speed 18 m/s at 30 degrees."
        )

        self.assertEqual(result["graphs"], [])
        self.assertTrue(any("launch height is missing" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
