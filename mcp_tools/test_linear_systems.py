import json
import unittest

from physics_mcp_tools.linear_systems import solve_linear_systems, validate_physics_equations


class MathMCPSystemTests(unittest.TestCase):
    def test_exact_batch_results_and_substitution_checks(self):
        batch = solve_linear_systems([
            {"equations": ["2v_i + v_f = 9", "v_f - v_i = 3"]},
            {"equations": ["0.5a + b = 2", "a - b = 1"]},
            {"equations": ["2x + y = 1", "x - y = 0"]},
        ])
        self.assertEqual(batch["schema_version"], 1)
        values = [{"v_i": "2", "v_f": "5"}, {"a": "2", "b": "1"}, {"x": "1/3", "y": "1/3"}]
        for item, expected in zip(batch["results"], values):
            self.assertTrue(item["ok"])
            self.assertEqual(item["calculation"]["solution"], expected)
            self.assertEqual(len(item["calculation"]["checks"]), 2)
            self.assertTrue(all(check["satisfied"] for check in item["calculation"]["checks"]))
        json.dumps(batch)  # Fractions stay exact but remain serializable over MCP.

    def test_singular_and_invalid_items_do_not_hide_valid_batch_results(self):
        batch = solve_linear_systems([
            {"equations": ["x + y = 2", "2x + 2y = 4"]},
            {"equations": ["x + y = 2", "2x + 2y = 5"]},
            {"equations": ["x*y = 2", "x-y = 1"]},
        ])
        self.assertEqual(batch["results"][0]["calculation"]["status"], "dependent")
        self.assertEqual(batch["results"][1]["calculation"]["status"], "inconsistent")
        self.assertFalse(batch["results"][2]["ok"])

    def test_generation_validation_and_history_duplicates(self):
        candidate = {"topic": "kinematics", "quantities": {"v_i": "velocity", "v_f": "velocity"},
                     "equations": [{"coefficients": {"v_i": 2, "v_f": 3}, "rhs": 21},
                                   {"coefficients": {"v_i": -1, "v_f": 1}, "rhs": 2}]}
        batch = validate_physics_equations([candidate, candidate], ["kinematics"])
        accepted, duplicate = batch["results"]
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["problem"]["calculation"]["solution"], {"v_i": "3", "v_f": "5"})
        self.assertFalse(duplicate["accepted"])
        excluded = validate_physics_equations([candidate], ["kinematics"], [accepted["fingerprint"]])
        self.assertFalse(excluded["results"][0]["accepted"])
        history = validate_physics_equations([candidate], ["kinematics"],
                                            previous_systems=[["4v_i+6v_f=42", "2v_f-2v_i=4"]])
        self.assertFalse(history["results"][0]["accepted"])
        json.dumps(batch)

    def test_common_physical_quantity_names_are_normalized(self):
        candidate = {"topic": "energy", "quantities": {"K": "kinetic energy", "U": "potential energy"},
                     "equations": ["K + 2U = 15", "3K + U = 14"]}
        result = validate_physics_equations([candidate], ["energy"])["results"][0]
        self.assertTrue(result["accepted"])
        self.assertEqual(result["problem"]["quantities"]["K"][0], "Kinetic energy")
        self.assertEqual(result["problem"]["calculation"]["solution"], {"K": "13/5", "U": "31/5"})

    def test_limits_and_quantity_metadata(self):
        with self.assertRaises(ValueError):
            solve_linear_systems([])
        with self.assertRaises(ValueError):
            validate_physics_equations([{}] * 11, ["forces"])
        result = solve_linear_systems([{"equations": ["x+y=3", "x-y=1"], "quantities": {"z": ["bad", ""]}}])
        self.assertFalse(result["results"][0]["ok"])


if __name__ == "__main__":
    unittest.main()
