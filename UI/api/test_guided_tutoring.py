import unittest

from guided_tutoring import (
    GUIDED_RESPONSE_MARKER,
    REPRESENTATIVE_TUTORING_PROBLEMS,
    active_problem_from_context,
    apply_full_solution_instruction,
    classify_physics_topic,
    evaluate_guided_tutoring,
    merge_tutoring_context,
)


class GuidedTutoringWorkflowTests(unittest.TestCase):
    def test_initial_problem_solving_request_is_guided(self):
        decision = evaluate_guided_tutoring(
            "forces_agent",
            "A 5 kg block is pulled with a 20 N force. Find its acceleration.",
        )

        self.assertTrue(decision.intercept)
        self.assertFalse(decision.allow_full_solution)
        self.assertEqual(decision.stage, "initial_diagnostic")
        self.assertIn(GUIDED_RESPONSE_MARKER, decision.response)
        self.assertIn("free-body", decision.response.lower())
        self.assertNotIn("Final answer:", decision.response)

    def test_diagnostic_questions_cover_ten_representative_problems(self):
        self.assertGreaterEqual(len(REPRESENTATIVE_TUTORING_PROBLEMS), 10)
        expected_templates = {
            "constant_velocity",
            "constant_acceleration",
            "projectile_motion",
            "newtons_second_law",
            "friction",
            "work_energy",
            "momentum_collision",
            "torque_equilibrium",
            "electric_field_force",
            "simple_circuit",
        }
        self.assertEqual(
            {case["template"] for case in REPRESENTATIVE_TUTORING_PROBLEMS[:10]},
            expected_templates,
        )

        for case in REPRESENTATIVE_TUTORING_PROBLEMS[:10]:
            with self.subTest(agent_id=case["agent_id"]):
                decision = evaluate_guided_tutoring(case["agent_id"], case["problem"])

                self.assertTrue(decision.intercept)
                self.assertIn(GUIDED_RESPONSE_MARKER, decision.response)
                self.assertIn("knowns/unknowns", decision.response.lower())
                for term in case["expected_terms"]:
                    self.assertIn(term.lower(), decision.response.lower())

    def test_topic_classifier_identifies_required_topics(self):
        cases = [
            ("A cyclist moves at constant velocity for 4 s.", "constant_velocity"),
            ("A runner moves at 4 m/s for 5 s and then stops for 3 s.", "piecewise_motion"),
            ("A car accelerates from rest at 2 m/s^2.", "constant_acceleration"),
            ("A projectile is launched at an angle.", "projectile_motion"),
            ("A ball is thrown upward with an initial velocity of 20 m/s. How high does it rise?", "vertical_motion"),
            ("A stone is dropped from rest from a 45 m bridge. How long does it take to hit the ground?", "free_fall"),
            ("A ball is launched horizontally at 12 m/s from a 20 m cliff. How far from the base does it land?", "horizontal_launch"),
            ("Use Newton's second law for a box.", "newtons_second_law"),
            ("A crate slides with kinetic friction.", "friction"),
            ("Use work-energy to find the speed.", "work_energy"),
            ("Two carts collide and stick together.", "momentum_collision"),
            ("A beam is in torque equilibrium.", "torque_equilibrium"),
            ("Find the electric field from a point charge.", "electric_field_force"),
            ("A simple circuit has a battery and resistor.", "simple_circuit"),
        ]

        for prompt, expected_topic in cases:
            with self.subTest(prompt=prompt):
                classification = classify_physics_topic(prompt)
                self.assertEqual(classification["topic"], expected_topic)

    def test_vertical_throw_upward_uses_vertical_1d_checkpoint(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A ball is thrown upward with an initial velocity of 20 m/s. How high does it rise?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "vertical_motion")
        self.assertIn("one-dimensional vertical motion", response)
        self.assertIn("initial velocity: 20 m/s upward", response)
        self.assertIn("final velocity at the highest point: 0 m/s", response)
        self.assertIn("acceleration: -9.8 m/s^2", response)
        self.assertIn("which quantity is the unknown", response)
        self.assertIn("v, v0, a, and displacement", response)
        self.assertNotIn("cos", response)
        self.assertNotIn("sin", response)
        self.assertNotIn("v0x", response)
        self.assertNotIn("horizontal", response)

    def test_vertical_throw_downward_stays_vertical_1d(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A ball is thrown downward with an initial velocity of 8 m/s from a 20 m balcony. How fast is it moving before it hits the ground?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "vertical_motion")
        self.assertIn("one-dimensional vertical motion", response)
        self.assertIn("initial velocity: 8 m/s downward", response)
        self.assertIn("20 m", response)
        self.assertIn("acceleration: -9.8 m/s^2", response)
        self.assertNotIn("cos", response)
        self.assertNotIn("sin", response)
        self.assertNotIn("v0x", response)

    def test_free_fall_from_rest_has_free_fall_checkpoint(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A stone is dropped from rest from a 45 m bridge. How long does it take to hit the ground?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "free_fall")
        self.assertIn("free-fall problem", response)
        self.assertIn("initial velocity: 0 m/s", response)
        self.assertIn("45 m", response)
        self.assertIn("acceleration: -9.8 m/s^2", response)
        self.assertNotIn("cos", response)
        self.assertNotIn("sin", response)

    def test_projectile_launched_at_angle_still_uses_components(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A ball is launched at 18 m/s at 35 degrees. Find its flight time and range.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "projectile_motion")
        self.assertIn("projectile motion", response)
        self.assertIn("18 m/s", response)
        self.assertIn("35 degrees", response)
        self.assertIn("v0x = v0 cos(theta)", response)
        self.assertIn("v0y = v0 sin(theta)", response)

    def test_horizontal_launch_from_cliff_has_horizontal_launch_checkpoint(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A ball is launched horizontally at 12 m/s from a 20 m cliff. How far from the base does it land?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "horizontal_launch")
        self.assertIn("horizontal-launch problem", response)
        self.assertIn("horizontal velocity: v0x = 12 m/s", response)
        self.assertIn("initial vertical velocity: v0y = 0 m/s", response)
        self.assertIn("initial height: 20 m", response)
        self.assertIn("vertical acceleration: -9.8 m/s^2", response)
        self.assertNotIn("cos", response)
        self.assertNotIn("sin", response)

    def test_full_solution_is_deferred_before_guidance(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Show me the full solution for a car accelerating from rest at 3 m/s^2 for 5 s.",
        )

        self.assertTrue(decision.intercept)
        self.assertFalse(decision.allow_full_solution)
        self.assertEqual(decision.stage, "full_solution_deferred")
        self.assertIn("worked example", decision.response.lower())

    def test_worked_example_request_is_allowed_immediately(self):
        decision = evaluate_guided_tutoring(
            "energy_agent",
            "I need a worked example for conservation of energy on a ramp.",
        )

        self.assertFalse(decision.intercept)
        self.assertTrue(decision.allow_full_solution)
        self.assertEqual(decision.stage, "worked_example_requested")

    def test_full_solution_allowed_after_guided_exchange(self):
        context = {
            "recent_conversation": [
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nList knowns and unknowns first.",
                },
                {
                    "role": "user",
                    "content": "Knowns: v0 = 0 m/s, a = 3 m/s^2, t = 5 s. I would use x = v0 t + 1/2 a t^2.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "show me the full solution",
            context,
        )

        self.assertFalse(decision.intercept)
        self.assertTrue(decision.allow_full_solution)
        self.assertEqual(decision.stage, "full_solution_after_guidance")

        merged = merge_tutoring_context(context, decision)
        prompted = apply_full_solution_instruction("show me the full solution", merged)
        self.assertIn("GUIDED_TUTORING_FULL_SOLUTION_APPROVED", prompted)

    def test_misconception_gets_targeted_correction(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "I think the heavier ball should fall faster than the lighter one.",
        )

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.stage, "misconception")
        self.assertIn("mass does not change", decision.response.lower())
        self.assertIn("air resistance", decision.response.lower())
        self.assertIn("what acceleration", decision.response.lower())

    def test_conceptual_question_starts_with_thinking_checkpoint(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Why is acceleration not zero at the top of projectile motion?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.stage, "conceptual_checkpoint")
        self.assertIn("current thinking", response)
        self.assertIn("top of the motion", response)
        self.assertNotIn("final answer", response)

    def test_conceptual_attempt_gets_targeted_hint(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "Why is acceleration not zero at the top of projectile motion?",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nWhat is your current thinking?",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "I think vertical velocity is zero there.",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.stage, "conceptual_guidance")
        self.assertIn("gravity still acts downward", response)
        self.assertIn("one short sentence", response)

    def test_student_attempt_gets_next_step_hint(self):
        context = {
            "recent_conversation": [
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nWhat are the knowns and unknowns?",
                }
            ],
            "total_messages": 1,
        }

        decision = evaluate_guided_tutoring(
            "momentum_agent",
            "Knowns: m1 = 2 kg, v1 = 4 m/s, m2 = 3 kg, v2 = 0 m/s. I need final speed.",
            context,
        )

        self.assertTrue(decision.intercept)
        self.assertIn("equation", decision.response.lower())
        self.assertIn("momentum", decision.response.lower())

    def test_kinematics_initial_checkpoint_has_concrete_base(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A car accelerates from rest at 3 m/s^2 for 5 s. How far does it travel?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("1d constant-acceleration", response)
        self.assertIn("v0 = 0 m/s", response)
        self.assertIn("a = +3 m/s^2", response)
        self.assertIn("t = 5 s", response)
        self.assertIn("unknown: displacement", response)
        self.assertIn("without needing final velocity", response)
        self.assertNotIn("37.5", response)

    def test_kinematics_equation_attempt_gets_substitution_scaffold(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A car accelerates from rest at 3 m/s^2 for 5 s. How far does it travel?",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nChoose the equation.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Knowns: v0 = 0 m/s, a = 3 m/s^2, t = 5 s. I should use dx = v0 t + 1/2 a t^2.",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("equation: delta x = v0*t + (1/2)*a*t^2", response)
        self.assertIn("(0 m/s)(5 s)", response)
        self.assertIn("(3 m/s^2)(5 s)^2", response)
        self.assertIn("unit check", response)
        self.assertNotIn("37.5", response)

    def test_kinematics_compact_meter_result_is_confirmed(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A car accelerates from rest at 3 m/s^2 for 5 s. How far does it travel?",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nChoose the equation.",
                },
                {
                    "role": "user",
                    "content": "dx = v0 t + 1/2 a t^2",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nSubstitute and check units.",
                },
            ],
            "total_messages": 4,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "result = 37.5m",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("yes", response)
        self.assertIn("37.5 m", response)
        self.assertIn("unit check", response)
        self.assertIn("sign check", response)

    def test_random_forces_prompt_gets_profiled_not_canned_setup(self):
        decision = evaluate_guided_tutoring(
            "forces_agent",
            "A 12 kg crate is pushed with 80 N across rough tile where mu_k is 0.35. Find the acceleration.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("12 kg", response)
        self.assertIn("80 n", response)
        self.assertIn("0.35", response)
        self.assertIn("newton", response)
        self.assertIn("sum f_x = m*a_x", response)
        self.assertIn("f_k = mu_k*n", response)
        self.assertNotIn("final answer", response)

    def test_random_projectile_graph_prompt_gets_graph_scaffold(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Graph a ball kicked from a 20 m hill at 25 m/s at 40 degrees.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("likely graph type: projectile motion", response)
        self.assertIn("launch speed", response)
        self.assertIn("25 m/s", response)
        self.assertIn("launch angle", response)
        self.assertIn("40 deg", response)
        self.assertIn("initial height", response)
        self.assertIn("20 m", response)
        self.assertIn("horizontal axis x", response)
        self.assertIn("y(t) = h0 + v0y*t - (1/2)g*t^2", response)
        self.assertNotIn("range", response)
        self.assertNotIn("flight time", response)

    def test_piecewise_runner_graph_uses_interval_checkpoint(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertNotIn("good", response)
        self.assertIn("piecewise motion", response)
        self.assertIn("interval 1: 0-5 s, v = 4 m/s", response)
        self.assertIn("interval 2: 5-8 s, v = 0 m/s", response)
        self.assertIn("horizontal line", response)
        self.assertIn("area under the velocity-time graph", response)
        self.assertIn("confirm these 2 intervals first", response)
        self.assertIn("then we will calculate the distance", response)
        self.assertNotIn("x(t) = x0 + v0*t + (1/2)*a*t^2", response)
        self.assertNotIn("v(t) = v0 + a*t", response)
        self.assertNotIn("a(t) = a", response)

    def test_piecewise_runner_non_graph_uses_area_not_constant_acceleration(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("piecewise motion", response)
        self.assertIn("interval 1: 0-5 s, v = 4 m/s", response)
        self.assertIn("interval 2: 5-8 s, v = 0 m/s", response)
        self.assertIn("area under the velocity-time graph", response)
        self.assertNotIn("1d constant-acceleration", response)
        self.assertNotIn("delta x = v0*t + (1/2)*a*t^2", response)

    def test_piecewise_runner_graph_advances_after_area_meaning_answer(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nConfirm these 2 intervals first.",
                },
                {
                    "role": "user",
                    "content": "Interval 1: 0-5 s, v = 4 m/s. Interval 2: 5-8 s, v = 0 m/s.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNext checkpoint: what does the area under each horizontal segment represent?",
                },
            ],
            "total_messages": 4,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "distance traveled",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("now calculate the area of the first rectangle", response)
        self.assertIn("base = 5 s", response)
        self.assertIn("height = 4 m/s", response)
        self.assertNotIn("what does the area under each horizontal segment represent", response)

    def test_piecewise_runner_short_confirmation_advances_to_area_meaning(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        confirmations = ["yes", "confirm", "correct", "yes confirm"]

        for confirmation in confirmations:
            with self.subTest(confirmation=confirmation):
                context = {
                    "recent_conversation": [
                        {"role": "user", "content": problem},
                        {
                            "role": "assistant",
                            "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph.\n\nConfirm these 2 intervals first. Then we will calculate the distance.",
                        },
                    ],
                    "total_messages": 2,
                }

                decision = evaluate_guided_tutoring(
                    "kinematics_agent",
                    confirmation,
                    context,
                )

                response = decision.response.lower()

                self.assertTrue(decision.intercept)
                self.assertIn("intervals are confirmed", response)
                self.assertIn("what does the area under a velocity-time graph represent", response)
                self.assertNotIn("choose a starting kinematics equation", response)
                self.assertNotIn("motion diagram", response)
                self.assertNotIn("first rectangle", response)

    def test_piecewise_runner_graph_progresses_through_rectangles_without_repeating(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."

        confirm_context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nConfirm these 2 intervals first.",
                },
            ],
            "total_messages": 2,
        }
        interval_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Interval 1: 0-5 s, v = 4 m/s. Interval 2: 5-8 s, v = 0 m/s.",
            confirm_context,
        )
        interval_response = interval_decision.response.lower()
        self.assertIn("what does the area under a velocity-time graph represent", interval_response)

        area_context = {
            "recent_conversation": [
                *confirm_context["recent_conversation"],
                {
                    "role": "user",
                    "content": "Interval 1: 0-5 s, v = 4 m/s. Interval 2: 5-8 s, v = 0 m/s.",
                },
                {"role": "assistant", "content": interval_decision.response},
            ],
            "total_messages": 4,
        }
        area_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "The area represents distance traveled.",
            area_context,
        )
        area_response = area_decision.response.lower()
        self.assertIn("area of the first rectangle", area_response)
        self.assertNotIn("what does the area under each horizontal segment represent", area_response)

        first_rectangle_context = {
            "recent_conversation": [
                *area_context["recent_conversation"],
                {"role": "user", "content": "The area represents distance traveled."},
                {"role": "assistant", "content": area_decision.response},
            ],
            "total_messages": 6,
        }
        first_rectangle_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "20 m",
            first_rectangle_context,
        )
        first_rectangle_response = first_rectangle_decision.response.lower()
        self.assertIn("interval 1 gives 20 m", first_rectangle_response)
        self.assertIn("area of the second rectangle", first_rectangle_response)

        second_rectangle_context = {
            "recent_conversation": [
                *first_rectangle_context["recent_conversation"],
                {"role": "user", "content": "20 m"},
                {"role": "assistant", "content": first_rectangle_decision.response},
            ],
            "total_messages": 8,
        }
        second_rectangle_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "0 m",
            second_rectangle_context,
        )
        second_rectangle_response = second_rectangle_decision.response.lower()
        self.assertIn("interval 2 gives 0 m", second_rectangle_response)
        self.assertIn("add the two distances", second_rectangle_response)

        total_context = {
            "recent_conversation": [
                *second_rectangle_context["recent_conversation"],
                {"role": "user", "content": "0 m"},
                {"role": "assistant", "content": second_rectangle_decision.response},
            ],
            "total_messages": 10,
        }
        total_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "20 m total",
            total_context,
        )
        total_response = total_decision.response.lower()
        self.assertIn("total distance is 20 m", total_response)
        self.assertIn("final graph checkpoint", total_response)
        self.assertIn("two horizontal segments", total_response)
        self.assertNotIn("what does the area under each horizontal segment represent", total_response)

        graph_context = {
            "recent_conversation": [
                *total_context["recent_conversation"],
                {"role": "user", "content": "20 m total"},
                {"role": "assistant", "content": total_decision.response},
            ],
            "total_messages": 12,
        }
        graph_decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Horizontal segment from 0-5 s at v = 4 m/s, then horizontal segment from 5-8 s at v = 0 m/s.",
            graph_context,
        )
        graph_response = graph_decision.response.lower()
        self.assertIn("completed the guided setup", graph_response)
        self.assertIn("show me the full solution", graph_response)
        self.assertNotIn("confirm", graph_response)

    def test_piecewise_runner_short_confirmation_full_flow_never_uses_generic_equation_prompt(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        conversation = [
            {
                "role": "user",
                "content": problem,
            },
            {
                "role": "assistant",
                "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph.\n\nConfirm these 2 intervals first. Then we will calculate the distance.",
            },
        ]
        student_turns = [
            ("yes confirm", "area under a velocity-time graph"),
            ("distance traveled", "area of the first rectangle"),
            ("20 m", "distance during interval 2"),
            ("0 m", "add the two distances"),
            ("20 m", "final graph checkpoint"),
            (
                "Horizontal segment from 0-5 s at v = 4 m/s, then horizontal segment from 5-8 s at v = 0 m/s.",
                "completed the guided setup",
            ),
        ]

        for message, expected in student_turns:
            decision = evaluate_guided_tutoring(
                "kinematics_agent",
                message,
                {
                    "recent_conversation": conversation,
                    "total_messages": len(conversation),
                },
            )

            response = decision.response.lower()

            self.assertTrue(decision.intercept)
            self.assertIn(expected, response)
            self.assertNotIn("choose a starting kinematics equation", response)
            self.assertNotIn("send that equation", response)
            self.assertNotIn("delta x = v0*t", response)
            self.assertNotIn("keep this piecewise", response)

            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": decision.response})

        self.assertGreaterEqual(len(conversation), 14)

    def test_piecewise_runner_graph_description_does_not_become_new_generic_graph_problem(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph. Confirm these 2 intervals first.",
                },
                {"role": "user", "content": "yes confirm"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nWhat does the area under a velocity-time graph represent?",
                },
                {"role": "user", "content": "distance"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nFor Interval 1, what is the rectangular area (4 m/s)(5 s)?",
                },
                {"role": "user", "content": "20 m"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nFor Interval 2, the velocity is 0 m/s for 3 s. What distance is traveled?",
                },
                {"role": "user", "content": "0 m"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNow add the two distances. What is the total distance?",
                },
                {"role": "user", "content": "20 m"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nFinal graph checkpoint: describe the velocity-time graph in words before we call it complete.",
                },
            ],
            "total_messages": 12,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "The velocity-time graph is horizontal at 4 m/s from 0-5 s and horizontal at 0 m/s from 5-8 s.",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "piecewise_motion")
        self.assertIn("completed the guided setup", response)
        self.assertNotIn("for a 1d motion graph, define the functions", response)
        self.assertNotIn("x(t) = x0 + v0*t + (1/2)*a*t^2", response)
        self.assertNotIn("v(t) = v0 + a*t", response)
        self.assertNotIn("choose a starting kinematics equation", response)

    def test_piecewise_runner_does_not_change_branches_across_eight_turns(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        conversation = [
            {"role": "user", "content": problem},
            {
                "role": "assistant",
                "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph.\n\nConfirm these 2 intervals first. Then we will calculate the distance.",
            },
        ]
        expected_turns = [
            ("yes confirm", "area under a velocity-time graph"),
            ("distance", "area of the first rectangle"),
            ("20 m", "distance during interval 2"),
            ("0 m", "add the two distances"),
            ("20 m total", "final graph checkpoint"),
            (
                "The velocity-time graph is horizontal at 4 m/s from 0-5 s and horizontal at 0 m/s from 5-8 s.",
                "completed the guided setup",
            ),
            ("show me the full solution", None),
        ]

        generic_forbidden = (
            "for a 1d motion graph, define the functions",
            "x(t) = x0 + v0*t + (1/2)*a*t^2",
            "v(t) = v0 + a*t",
            "a(t) = a",
            "choose a starting kinematics equation",
            "send that equation",
        )

        for message, expected in expected_turns:
            decision = evaluate_guided_tutoring(
                "kinematics_agent",
                message,
                {
                    "recent_conversation": conversation,
                    "total_messages": len(conversation),
                },
            )

            if expected is None:
                self.assertFalse(decision.intercept)
                self.assertTrue(decision.allow_full_solution)
                break

            response = decision.response.lower()
            self.assertTrue(decision.intercept)
            self.assertEqual(decision.metadata["topic_classification"]["topic"], "piecewise_motion")
            self.assertIn(expected, response)
            for forbidden in generic_forbidden:
                self.assertNotIn(forbidden, response)

            conversation.append({"role": "user", "content": message})
            conversation.append({"role": "assistant", "content": decision.response})

    def test_piecewise_runner_interval_one_area_answer_does_not_reset_from_variant_prompt(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph. Confirm these 2 intervals first.",
                },
                {
                    "role": "user",
                    "content": "Interval 1: 0-5 s, v = 4 m/s. Interval 2: 5-8 s, v = 0 m/s.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNext checkpoint: what does the area under each horizontal segment represent?",
                },
                {
                    "role": "user",
                    "content": "distance traveled",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nUse area = velocity × time for Interval 1.\n\nSubstitute: (4 m/s)(5 s). What distance does that give in meters?",
                },
            ],
            "total_messages": 6,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "20 m",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("interval 1 gives 20 m", response)
        self.assertIn("distance during interval 2", response)
        self.assertNotIn("keep this piecewise", response)
        self.assertNotIn("confirm the intervals", response)
        self.assertNotIn("confirm these", response)

    def test_piecewise_runner_state_survives_long_frontend_context_window(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        conversation = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nConfirm these 2 intervals first."},
            {"role": "user", "content": "Interval 1: 0-5 s, v = 4 m/s. Interval 2: 5-8 s, v = 0 m/s."},
            {"role": "assistant", "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNext checkpoint: what does the area under each horizontal segment represent?"},
            {"role": "user", "content": "distance traveled"},
            {"role": "assistant", "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nUse area = velocity × time for Interval 1.\n\nSubstitute: (4 m/s)(5 s). What distance does that give in meters?"},
            {"role": "user", "content": "20 m"},
            {"role": "assistant", "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nCorrect. Interval 1 gives 20 m.\n\nNow calculate the area of the second rectangle:\n- base = 3 s\n- height = 0 m/s\n\nWhat is the distance during Interval 2?"},
            {"role": "user", "content": "0 m"},
            {"role": "assistant", "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nCorrect. Interval 2 gives 0 m.\n\nNow add the two distances:\n- Interval 1: 20 m\n- Interval 2: 0 m\n\nWhat is the total distance traveled?"},
        ]

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "20 m",
            {
                "recent_conversation": conversation,
                "total_messages": len(conversation),
            },
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("total distance is 20 m", response)
        self.assertIn("final graph checkpoint", response)
        self.assertNotIn("keep this piecewise", response)
        self.assertNotIn("confirm these", response)

    def test_piecewise_runner_never_uses_constant_acceleration_graph_template_after_lock(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": (
                        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                        "This is a piecewise motion graph.\n\n"
                        "Piecewise intervals I can read:\n"
                        "- Interval 1: 0-5 s, v = 4 m/s\n"
                        "- Interval 2: 5-8 s, v = 0 m/s\n\n"
                        "Confirm these 2 intervals first."
                    ),
                },
                {"role": "user", "content": "yes confirm"},
                {
                    "role": "assistant",
                    "content": (
                        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                        "Good. For a 1D motion graph, define the functions before drawing points.\n\n"
                        "Use:\n"
                        "- x(t) = x0 + v0*t + (1/2)*a*t^2\n"
                        "- v(t) = v0 + a*t\n"
                        "- a(t) = a"
                    ),
                },
            ],
            "total_messages": 4,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "The graph is a velocity-time graph from 0 to 8 s.",
            context,
        )

        response = decision.response.lower()
        self.assertTrue(decision.intercept)
        self.assertEqual(decision.metadata["topic_classification"]["topic"], "piecewise_motion")
        self.assertIn("piecewise", response)
        self.assertIn("horizontal", response)
        self.assertIn("area", response)
        self.assertNotIn("for a 1d motion graph, define the functions", response)
        self.assertNotIn("x(t) = x0 + v0*t + (1/2)*a*t^2", response)
        self.assertNotIn("v(t) = v0 + a*t", response)
        self.assertNotIn("a(t) = a", response)

    def test_piecewise_runner_truncated_history_stays_piecewise_from_interval_lines(self):
        context = {
            "recent_conversation": [
                {
                    "role": "assistant",
                    "content": (
                        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                        "This is a piecewise motion graph.\n\n"
                        "- Interval 1: 0-5 s, v = 4 m/s\n"
                        "- Interval 2: 5-8 s, v = 0 m/s\n\n"
                        "Confirm these 2 intervals first."
                    ),
                },
                {"role": "user", "content": "yes confirm"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nWhat does the area under a velocity-time graph represent?",
                },
                {"role": "user", "content": "distance traveled"},
                {
                    "role": "assistant",
                    "content": (
                        f"**{GUIDED_RESPONSE_MARKER}**\n\n"
                        "Now calculate the area of the first rectangle:\n"
                        "- base = 5 s\n"
                        "- height = 4 m/s\n\n"
                        "What distance does Interval 1 give?"
                    ),
                },
            ],
            "total_messages": 5,
        }

        decision = evaluate_guided_tutoring("kinematics_agent", "20 m", context)

        response = decision.response.lower()
        self.assertTrue(decision.intercept)
        self.assertIn("interval 1 gives 20 m", response)
        self.assertIn("distance during interval 2", response)
        self.assertNotIn("for a 1d motion graph, define the functions", response)
        self.assertNotIn("x(t) = x0 + v0*t + (1/2)*a*t^2", response)
        self.assertNotIn("v(t) = v0 + a*t", response)

    def test_full_solution_request_recovers_active_piecewise_problem(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "recent_conversation": [
                {"role": "user", "content": problem},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a piecewise motion graph. Confirm these 2 intervals first.",
                },
                {"role": "user", "content": "yes confirm"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nWhat does the area under a velocity-time graph represent?",
                },
                {"role": "user", "content": "distance traveled"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNow calculate the area of the first rectangle.",
                },
                {"role": "user", "content": "20 m"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNow calculate Interval 2.",
                },
                {"role": "user", "content": "0 m"},
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nNow add the two distances.",
                },
            ],
            "total_messages": 10,
        }

        active_problem = active_problem_from_context(
            "kinematics_agent",
            "show me the full solution",
            context,
        )

        self.assertEqual(active_problem, problem)

    def test_full_solution_request_uses_explicit_active_problem_when_recent_window_lost_original(self):
        problem = "A runner moves at 4 m/s for 5 s and then stops for 3 s. Find the total distance traveled and draw the velocity-time graph."
        context = {
            "active_problem": problem,
            "recent_conversation": [
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nYou have completed the guided setup.",
                },
                {"role": "user", "content": "show me the full solution"},
            ],
            "total_messages": 24,
        }

        active_problem = active_problem_from_context(
            "kinematics_agent",
            "show me the full solution",
            context,
        )

        self.assertEqual(active_problem, problem)

    def test_constant_acceleration_graph_first_checkpoint_is_knowns_only(self):
        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "A car starts from rest and accelerates at 2 m/s² for 6 s. Find its final velocity and draw the velocity-time graph.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("constant-acceleration problem", response)
        self.assertIn("v0 = 0 m/s because the car starts from rest", response)
        self.assertIn("a = +2 m/s^2", response)
        self.assertIn("t = 6 s", response)
        self.assertIn("positive, negative, or zero", response)
        self.assertIn("what should go on each axis", response)
        self.assertNotIn("x(t) = x0 + v0*t + (1/2)*a*t^2", response)
        self.assertNotIn("v(t) = v0 + a*t", response)
        self.assertNotIn("a(t) = a", response)

    def test_constant_acceleration_graph_reveals_equation_after_setup_confirmation(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A car starts from rest and accelerates at 2 m/s² for 6 s. Find its final velocity and draw the velocity-time graph.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nThis is a constant-acceleration problem.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "Acceleration is positive. Horizontal axis is time in seconds, vertical axis is velocity in m/s.",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("v(t) = v0 + a*t", response)
        self.assertIn("starts at (0 s, 0 m/s)", response)
        self.assertIn("slope +2 m/s^2", response)
        self.assertIn("compute v(6)", response)

    def test_kinematics_graph_followup_advances_after_interval_and_sign(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A car starts from rest and accelerates at 2m/s 2 for 6 s. Find its final velocity and draw the velocity-time graph.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nGood. For a 1D motion graph, define the functions before drawing points.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "time interval: 0 <= t <= 6 ; acceleration: positive",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("v(t) = v0 + a*t", response)
        self.assertIn("0 <= t <= 6 s", response)
        self.assertIn("slope +2 m/s^2", response)
        self.assertIn("compute v(6)", response)
        self.assertNotIn("now tell me the time interval", response)

    def test_kinematics_graph_endpoint_is_confirmed(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A car starts from rest and accelerates at 2m/s^2 for 6 s. Find its final velocity and draw the velocity-time graph.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nGood. For a 1D motion graph, define the functions before drawing points.",
                },
                {
                    "role": "user",
                    "content": "time interval: 0 <= t <= 6 ; acceleration: positive",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nFor the velocity-time graph, use v(t) = v0 + a*t.",
                },
            ],
            "total_messages": 4,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "v(6) = 12 m/s",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("correct", response)
        self.assertIn("(6 s, 12 m/s)", response)
        self.assertIn("straight line", response)
        self.assertIn("velocity-time graph", response)

    def test_random_thermodynamics_prompt_gets_specific_equation_choices(self):
        decision = evaluate_guided_tutoring(
            "thermodynamics_agent",
            "A gas expands from 2.0 L to 5.0 L at constant pressure of 120 kPa. What work is done?",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("120 kpa", response)
        self.assertIn("constant pressure", response)
        self.assertIn("w = p*delta v", response)
        self.assertIn("delta u = q - w", response)
        self.assertNotIn("final answer", response)

    def test_short_unit_followup_uses_previous_force_attempt(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "A 12 kg crate is pushed with 80 N across rough tile where mu_k is 0.35. Find the acceleration.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nChoose the force equation.",
                },
                {
                    "role": "user",
                    "content": "Knowns: m=12 kg, push=80 N, mu_k=0.35. Use sum F_x = m*a and f_k = mu_k*N.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nAdd units before calculating.",
                },
            ],
            "total_messages": 4,
        }

        decision = evaluate_guided_tutoring(
            "forces_agent",
            "final unit should be m/s^2",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("newton", response)
        self.assertIn("acceleration", response)
        self.assertIn("m/s^2", response)
        self.assertNotIn("what equation would you write", response)

    def test_random_wave_graph_followup_advances_to_plotting_points(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "Graph wavelength versus frequency for sound in air at 340 m/s.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nDefine graph axes and relation.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "waves_agent",
            "Horizontal axis is frequency in Hz, vertical axis is wavelength in m. Use v = f lambda, so it is an inverse curve.",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("plotting anchors", response)
        self.assertIn("v = f*lambda", response)
        self.assertIn("two or three", response)
        self.assertNotIn("state the horizontal-axis", response)

    def test_projectile_graph_components_advance_to_landing_condition(self):
        context = {
            "recent_conversation": [
                {
                    "role": "user",
                    "content": "Graph a ball kicked from a 20 m hill at 25 m/s at 40 degrees.",
                },
                {
                    "role": "assistant",
                    "content": f"**{GUIDED_RESPONSE_MARKER}**\n\nResolve launch velocity first.",
                },
            ],
            "total_messages": 2,
        }

        decision = evaluate_guided_tutoring(
            "kinematics_agent",
            "v0x = 25 cos(40), v0y = 25 sin(40)",
            context,
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("velocity components", response)
        self.assertIn("y(t) = h0 + v0y*t - (1/2)g*t^2", response)
        self.assertIn("y(t) = 0", response)
        self.assertNotIn("send v0x, v0y", response)

    def test_random_optics_prompt_is_prompt_specific(self):
        decision = evaluate_guided_tutoring(
            "optics_agent",
            "A candle is 32 cm from a converging lens with focal length 12 cm. Determine the image distance and magnification.",
        )

        response = decision.response.lower()

        self.assertTrue(decision.intercept)
        self.assertIn("32 cm", response)
        self.assertIn("12 cm", response)
        self.assertIn("thin lens", response)
        self.assertIn("1/f = 1/do + 1/di", response)
        self.assertIn("m = -di/do", response)
        self.assertNotIn("final answer", response)


if __name__ == "__main__":
    unittest.main()
