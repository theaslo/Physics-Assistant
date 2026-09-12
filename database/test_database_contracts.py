import asyncio
import json
import sys
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_server


class FakeConnectionContext:
    def __init__(self, conn: "FakeConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "FakeConnection":
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class FakePostgres:
    def __init__(self, conn: "FakeConnection") -> None:
        self.conn = conn

    def get_connection(self) -> FakeConnectionContext:
        return FakeConnectionContext(self.conn)


class FakeDatabase:
    def __init__(self, conn: "FakeConnection") -> None:
        self.postgres = FakePostgres(conn)
        self.logged_interaction: Optional[Dict[str, Any]] = None

    async def log_interaction(self, **kwargs: Any) -> str:
        self.logged_interaction = kwargs
        return "11111111-1111-4111-8111-111111111111"


class FakeBackgroundTasks:
    def __init__(self) -> None:
        self.tasks: List[Dict[str, Any]] = []

    def add_task(self, fn, *args: Any, **kwargs: Any) -> None:
        self.tasks.append({"fn": fn, "args": args, "kwargs": kwargs})


class FakeConnection:
    def __init__(
        self,
        fetchrow_results: Optional[List[Dict[str, Any]]] = None,
        fetch_results: Optional[List[List[Dict[str, Any]]]] = None,
        fetchval_results: Optional[List[Any]] = None,
    ) -> None:
        self.fetchrow_results = list(fetchrow_results or [])
        self.fetch_results = list(fetch_results or [])
        self.fetchval_results = list(fetchval_results or [])
        self.calls: List[Dict[str, Any]] = []

    async def fetchrow(self, query: str, *args: Any) -> Dict[str, Any]:
        self.calls.append({"method": "fetchrow", "query": query, "args": args})
        if not self.fetchrow_results:
            raise AssertionError(f"Unexpected fetchrow call: {query}")
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        self.calls.append({"method": "fetch", "query": query, "args": args})
        if not self.fetch_results:
            raise AssertionError(f"Unexpected fetch call: {query}")
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.calls.append({"method": "fetchval", "query": query, "args": args})
        if not self.fetchval_results:
            raise AssertionError(f"Unexpected fetchval call: {query}")
        return self.fetchval_results.pop(0)


class DatabaseApiContractTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_normalize_agent_type_accepts_current_identifier_forms(self):
        cases = {
            "physics_thermodynamics_agent": "thermodynamics",
            "thermodynamics_agent": "thermodynamics",
            "physics_forces": "forces",
            "forces_agent": "forces",
            "physics_kinematics_agent": "kinematics",
            "kinematics_agent": "kinematics",
            "math_agent": "math_helper",
            " physics_modern_physics_agent ": "modern_physics",
        }

        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(api_server.normalize_agent_type(raw_value), expected)

    def test_pick_question_decodes_jsonb_strings_and_hides_answer_from_options(self):
        question_id = uuid.UUID("22222222-2222-4222-8222-222222222222")
        row = {
            "id": question_id,
            "agent_type": "forces",
            "concept_tag": "hookes_law",
            "question_text": "For an ideal spring, which relation is Hooke's law?",
            "options": json.dumps([
                {
                    "id": "A",
                    "text": "F = kx",
                    "is_correct": True,
                    "feedback": "Correct for magnitudes.",
                },
                {
                    "id": "B",
                    "text": "F = x/k",
                    "is_correct": False,
                    "feedback": "This inverts the relationship.",
                },
            ]),
            "explanation_correct": "Hooke's law is linear in displacement.",
            "explanation_incorrect": "Use force proportional to displacement.",
            "difficulty": 1,
            "metadata": json.dumps({"source": "seed"}),
        }
        conn = FakeConnection(fetchrow_results=[row])
        db = FakeDatabase(conn)

        result = self.run_async(
            api_server.pick_knowledge_transfer_question(
                agent_type="physics_forces_agent",
                concept_tag="hookes_law",
                db=db,
            )
        )

        self.assertEqual(result["question_id"], str(question_id))
        self.assertEqual(result["agent_type"], "forces")
        self.assertEqual(result["correct_option_id"], "A")
        self.assertEqual(result["distractor_feedback"], {"B": "This inverts the relationship."})
        self.assertEqual(result["metadata"], {"source": "seed"})
        self.assertEqual(result["options"], [{"id": "A", "text": "F = kx"}, {"id": "B", "text": "F = x/k"}])
        self.assertNotIn("is_correct", result["options"][0])
        self.assertEqual(conn.calls[0]["args"][0], "forces")

    def test_post_attempt_normalizes_agent_and_persists_scoring_payload(self):
        created_at = datetime(2026, 9, 12, 12, 0, 0)
        attempt_id = uuid.UUID("33333333-3333-4333-8333-333333333333")
        conn = FakeConnection(fetchrow_results=[{"id": attempt_id, "created_at": created_at}])
        db = FakeDatabase(conn)
        attempt = api_server.KnowledgeTransferAttemptRequest(
            user_identifier="student-a",
            session_identifier="session-a",
            class_identifier="class-a",
            agent_type="physics_thermodynamics_agent",
            concept_tag="ideal_gas_law",
            question_id="44444444-4444-4444-8444-444444444444",
            check_id="check-a",
            confidence_score=0.92,
            threshold_score=0.70,
            selected_option_id="A",
            was_correct=True,
            metadata={"source": "hitl_gate"},
        )

        result = self.run_async(api_server.log_knowledge_transfer_attempt(attempt=attempt, db=db))

        call = conn.calls[0]
        self.assertIn("INSERT INTO knowledge_transfer_attempts", call["query"])
        self.assertEqual(call["args"][4], "thermodynamics")
        self.assertEqual(call["args"][9], "A")
        self.assertTrue(call["args"][10])
        self.assertEqual(json.loads(call["args"][11]), {"source": "hitl_gate"})
        self.assertEqual(result["attempt_id"], str(attempt_id))
        self.assertEqual(result["status"], "success")

    def test_aggregate_attempts_normalizes_agent_filter_and_computes_accuracy(self):
        conn = FakeConnection(
            fetchrow_results=[{"total_attempts": 2, "total_correct": 1, "distinct_students": 1}],
            fetch_results=[[{"concept_tag": "hookes_law", "attempts": 2, "correct": 1}]],
        )
        db = FakeDatabase(conn)

        result = self.run_async(
            api_server.aggregate_knowledge_transfer_attempts(
                user_identifier="student-a",
                agent_type="physics_forces_agent",
                days=14,
                db=db,
            )
        )

        self.assertEqual(result["filters"]["agent_type"], "forces")
        self.assertEqual(result["summary"]["total_attempts"], 2)
        self.assertEqual(result["summary"]["total_correct"], 1)
        self.assertEqual(result["summary"]["accuracy"], 0.5)
        self.assertEqual(result["by_concept"][0]["accuracy"], 0.5)
        self.assertEqual(conn.calls[0]["args"][2], "forces")
        self.assertEqual(conn.calls[1]["args"][2], "forces")

    def test_interactions_log_alias_matches_agent_logging_contract(self):
        resolved_user = uuid.UUID("55555555-5555-4555-8555-555555555555")
        conn = FakeConnection(fetchrow_results=[{"id": resolved_user}])
        db = FakeDatabase(conn)
        request = api_server.InteractionRequest(
            user_id="student-a",
            session_id=None,
            agent_type="physics_thermodynamics_agent",
            message="Use the ideal gas law.",
            response="PV = nRT",
            execution_time_ms=123,
            metadata={"tools_used": ["ideal_gas_law"], "framework": "strands"},
        )
        background_tasks = FakeBackgroundTasks()

        result = self.run_async(
            api_server.log_interaction_compat(
                interaction=request,
                background_tasks=background_tasks,
                db=db,
            )
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(db.logged_interaction["user_id"], str(resolved_user))
        self.assertEqual(db.logged_interaction["agent_type"], "thermodynamics")
        self.assertEqual(db.logged_interaction["message"], "Use the ideal gas law.")
        self.assertEqual(db.logged_interaction["response"], "PV = nRT")
        self.assertEqual(db.logged_interaction["metadata"]["tools_used"], ["ideal_gas_law"])
        self.assertEqual(len(background_tasks.tasks), 1)


if __name__ == "__main__":
    unittest.main()
