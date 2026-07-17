import json
import sys
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

DATABASE_DIR = Path(__file__).resolve().parent
if str(DATABASE_DIR) not in sys.path:
    sys.path.insert(0, str(DATABASE_DIR))

import api_server
from db_manager import DatabaseManager, normalize_agent_type


FORCES_Q1_ID = uuid.UUID("11111111-1111-4111-8111-111111111111")
FORCES_Q2_ID = uuid.UUID("22222222-2222-4222-8222-222222222222")


def hitl_row(question_id, key, leg, choices, correct_choice_id, correct_answer):
    return {
        "id": question_id,
        "question_key": key,
        "agent_type": "forces",
        "topic": "newtons_second_law",
        "leg": leg,
        "question_type": "multiple_choice",
        "question_text": "Forces HITL question",
        "choices": json.dumps(choices),
        "correct_choice_id": correct_choice_id,
        "correct_answer": correct_answer,
        "explanation": "Use the force setup before solving.",
        "metadata": json.dumps({"next_leg_on_wrong": 2} if leg == 1 else {}),
    }


class FakeConnectionContext:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePostgres:
    def __init__(self, conn):
        self.conn = conn

    def get_connection(self):
        return FakeConnectionContext(self.conn)


class FakeConnection:
    def __init__(self):
        self.q1 = hitl_row(
            FORCES_Q1_ID,
            "forces_free_body_leg_1",
            1,
            [
                {"id": "free_body_axes", "text": "Draw a free-body diagram and choose useful axes"},
                {"id": "kinematics_first", "text": "Start with kinematics"},
            ],
            "free_body_axes",
            "Draw a free-body diagram and choose useful axes",
        )
        self.q2 = hitl_row(
            FORCES_Q2_ID,
            "forces_newton_second_leg_2",
            2,
            [
                {"id": "sum_fx", "text": "sum F_x = m*a_x"},
                {"id": "vf_squared", "text": "v_f^2 = v_i^2 + 2*a*dx"},
            ],
            "sum_fx",
            "sum F_x = m*a_x",
        )
        self.fetchrow_calls = []
        self.fetchval_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        if "WHERE is_active = TRUE AND leg = $1" in query:
            return self.q1 if args[0] == 1 else self.q2
        if "WHERE id = $1 AND is_active = TRUE" in query:
            return self.q1 if args[0] == FORCES_Q1_ID else self.q2
        if "AND agent_type = $1" in query and "AND leg = $2" in query:
            return self.q2 if args == ("forces", 2) else None
        return None

    async def fetchval(self, query, *args):
        self.fetchval_calls.append((query, args))
        return uuid.UUID("33333333-3333-4333-8333-333333333333")

    async def fetch(self, query, *args):
        return [
            {
                "agent_type": "forces",
                "total_attempts": 2,
                "correct_attempts": 1,
                "incorrect_attempts": 1,
            }
        ]


class FakeDatabase:
    def __init__(self):
        self.conn = FakeConnection()
        self.postgres = FakePostgres(self.conn)
        self.logged_interactions = []
        self._initialized = True

    async def log_interaction(self, **kwargs):
        self.logged_interactions.append(kwargs)
        return "44444444-4444-4444-8444-444444444444"


@pytest.fixture
def fake_db():
    return FakeDatabase()


@pytest.fixture
def client(fake_db):
    api_server.app.dependency_overrides[api_server.get_db] = lambda: fake_db
    yield TestClient(api_server.app)
    api_server.app.dependency_overrides.clear()


def test_fresh_db_schema_contains_hitl_tables_and_agent_enum_values():
    schema_sql = (DATABASE_DIR / "schema" / "01_core_tables.sql").read_text()

    assert "CREATE TABLE IF NOT EXISTS hitl_questions" in schema_sql
    assert "CREATE TABLE IF NOT EXISTS hitl_attempts" in schema_sql
    assert "ALTER TABLE hitl_questions ADD COLUMN IF NOT EXISTS question_key" in schema_sql
    assert "ALTER TABLE hitl_attempts ADD COLUMN IF NOT EXISTS is_correct" in schema_sql
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_hitl_questions_question_key" in schema_sql
    assert "ALTER TYPE agent_type ADD VALUE IF NOT EXISTS 'thermodynamics'" in schema_sql
    assert "INSERT INTO hitl_questions" in schema_sql


@pytest.mark.asyncio
async def test_existing_db_startup_verifies_schema(monkeypatch):
    db = DatabaseManager()
    db.postgres.initialize = AsyncMock(return_value=True)
    db.postgres.ensure_schema = AsyncMock(return_value=True)
    db.neo4j.initialize = AsyncMock(return_value=True)
    db.redis.initialize = AsyncMock(return_value=True)

    assert await db.initialize() is True
    db.postgres.ensure_schema.assert_awaited_once()


@pytest.mark.parametrize(
    ("incoming_agent_id", "expected_agent_type"),
    [
        ("physics_forces_agent", "forces"),
        ("forces-agent", "forces"),
        ("kinematics_agent", "kinematics"),
        ("physics_energy_agent", "energy"),
        ("momentum-agent", "momentum"),
        ("physics_angular_motion_agent", "angular_motion"),
        ("math_agent", "math"),
        ("physics_thermodynamics_agent", "thermodynamics"),
        ("waves-agent", "waves"),
        ("physics_electromagnetism_agent", "electromagnetism"),
        ("optics_agent", "optics"),
        ("physics_modern_physics_agent", "modern_physics"),
    ],
)
def test_agent_id_normalization_handles_current_agent_forms(incoming_agent_id, expected_agent_type):
    assert normalize_agent_type(incoming_agent_id) == expected_agent_type


def test_thermodynamics_interaction_logging_uses_database_contract(client, fake_db):
    response = client.post(
        "/interactions/log",
        json={
            "user_id": "react_user",
            "agent_id": "physics_thermodynamics_agent",
            "problem": "Find the heat added during an isobaric process.",
            "solution": "Use Q = n*C_p*Delta T.",
            "tools_used": ["thermodynamics_solver"],
            "execution_time_ms": 125,
            "framework": "strands",
        },
    )

    assert response.status_code == 200
    assert response.json()["interaction_id"] == "44444444-4444-4444-8444-444444444444"
    assert fake_db.logged_interactions[0]["agent_type"] == "thermodynamics"
    assert fake_db.logged_interactions[0]["interaction_type"] == "agent_call"
    assert fake_db.logged_interactions[0]["metadata"]["framework"] == "strands"


def test_forces_agent_hitl_first_leg_question_flow_decodes_jsonb(client):
    response = client.get("/knowledge-transfer/questions/pick?agent_type=forces-agent")

    assert response.status_code == 200
    question = response.json()["question"]
    assert question["agent_type"] == "forces"
    assert question["leg"] == 1
    assert question["choices"][0]["id"] == "free_body_axes"
    assert "correct_choice_id" not in question


def test_wrong_first_leg_returns_second_hitl_leg_then_scores_correct(client, fake_db):
    wrong_response = client.post(
        "/knowledge-transfer/attempts",
        json={
            "user_id": "react_user",
            "question_id": str(FORCES_Q1_ID),
            "agent_type": "physics_forces_agent",
            "selected_choice_id": "kinematics_first",
        },
    )

    assert wrong_response.status_code == 200
    wrong_payload = wrong_response.json()
    assert wrong_payload["is_correct"] is False
    assert wrong_payload["proceed"] is False
    assert wrong_payload["next_question"]["id"] == str(FORCES_Q2_ID)
    assert wrong_payload["next_question"]["leg"] == 2

    first_insert_args = fake_db.conn.fetchval_calls[0][1]
    assert first_insert_args[3] == "forces"
    assert first_insert_args[6] is False

    correct_response = client.post(
        "/knowledge-transfer/attempts",
        json={
            "user_id": "react_user",
            "question_id": str(FORCES_Q2_ID),
            "agent_type": "forces-agent",
            "selected_choice_id": "sum_fx",
        },
    )

    assert correct_response.status_code == 200
    correct_payload = correct_response.json()
    assert correct_payload["is_correct"] is True
    assert correct_payload["proceed"] is True
    assert correct_payload["next_question"] is None

    second_insert_args = fake_db.conn.fetchval_calls[1][1]
    assert second_insert_args[3] == "forces"
    assert second_insert_args[6] is True


def test_hitl_attempt_aggregate_normalizes_agent_filter(client):
    response = client.get(
        "/knowledge-transfer/attempts/aggregate?user_id=react_user&agent_type=physics_forces_agent"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_attempts"] == 2
    assert payload["correct_attempts"] == 1
    assert payload["incorrect_attempts"] == 1
    assert payload["accuracy"] == 0.5
    assert payload["by_agent"][0]["agent_type"] == "forces"
