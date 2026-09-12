import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    return (ROOT / relative_path).read_text()


class DatabaseBootstrapStaticTests(unittest.TestCase):
    def test_existing_db_startup_applies_additive_migrations(self):
        source = read_repo_file("database/setup_schema.py")

        self.assertIn("_table_exists", source)
        self.assertIn("Core tables already exist", source)
        self.assertIn("03_hitl_knowledge_transfer.sql", source)
        self.assertIn("04_agent_type_enum_updates.sql", source)
        self.assertLess(
            source.index("03_hitl_knowledge_transfer.sql"),
            source.index("04_agent_type_enum_updates.sql"),
        )

    def test_fresh_db_init_runs_hitl_and_enum_migrations(self):
        script = read_repo_file("docker/database/postgresql/init/02-create-schema.sh")
        required_scripts = [
            "01_core_tables.sql",
            "03_hitl_knowledge_transfer.sql",
            "04_agent_type_enum_updates.sql",
            "02_sample_data.sql",
        ]

        for schema_file in required_scripts:
            self.assertIn(schema_file, script)

        ordered_positions = [script.index(schema_file) for schema_file in required_scripts]
        self.assertEqual(ordered_positions, sorted(ordered_positions))
        self.assertIn("ON_ERROR_STOP=1", script)

    def test_hitl_schema_is_additive_and_idempotent(self):
        schema = read_repo_file("database/schema/03_hitl_knowledge_transfer.sql")

        self.assertRegex(schema, r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+knowledge_transfer_guidance")
        self.assertRegex(schema, r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+knowledge_transfer_attempts")
        self.assertIn("CREATE UNIQUE INDEX IF NOT EXISTS idx_kt_guidance_unique_active", schema)
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_kt_attempts_user", schema)
        self.assertIn("ON CONFLICT (agent_type, concept_tag, question_text)", schema)

    def test_agent_enum_covers_current_agents(self):
        core_schema = read_repo_file("database/schema/01_core_tables.sql")
        enum_migration = read_repo_file("database/schema/04_agent_type_enum_updates.sql")
        combined_sql = f"{core_schema}\n{enum_migration}"
        required_enum_values = {
            "kinematics",
            "forces",
            "energy",
            "momentum",
            "angular_motion",
            "math_helper",
            "thermodynamics",
            "waves",
            "electromagnetism",
            "optics",
            "modern_physics",
        }

        for enum_value in required_enum_values:
            self.assertRegex(combined_sql, rf"'{re.escape(enum_value)}'")

        for migrated_value in {
            "thermodynamics",
            "waves",
            "electromagnetism",
            "optics",
            "modern_physics",
        }:
            self.assertIn(
                f"ALTER TYPE agent_type ADD VALUE IF NOT EXISTS '{migrated_value}'",
                enum_migration,
            )


if __name__ == "__main__":
    unittest.main()
