import unittest
from pathlib import Path


DATABASE_DIR = Path(__file__).resolve().parent


class AnalyticsStartupContractTests(unittest.TestCase):
    def test_database_api_startup_analytics_imports_are_available(self):
        import analytics.concept_mastery
        import analytics.educational_data_mining
        import analytics.learning_analytics
        import analytics.learning_path_optimizer
        import analytics.realtime_analytics

        self.assertTrue(analytics.concept_mastery)
        self.assertTrue(analytics.educational_data_mining)
        self.assertTrue(analytics.learning_analytics)
        self.assertTrue(analytics.learning_path_optimizer)
        self.assertTrue(analytics.realtime_analytics)

    def test_user_progress_decimal_values_are_converted_before_float_math(self):
        source = (DATABASE_DIR / "analytics" / "learning_analytics.py").read_text()

        self.assertIn("proficiency_score = float(progress['proficiency_score'] or 0.0)", source)
        self.assertIn("proficiency_score / 100.0", source)

    def test_agent_type_enum_is_cast_before_joining_to_progress_topic(self):
        source = (DATABASE_DIR / "analytics" / "educational_data_mining.py").read_text()

        self.assertIn("i.agent_type::text = up.topic", source)
        self.assertIn("proficiency_score if proficiency_score is not None else 50.0", source)


if __name__ == "__main__":
    unittest.main()
