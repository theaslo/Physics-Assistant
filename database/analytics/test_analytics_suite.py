#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Suite for Physics Assistant Analytics
Tests all analytics algorithms, validates model performance, and ensures
system reliability and accuracy of educational insights.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import pytest
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

# Import analytics modules
try:
    from learning_analytics import LearningAnalyticsEngine, StudentProfile
    from concept_mastery import ConceptMasteryDetector, ConceptAssessment, MisconcepationPattern
    from learning_path_optimizer import LearningPathOptimizer, LearningObjective, LearningPath
    from educational_data_mining import EducationalDataMiner, LearningPattern, PerformancePrediction
    from realtime_analytics import RealTimeAnalyticsEngine, AnalyticsEvent, EventType
except ImportError:
    # Handle relative imports for testing
    import sys
    sys.path.append('.')
    from learning_analytics import LearningAnalyticsEngine, StudentProfile
    from concept_mastery import ConceptMasteryDetector, ConceptAssessment
    from learning_path_optimizer import LearningPathOptimizer, LearningObjective
    from educational_data_mining import EducationalDataMiner
    from realtime_analytics import RealTimeAnalyticsEngine, AnalyticsEvent, EventType

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDatabaseManager:
    """Mock database manager for testing"""
    
    def __init__(self):
        self.postgres = Mock()
        self.neo4j = Mock()
        self.redis = Mock()
        self._initialized = True
        
        # Setup mock data
        self._setup_mock_data()
    
    def _setup_mock_data(self):
        """Setup mock data for testing"""
        # Mock PostgreSQL data
        self.mock_interactions = [
            {
                'id': 'test_interaction_1',
                'user_id': 'test_user_1', 
                'agent_type': 'kinematics',
                'success': True,
                'created_at': datetime.now() - timedelta(days=1),
                'execution_time_ms': 15000,
                'metadata': '{"difficulty_level": 1.0}'
            },
            {
                'id': 'test_interaction_2',
                'user_id': 'test_user_1',
                'agent_type': 'kinematics', 
                'success': False,
                'created_at': datetime.now(),
                'execution_time_ms': 25000,
                'metadata': '{"difficulty_level": 1.5, "error_type": "unit_error"}'
            }
        ]
        
        self.mock_users = [
            {
                'id': 'test_user_1',
                'username': 'test_student',
                'created_at': datetime.now() - timedelta(days=30),
                'is_active': True
            }
        ]
        
        self.mock_progress = [
            {
                'user_id': 'test_user_1',
                'topic': 'kinematics',
                'problems_attempted': 10,
                'problems_solved': 7,
                'proficiency_score': 70.0
            }
        ]
        
        # Setup mock methods
        self.postgres.get_connection = self._mock_postgres_connection
        self.neo4j.run_query = AsyncMock(return_value=[
            {'name': 'kinematics', 'category': 'mechanics', 'difficulty': 0.5},
            {'name': 'forces', 'category': 'mechanics', 'difficulty': 0.7}
        ])
        self.redis.get = AsyncMock(return_value=None)
        self.redis.set = AsyncMock(return_value=True)
    
    async def _mock_postgres_connection(self):
        """Mock PostgreSQL connection context manager"""
        class MockConnection:
            def __init__(self, mock_data):
                self.mock_data = mock_data
            
            async def fetch(self, query, *args):
                # Return appropriate mock data based on query
                if 'interactions' in query.lower():
                    return self.mock_data.mock_interactions
                elif 'users' in query.lower():
                    return self.mock_data.mock_users
                elif 'user_progress' in query.lower():
                    return self.mock_data.mock_progress
                else:
                    return []
            
            async def fetchval(self, query, *args):
                if 'EXISTS' in query:
                    return True
                return 1
            
            async def fetchrow(self, query, *args):
                if 'user_progress' in query.lower():
                    return self.mock_data.mock_progress[0] if self.mock_data.mock_progress else None
                return self.mock_data.mock_interactions[0] if self.mock_data.mock_interactions else None
            
            async def execute(self, query, *args):
                return "INSERT 0 1"
        
        class MockContextManager:
            def __init__(self, mock_data):
                self.mock_data = mock_data
            
            async def __aenter__(self):
                return MockConnection(self.mock_data)
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
        
        return MockContextManager(self)

class TestLearningAnalyticsEngine(unittest.TestCase):
    """Test cases for Learning Analytics Engine"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_db = MockDatabaseManager()
        self.analytics_engine = LearningAnalyticsEngine(self.mock_db)
    
    async def test_initialization(self):
        """Test analytics engine initialization"""
        try:
            result = await self.analytics_engine.initialize()
            self.assertTrue(result)
            logger.info("✅ Analytics engine initialization test passed")
        except Exception as e:
            self.fail(f"Analytics engine initialization failed: {e}")
    
    async def test_student_progress_tracking(self):
        """Test student progress tracking"""
        try:
            # Initialize analytics engine
            await self.analytics_engine.initialize()
            
            # Test progress tracking
            progress = await self.analytics_engine.track_student_progress("test_user_1", 30)
            
            # Validate progress data structure
            self.assertIsInstance(progress, dict)
            self.assertIn('user_id', progress)
            self.assertIn('overall_mastery', progress)
            self.assertIn('learning_velocity', progress)
            self.assertIn('engagement_score', progress)
            
            logger.info("✅ Student progress tracking test passed")
        except Exception as e:
            self.fail(f"Student progress tracking test failed: {e}")
    
    async def test_learning_difficulty_detection(self):
        """Test learning difficulty detection"""
        try:
            await self.analytics_engine.initialize()
            
            difficulties = await self.analytics_engine.detect_learning_difficulties("test_user_1")
            
            # Validate response structure
            self.assertIsInstance(difficulties, dict)
            self.assertIn('warning_level', difficulties)
            self.assertIn('recommendations', difficulties)
            
            logger.info("✅ Learning difficulty detection test passed")
        except Exception as e:
            self.fail(f"Learning difficulty detection test failed: {e}")
    
    async def test_learning_efficiency_calculation(self):
        """Test learning efficiency calculation"""
        try:
            await self.analytics_engine.initialize()
            
            efficiency = await self.analytics_engine.calculate_learning_efficiency("test_user_1")
            
            # Validate efficiency metrics
            self.assertIsInstance(efficiency, dict)
            self.assertIn('efficiency_score', efficiency)
            self.assertIn('success_efficiency', efficiency)
            self.assertIn('time_efficiency', efficiency)
            
            # Check score ranges
            self.assertGreaterEqual(efficiency['efficiency_score'], 0.0)
            self.assertLessEqual(efficiency['efficiency_score'], 1.0)
            
            logger.info("✅ Learning efficiency calculation test passed")
        except Exception as e:
            self.fail(f"Learning efficiency calculation test failed: {e}")

class TestConceptMasteryDetector(unittest.TestCase):
    """Test cases for Concept Mastery Detector"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_db = MockDatabaseManager()
        self.mastery_detector = ConceptMasteryDetector(self.mock_db)
    
    async def test_concept_mastery_assessment(self):
        """Test concept mastery assessment"""
        try:
            assessment = await self.mastery_detector.assess_concept_mastery(
                "test_user_1", "kinematics", 14
            )
            
            # Validate assessment structure
            self.assertIsInstance(assessment, ConceptAssessment)
            self.assertEqual(assessment.concept_name, "kinematics")
            self.assertGreaterEqual(assessment.mastery_score, 0.0)
            self.assertLessEqual(assessment.mastery_score, 1.0)
            
            logger.info("✅ Concept mastery assessment test passed")
        except Exception as e:
            self.fail(f"Concept mastery assessment test failed: {e}")
    
    async def test_misconception_detection(self):
        """Test misconception detection"""
        try:
            misconceptions = await self.mastery_detector.detect_misconceptions(
                "test_user_1", "kinematics"
            )
            
            # Validate misconceptions structure
            self.assertIsInstance(misconceptions, list)
            for misconception in misconceptions:
                self.assertIsInstance(misconception, MisconcepationPattern)
                self.assertIsInstance(misconception.misconception_id, str)
                self.assertIsInstance(misconception.description, str)
            
            logger.info("✅ Misconception detection test passed")
        except Exception as e:
            self.fail(f"Misconception detection test failed: {e}")
    
    def test_error_pattern_extraction(self):
        """Test error pattern extraction"""
        try:
            # Test error type extraction
            error_types = self.mastery_detector._extract_error_types(
                "Unit conversion error in velocity calculation",
                {"error_type": "unit_error"},
                None, None, "kinematics"
            )
            
            self.assertIsInstance(error_types, list)
            self.assertTrue(len(error_types) > 0)
            
            logger.info("✅ Error pattern extraction test passed")
        except Exception as e:
            self.fail(f"Error pattern extraction test failed: {e}")

class TestLearningPathOptimizer(unittest.TestCase):
    """Test cases for Learning Path Optimizer"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_db = MockDatabaseManager()
        self.path_optimizer = LearningPathOptimizer(self.mock_db)
    
    async def test_path_optimizer_initialization(self):
        """Test path optimizer initialization"""
        try:
            result = await self.path_optimizer.initialize()
            self.assertTrue(result)
            logger.info("✅ Path optimizer initialization test passed")
        except Exception as e:
            self.fail(f"Path optimizer initialization failed: {e}")
    
    async def test_learning_path_generation(self):
        """Test learning path generation"""
        try:
            await self.path_optimizer.initialize()
            
            # Create test objective
            objective = LearningObjective(
                target_concepts=["forces", "energy"],
                difficulty_preference="adaptive"
            )
            
            # Generate path
            path = await self.path_optimizer.generate_learning_path(
                "test_user_1", objective
            )
            
            # Validate path structure
            self.assertIsInstance(path, LearningPath)
            self.assertIsInstance(path.concept_sequence, list)
            self.assertGreaterEqual(path.estimated_total_time, 0.0)
            self.assertGreaterEqual(path.success_probability, 0.0)
            self.assertLessEqual(path.success_probability, 1.0)
            
            logger.info("✅ Learning path generation test passed")
        except Exception as e:
            self.fail(f"Learning path generation test failed: {e}")
    
    def test_transition_cost_calculation(self):
        """Test transition cost calculation"""
        try:
            # Create mock student state
            from learning_path_optimizer import StudentState
            
            student_state = StudentState(
                concept_masteries={"kinematics": 0.7},
                learning_velocity=0.5,
                engagement_level=0.8,
                preferred_difficulty=0.6,
                strong_areas=["kinematics"],
                weak_areas=[],
                learning_patterns={}
            )
            
            # Create mock objective
            objective = LearningObjective(
                target_concepts=["forces"],
                difficulty_preference="adaptive"
            )
            
            # Test cost calculation (requires concept graph)
            if hasattr(self.path_optimizer, '_calculate_transition_cost'):
                cost = self.path_optimizer._calculate_transition_cost(
                    "kinematics", "forces", student_state, objective
                )
                self.assertIsInstance(cost, float)
                self.assertGreaterEqual(cost, 0.0)
            
            logger.info("✅ Transition cost calculation test passed")
        except Exception as e:
            self.fail(f"Transition cost calculation test failed: {e}")

class TestEducationalDataMiner(unittest.TestCase):
    """Test cases for Educational Data Miner"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_db = MockDatabaseManager()
        self.data_miner = EducationalDataMiner(self.mock_db)
    
    async def test_data_miner_initialization(self):
        """Test data miner initialization"""
        try:
            result = await self.data_miner.initialize()
            self.assertTrue(result)
            logger.info("✅ Data miner initialization test passed")
        except Exception as e:
            self.fail(f"Data miner initialization failed: {e}")
    
    async def test_performance_prediction(self):
        """Test performance prediction"""
        try:
            await self.data_miner.initialize()
            
            prediction = await self.data_miner.predict_student_performance(
                "test_user_1", "success_rate"
            )
            
            # Validate prediction structure
            self.assertIsInstance(prediction, PerformancePrediction)
            self.assertEqual(prediction.student_id, "test_user_1")
            self.assertGreaterEqual(prediction.predicted_value, 0.0)
            self.assertLessEqual(prediction.predicted_value, 1.0)
            
            logger.info("✅ Performance prediction test passed")
        except Exception as e:
            self.fail(f"Performance prediction test failed: {e}")
    
    async def test_student_clustering(self):
        """Test student clustering"""
        try:
            await self.data_miner.initialize()
            
            clusters = await self.data_miner.identify_student_clusters()
            
            # Validate clustering results
            self.assertIsInstance(clusters, list)
            
            logger.info("✅ Student clustering test passed")
        except Exception as e:
            self.fail(f"Student clustering test failed: {e}")
    
    async def test_educational_insights(self):
        """Test educational insights generation"""
        try:
            await self.data_miner.initialize()
            
            insights = await self.data_miner.generate_educational_insights(30)
            
            # Validate insights structure
            self.assertIsInstance(insights, list)
            
            logger.info("✅ Educational insights test passed")
        except Exception as e:
            self.fail(f"Educational insights test failed: {e}")
    
    def test_pattern_detection_methods(self):
        """Test pattern detection utility methods"""
        try:
            # Test session regularity calculation
            mock_interactions = [
                {'timestamp': datetime.now() - timedelta(hours=i)} 
                for i in range(5)
            ]
            
            regularity = self.data_miner._calculate_session_regularity(mock_interactions)
            self.assertIsInstance(regularity, float)
            self.assertGreaterEqual(regularity, 0.0)
            self.assertLessEqual(regularity, 1.0)
            
            # Test concept switching rate
            mock_interactions = [
                {'agent_type': 'kinematics'},
                {'agent_type': 'forces'},
                {'agent_type': 'kinematics'}
            ]
            
            switching_rate = self.data_miner._calculate_concept_switching_rate(mock_interactions)
            self.assertIsInstance(switching_rate, float)
            self.assertGreaterEqual(switching_rate, 0.0)
            self.assertLessEqual(switching_rate, 1.0)
            
            logger.info("✅ Pattern detection methods test passed")
        except Exception as e:
            self.fail(f"Pattern detection methods test failed: {e}")

class TestRealTimeAnalytics(unittest.TestCase):
    """Test cases for Real-time Analytics Engine"""
    
    def setUp(self):
        """Setup test environment"""
        self.mock_db = MockDatabaseManager()
        self.realtime_engine = RealTimeAnalyticsEngine(self.mock_db)
    
    async def test_realtime_engine_startup_shutdown(self):
        """Test real-time engine startup and shutdown"""
        try:
            # Test startup
            start_task = asyncio.create_task(self.realtime_engine.start())
            await asyncio.sleep(0.1)  # Let it start
            
            # Test that engine is running
            self.assertTrue(self.realtime_engine.is_running)
            
            # Test shutdown
            await self.realtime_engine.stop()
            start_task.cancel()
            
            self.assertFalse(self.realtime_engine.is_running)
            
            logger.info("✅ Real-time engine startup/shutdown test passed")
        except Exception as e:
            self.fail(f"Real-time engine startup/shutdown test failed: {e}")
    
    async def test_event_submission_and_processing(self):
        """Test event submission and processing"""
        try:
            # Start engine
            start_task = asyncio.create_task(self.realtime_engine.start())
            await asyncio.sleep(0.1)
            
            # Create test event
            from realtime_analytics import create_interaction_event
            
            event = create_interaction_event(
                "test_user_1", "kinematics", True, 15000
            )
            
            # Submit event
            await self.realtime_engine.submit_event(event)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Check processing stats
            stats = self.realtime_engine.get_processing_stats()
            self.assertGreaterEqual(stats['events_processed'], 1)
            
            # Cleanup
            await self.realtime_engine.stop()
            start_task.cancel()
            
            logger.info("✅ Event submission and processing test passed")
        except Exception as e:
            self.fail(f"Event submission and processing test failed: {e}")
    
    def test_event_processors(self):
        """Test individual event processors"""
        try:
            from realtime_analytics import InteractionProcessor, MasteryProcessor
            
            # Test interaction processor
            interaction_processor = InteractionProcessor()
            self.assertTrue(interaction_processor.is_active)
            self.assertEqual(interaction_processor.name, "interaction_processor")
            
            # Test mastery processor
            mastery_processor = MasteryProcessor()
            self.assertTrue(mastery_processor.is_active)
            self.assertEqual(mastery_processor.name, "mastery_processor")
            
            logger.info("✅ Event processors test passed")
        except Exception as e:
            self.fail(f"Event processors test failed: {e}")
    
    def test_intervention_engine(self):
        """Test intervention engine"""
        try:
            from realtime_analytics import InterventionEngine, StreamingUpdate
            
            # Create intervention engine
            intervention_engine = InterventionEngine()
            
            # Test with mock update
            mock_update = StreamingUpdate(
                update_id="test_update",
                user_id="test_user_1",
                metric_type="success_rate",
                data={"success_rate": 0.3},  # Low success rate should trigger intervention
                timestamp=datetime.now()
            )
            
            # This would be tested with actual async call in full implementation
            self.assertIsNotNone(intervention_engine.intervention_rules)
            self.assertTrue(len(intervention_engine.intervention_rules) > 0)
            
            logger.info("✅ Intervention engine test passed")
        except Exception as e:
            self.fail(f"Intervention engine test failed: {e}")

class PerformanceTests(unittest.TestCase):
    """Performance and load testing for analytics components"""
    
    def setUp(self):
        """Setup performance test environment"""
        self.mock_db = MockDatabaseManager()
    
    async def test_analytics_performance_under_load(self):
        """Test analytics performance under load"""
        try:
            analytics_engine = LearningAnalyticsEngine(self.mock_db)
            await analytics_engine.initialize()
            
            # Simulate multiple concurrent requests
            tasks = []
            start_time = datetime.now()
            
            for i in range(10):
                task = analytics_engine.track_student_progress(f"test_user_{i}", 30)
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Validate results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            self.assertEqual(len(successful_results), 10)
            
            # Performance assertion (should complete within reasonable time)
            self.assertLess(execution_time, 10.0)  # 10 seconds max
            
            logger.info(f"✅ Performance test passed: {len(successful_results)} requests in {execution_time:.2f}s")
            
        except Exception as e:
            self.fail(f"Performance test failed: {e}")
    
    async def test_realtime_event_throughput(self):
        """Test real-time event processing throughput"""
        try:
            realtime_engine = RealTimeAnalyticsEngine(self.mock_db)
            
            # Start engine
            start_task = asyncio.create_task(realtime_engine.start())
            await asyncio.sleep(0.1)
            
            # Submit multiple events rapidly
            from realtime_analytics import create_interaction_event
            
            start_time = datetime.now()
            event_count = 50
            
            for i in range(event_count):
                event = create_interaction_event(
                    f"user_{i % 5}", "kinematics", True, 15000
                )
                await realtime_engine.submit_event(event)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Check processing stats
            stats = realtime_engine.get_processing_stats()
            
            # Calculate throughput
            throughput = stats['events_processed'] / execution_time
            
            # Performance assertions
            self.assertGreaterEqual(stats['events_processed'], event_count * 0.8)  # At least 80% processed
            self.assertGreater(throughput, 10)  # At least 10 events per second
            
            # Cleanup
            await realtime_engine.stop()
            start_task.cancel()
            
            logger.info(f"✅ Throughput test passed: {throughput:.2f} events/sec")
            
        except Exception as e:
            self.fail(f"Throughput test failed: {e}")

class IntegrationTests(unittest.TestCase):
    """Integration tests for full analytics pipeline"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.mock_db = MockDatabaseManager()
    
    async def test_full_analytics_pipeline(self):
        """Test complete analytics pipeline integration"""
        try:
            # Initialize all components
            analytics_engine = LearningAnalyticsEngine(self.mock_db)
            mastery_detector = ConceptMasteryDetector(self.mock_db)
            path_optimizer = LearningPathOptimizer(self.mock_db)
            data_miner = EducationalDataMiner(self.mock_db)
            
            # Initialize engines
            await analytics_engine.initialize()
            await path_optimizer.initialize()
            await data_miner.initialize()
            
            # Test pipeline: Progress -> Mastery -> Path -> Insights
            user_id = "test_user_1"
            
            # 1. Get student progress
            progress = await analytics_engine.track_student_progress(user_id, 30)
            self.assertIsInstance(progress, dict)
            
            # 2. Assess concept mastery
            assessment = await mastery_detector.assess_concept_mastery(user_id, "kinematics", 14)
            self.assertIsInstance(assessment, ConceptAssessment)
            
            # 3. Generate learning path
            objective = LearningObjective(target_concepts=["forces"], difficulty_preference="adaptive")
            path = await path_optimizer.generate_learning_path(user_id, objective)
            self.assertIsInstance(path, LearningPath)
            
            # 4. Generate insights
            insights = await data_miner.generate_educational_insights(30)
            self.assertIsInstance(insights, list)
            
            logger.info("✅ Full analytics pipeline integration test passed")
            
        except Exception as e:
            self.fail(f"Full analytics pipeline integration test failed: {e}")
    
    async def test_analytics_with_realtime_events(self):
        """Test analytics integration with real-time events"""
        try:
            # Initialize components
            analytics_engine = LearningAnalyticsEngine(self.mock_db)
            realtime_engine = RealTimeAnalyticsEngine(self.mock_db)
            
            await analytics_engine.initialize()
            
            # Start real-time engine
            start_task = asyncio.create_task(realtime_engine.start())
            await asyncio.sleep(0.1)
            
            # Submit real-time events
            from realtime_analytics import create_interaction_event, create_mastery_event
            
            # Submit interaction event
            interaction_event = create_interaction_event("test_user_1", "kinematics", True, 15000)
            await realtime_engine.submit_event(interaction_event)
            
            # Submit mastery event
            mastery_event = create_mastery_event("test_user_1", "kinematics", 0.75)
            await realtime_engine.submit_event(mastery_event)
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify events were processed
            stats = realtime_engine.get_processing_stats()
            self.assertGreaterEqual(stats['events_processed'], 2)
            
            # Cleanup
            await realtime_engine.stop()
            start_task.cancel()
            
            logger.info("✅ Analytics with real-time events integration test passed")
            
        except Exception as e:
            self.fail(f"Analytics with real-time events integration test failed: {e}")

class DataValidationTests(unittest.TestCase):
    """Data validation and accuracy tests"""
    
    def test_mastery_score_validation(self):
        """Test mastery score calculation accuracy"""
        try:
            from concept_mastery import ConceptMasteryDetector, MasteryEvidence
            
            detector = ConceptMasteryDetector()
            
            # Create test evidence
            evidence = [
                MasteryEvidence("1", datetime.now(), True, 15.0, 1.0, {}, 1.0),
                MasteryEvidence("2", datetime.now(), True, 12.0, 1.0, {}, 1.0),
                MasteryEvidence("3", datetime.now(), False, 25.0, 1.5, {}, 1.0),
                MasteryEvidence("4", datetime.now(), True, 18.0, 1.2, {}, 1.0)
            ]
            
            # Calculate mastery score
            mastery_score, confidence_interval = detector._calculate_mastery_score(evidence)
            
            # Validate score
            self.assertGreaterEqual(mastery_score, 0.0)
            self.assertLessEqual(mastery_score, 1.0)
            self.assertIsInstance(confidence_interval, tuple)
            self.assertEqual(len(confidence_interval), 2)
            
            # Expected score should be 0.75 (3 successes out of 4)
            self.assertAlmostEqual(mastery_score, 0.75, places=1)
            
            logger.info("✅ Mastery score validation test passed")
            
        except Exception as e:
            self.fail(f"Mastery score validation test failed: {e}")
    
    def test_learning_trajectory_calculation(self):
        """Test learning trajectory calculation accuracy"""
        try:
            from concept_mastery import ConceptMasteryDetector, MasteryEvidence
            
            detector = ConceptMasteryDetector()
            
            # Create evidence with improving trend
            evidence = []
            for i in range(10):
                success = i >= 5  # First 5 fail, last 5 succeed
                evidence.append(MasteryEvidence(
                    str(i), 
                    datetime.now() - timedelta(days=9-i), 
                    success, 
                    15.0, 
                    1.0, 
                    {}, 
                    1.0
                ))
            
            # Calculate trajectory
            trajectory = detector._calculate_learning_trajectory(evidence)
            
            # Validate trajectory
            self.assertIsInstance(trajectory, list)
            self.assertGreater(len(trajectory), 0)
            
            # Should show improvement trend
            if len(trajectory) > 1:
                self.assertGreater(trajectory[-1], trajectory[0])
            
            logger.info("✅ Learning trajectory calculation test passed")
            
        except Exception as e:
            self.fail(f"Learning trajectory calculation test failed: {e}")

# Test runner and utilities
class AnalyticsTestRunner:
    """Main test runner for analytics components"""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
    
    async def run_all_tests(self):
        """Run all analytics tests"""
        logger.info("🧪 Starting Comprehensive Analytics Test Suite")
        
        test_classes = [
            TestLearningAnalyticsEngine,
            TestConceptMasteryDetector,
            TestLearningPathOptimizer,
            TestEducationalDataMiner,
            TestRealTimeAnalytics,
            PerformanceTests,
            IntegrationTests,
            DataValidationTests
        ]
        
        total_tests = 0
        passed_tests = 0
        
        for test_class in test_classes:
            logger.info(f"🔍 Running {test_class.__name__}")
            
            # Get test methods
            test_methods = [
                method for method in dir(test_class) 
                if method.startswith('test_')
            ]
            
            for test_method in test_methods:
                total_tests += 1
                
                try:
                    # Create test instance
                    test_instance = test_class()
                    test_instance.setUp()
                    
                    # Get the test method
                    method = getattr(test_instance, test_method)
                    
                    # Run test (handle both sync and async)
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        method()
                    
                    passed_tests += 1
                    logger.info(f"  ✅ {test_method}")
                    
                except Exception as e:
                    self.failed_tests.append(f"{test_class.__name__}.{test_method}: {str(e)}")
                    logger.error(f"  ❌ {test_method}: {str(e)}")
        
        # Print test summary
        logger.info(f"\n📊 Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {len(self.failed_tests)}")
        logger.info(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            logger.info(f"\n❌ Failed Tests:")
            for failed_test in self.failed_tests:
                logger.info(f"   - {failed_test}")
        
        return passed_tests, len(self.failed_tests)

# Main execution
async def main():
    """Main test execution function"""
    try:
        test_runner = AnalyticsTestRunner()
        passed, failed = await test_runner.run_all_tests()
        
        if failed == 0:
            logger.info("🎉 All analytics tests passed successfully!")
            return True
        else:
            logger.error(f"😞 {failed} tests failed. Please review and fix issues.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    exit(0 if success else 1)