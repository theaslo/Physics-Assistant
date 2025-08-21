#!/usr/bin/env python3
"""
Analytics Package for Physics Assistant
Comprehensive learning analytics, data mining, and educational insights system.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Physics Assistant Analytics Team"

# Core analytics modules
try:
    from .learning_analytics import LearningAnalyticsEngine, StudentProfile
    from .concept_mastery import ConceptMasteryDetector, ConceptAssessment
    from .learning_path_optimizer import LearningPathOptimizer, LearningObjective
    from .educational_data_mining import EducationalDataMiner
    from .realtime_analytics import RealTimeAnalyticsEngine
    
    __all__ = [
        'LearningAnalyticsEngine',
        'StudentProfile', 
        'ConceptMasteryDetector',
        'ConceptAssessment',
        'LearningPathOptimizer',
        'LearningObjective',
        'EducationalDataMiner',
        'RealTimeAnalyticsEngine'
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    __all__ = []
    import logging
    logging.warning(f"Some analytics modules could not be imported: {e}")

# Package metadata
PACKAGE_INFO = {
    "name": "physics-assistant-analytics",
    "version": __version__,
    "description": "Advanced learning analytics and educational data mining for Physics Assistant",
    "features": [
        "Student progress tracking",
        "Concept mastery detection", 
        "Learning path optimization",
        "Educational data mining",
        "Real-time analytics processing",
        "Predictive modeling",
        "Error pattern analysis",
        "Adaptive interventions"
    ],
    "dependencies": [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0"
    ]
}