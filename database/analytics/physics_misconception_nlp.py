#!/usr/bin/env python3
"""
Advanced NLP for Physics Misconception Detection - Phase 6
Analyzes student explanations, questions, and responses to identify
common physics misconceptions and provide targeted interventions.
"""

import asyncio
import json
import logging
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import pickle
import warnings

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MisconceptionCategory(Enum):
    FORCE_MOTION = "force_and_motion"
    ENERGY_WORK = "energy_and_work"
    MOMENTUM_COLLISIONS = "momentum_and_collisions"
    WAVES_OSCILLATIONS = "waves_and_oscillations"
    ELECTRICITY_MAGNETISM = "electricity_and_magnetism"
    THERMODYNAMICS = "thermodynamics"
    QUANTUM_MECHANICS = "quantum_mechanics"
    MATHEMATICAL_REASONING = "mathematical_reasoning"

class MisconceptionSeverity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"

@dataclass
class PhysicsMisconception:
    """Identified physics misconception"""
    misconception_id: str
    category: MisconceptionCategory
    severity: MisconceptionSeverity
    description: str
    student_text: str
    confidence_score: float
    
    # Educational context
    correct_concept: str
    explanation: str
    intervention_strategies: List[str]
    related_topics: List[str]
    
    # Evidence
    linguistic_indicators: List[str]
    reasoning_errors: List[str]
    concept_confusion: Dict[str, str]
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"

@dataclass
class StudentResponse:
    """Student response for analysis"""
    response_id: str
    student_id: str
    question_context: str
    student_answer: str
    response_type: str  # "explanation", "solution", "question", "discussion"
    topic: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MisconceptionPattern:
    """Pattern definition for misconception detection"""
    pattern_id: str
    category: MisconceptionCategory
    name: str
    description: str
    
    # Detection patterns
    keyword_patterns: List[str]
    phrase_patterns: List[str]
    semantic_patterns: List[str]
    reasoning_patterns: List[str]
    
    # Context
    typical_contexts: List[str]
    severity_indicators: Dict[str, MisconceptionSeverity]
    
    # Intervention
    intervention_templates: List[str]
    prerequisite_concepts: List[str]

class PhysicsConceptExtractor:
    """Extract physics concepts and relationships from text"""
    
    def __init__(self):
        self.physics_vocabulary = {}
        self.concept_relationships = {}
        self.unit_patterns = {}
        self.equation_patterns = {}
        
        # Initialize NLP models
        self.nlp = None
        self.tokenizer = None
        self.embeddings_model = None
    
    async def initialize(self):
        """Initialize physics concept extraction models"""
        try:
            logger.info("🚀 Initializing Physics Concept Extractor")
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("⚠️ spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Initialize physics vocabulary
            await self._initialize_physics_vocabulary()
            
            # Initialize concept relationships
            await self._initialize_concept_relationships()
            
            # Initialize pattern matching
            await self._initialize_patterns()
            
            # Load sentence transformer for semantic analysis
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"⚠️ Could not load sentence transformer: {e}")
                self.embeddings_model = None
            
            logger.info("✅ Physics Concept Extractor initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Physics Concept Extractor: {e}")
    
    async def _initialize_physics_vocabulary(self):
        """Initialize physics-specific vocabulary"""
        try:
            self.physics_vocabulary = {
                'mechanics': {
                    'force': ['force', 'forces', 'push', 'pull', 'newton', 'weight', 'tension', 'friction'],
                    'motion': ['velocity', 'acceleration', 'speed', 'displacement', 'distance', 'momentum'],
                    'energy': ['kinetic', 'potential', 'work', 'energy', 'joule', 'power', 'conservative'],
                    'mass': ['mass', 'weight', 'inertia', 'kilogram', 'matter'],
                    'time': ['time', 'second', 'duration', 'interval', 'period', 'frequency']
                },
                'waves': {
                    'wave_properties': ['amplitude', 'wavelength', 'frequency', 'period', 'phase'],
                    'wave_types': ['transverse', 'longitudinal', 'standing', 'traveling'],
                    'sound': ['sound', 'acoustic', 'decibel', 'pitch', 'resonance'],
                    'light': ['light', 'electromagnetic', 'photon', 'refraction', 'reflection']
                },
                'electricity': {
                    'current': ['current', 'ampere', 'flow', 'charge', 'electron'],
                    'voltage': ['voltage', 'potential', 'volt', 'electric field'],
                    'resistance': ['resistance', 'ohm', 'resistor', 'impedance'],
                    'power': ['electrical power', 'watt', 'energy consumption']
                }
            }
            
            logger.info("📚 Physics vocabulary initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vocabulary: {e}")
    
    async def _initialize_concept_relationships(self):
        """Initialize relationships between physics concepts"""
        try:
            self.concept_relationships = {
                'force_motion': {
                    'causes': ['force causes acceleration', 'unbalanced force causes motion'],
                    'proportional': ['force proportional to acceleration', 'acceleration inversely proportional to mass'],
                    'conserved': ['momentum conserved in isolated systems'],
                    'dependent': ['weight depends on mass and gravity']
                },
                'energy': {
                    'conserved': ['energy is conserved', 'mechanical energy constant without friction'],
                    'transforms': ['potential to kinetic energy', 'work transfers energy'],
                    'proportional': ['kinetic energy proportional to velocity squared']
                },
                'waves': {
                    'relationships': ['wave speed equals frequency times wavelength'],
                    'behaviors': ['waves reflect and refract', 'waves interfere constructively and destructively']
                }
            }
            
            logger.info("🔗 Concept relationships initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize relationships: {e}")
    
    async def _initialize_patterns(self):
        """Initialize pattern matching for units and equations"""
        try:
            # Unit patterns
            self.unit_patterns = {
                'force': r'\b(?:N|newton|newtons|pounds?|lbs?)\b',
                'mass': r'\b(?:kg|kilogram|grams?|g|pounds?|lbs?)\b',
                'distance': r'\b(?:m|meter|meters|km|kilometer|feet|ft|inches?|in)\b',
                'time': r'\b(?:s|sec|second|seconds|min|minute|minutes|hr|hour|hours)\b',
                'velocity': r'\b(?:m/s|mph|km/h|ft/s)\b',
                'acceleration': r'\b(?:m/s²|m/s2|ft/s²|ft/s2)\b'
            }
            
            # Common equation patterns
            self.equation_patterns = {
                'newtons_second_law': r'F\s*=\s*ma|force\s*=\s*mass\s*\*\s*acceleration',
                'kinematic': r'v\s*=\s*u\s*\+\s*at|s\s*=\s*ut\s*\+\s*½at²',
                'energy': r'KE\s*=\s*½mv²|PE\s*=\s*mgh',
                'momentum': r'p\s*=\s*mv|momentum\s*=\s*mass\s*\*\s*velocity'
            }
            
            logger.info("📐 Patterns initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize patterns: {e}")
    
    async def extract_concepts(self, text: str) -> Dict[str, Any]:
        """Extract physics concepts from text"""
        try:
            extracted_concepts = {
                'physics_terms': [],
                'equations': [],
                'units': [],
                'relationships': [],
                'concept_density': 0.0
            }
            
            text_lower = text.lower()
            
            # Extract physics terms
            for category, subcategories in self.physics_vocabulary.items():
                for subcategory, terms in subcategories.items():
                    for term in terms:
                        if term in text_lower:
                            extracted_concepts['physics_terms'].append({
                                'term': term,
                                'category': category,
                                'subcategory': subcategory
                            })
            
            # Extract units
            for unit_type, pattern in self.unit_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    extracted_concepts['units'].append({
                        'unit': match,
                        'type': unit_type
                    })
            
            # Extract equations
            for equation_type, pattern in self.equation_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    extracted_concepts['equations'].append(equation_type)
            
            # Calculate concept density
            word_count = len(text.split())
            concept_count = len(extracted_concepts['physics_terms'])
            extracted_concepts['concept_density'] = concept_count / max(1, word_count)
            
            return extracted_concepts
            
        except Exception as e:
            logger.error(f"❌ Failed to extract concepts: {e}")
            return {'physics_terms': [], 'equations': [], 'units': [], 'relationships': [], 'concept_density': 0.0}

class MisconceptionDetector:
    """Core misconception detection engine"""
    
    def __init__(self):
        self.misconception_patterns = []
        self.trained_models = {}
        self.concept_extractor = PhysicsConceptExtractor()
        
        # NLP components
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.embeddings_model = None
        
        # Pattern-based detection
        self.keyword_detector = None
        self.semantic_detector = None
        
        # Training data
        self.training_examples = []
        self.misconception_examples = {}
    
    async def initialize(self):
        """Initialize misconception detection models"""
        try:
            logger.info("🚀 Initializing Misconception Detector")
            
            # Initialize concept extractor
            await self.concept_extractor.initialize()
            
            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize sentiment analyzer: {e}")
            
            # Initialize embeddings model
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"⚠️ Could not load embeddings model: {e}")
            
            # Load misconception patterns
            await self._load_misconception_patterns()
            
            # Initialize detection models
            await self._initialize_detection_models()
            
            logger.info("✅ Misconception Detector initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Misconception Detector: {e}")
    
    async def _load_misconception_patterns(self):
        """Load predefined misconception patterns"""
        try:
            # Common physics misconceptions
            patterns = [
                MisconceptionPattern(
                    pattern_id="force_motion_01",
                    category=MisconceptionCategory.FORCE_MOTION,
                    name="Force Required for Motion",
                    description="Belief that force is required to maintain motion",
                    keyword_patterns=[
                        "force needed to keep moving",
                        "force required for motion",
                        "object stops without force",
                        "need force to maintain velocity"
                    ],
                    phrase_patterns=[
                        "objects naturally slow down",
                        "force keeps things moving",
                        "motion requires continuous force"
                    ],
                    semantic_patterns=[
                        "motion cessation without force",
                        "continuous force for constant velocity"
                    ],
                    reasoning_patterns=[
                        "conflating force with velocity",
                        "ignoring Newton's first law"
                    ],
                    typical_contexts=["friction problems", "space motion", "constant velocity"],
                    severity_indicators={
                        "always": MisconceptionSeverity.MAJOR,
                        "usually": MisconceptionSeverity.MODERATE,
                        "sometimes": MisconceptionSeverity.MINOR
                    },
                    intervention_templates=[
                        "Consider Newton's first law: objects in motion stay in motion unless acted upon by a force",
                        "Think about what happens to objects in space where there's no friction",
                        "Remember that force causes acceleration, not velocity"
                    ],
                    prerequisite_concepts=["Newton's first law", "inertia", "friction"]
                ),
                
                MisconceptionPattern(
                    pattern_id="energy_01",
                    category=MisconceptionCategory.ENERGY_WORK,
                    name="Energy as Material Substance",
                    description="Treating energy as a physical substance that can be used up",
                    keyword_patterns=[
                        "energy gets used up",
                        "energy is consumed",
                        "running out of energy",
                        "energy disappears"
                    ],
                    phrase_patterns=[
                        "energy is lost",
                        "energy gets destroyed",
                        "using up all the energy"
                    ],
                    semantic_patterns=[
                        "energy depletion",
                        "energy consumption"
                    ],
                    reasoning_patterns=[
                        "violating conservation of energy",
                        "treating energy as matter"
                    ],
                    typical_contexts=["energy transformations", "conservation problems"],
                    severity_indicators={
                        "destroyed": MisconceptionSeverity.MAJOR,
                        "lost": MisconceptionSeverity.MODERATE,
                        "used": MisconceptionSeverity.MINOR
                    },
                    intervention_templates=[
                        "Energy is conserved - it transforms from one type to another",
                        "Think about where the energy goes rather than being 'used up'",
                        "Consider the energy transformations in the system"
                    ],
                    prerequisite_concepts=["conservation of energy", "energy transformations"]
                ),
                
                MisconceptionPattern(
                    pattern_id="momentum_01",
                    category=MisconceptionCategory.MOMENTUM_COLLISIONS,
                    name="Momentum Confusion with Force",
                    description="Confusing momentum with force or impact",
                    keyword_patterns=[
                        "momentum is force",
                        "more momentum means more force",
                        "momentum creates force",
                        "momentum is impact"
                    ],
                    phrase_patterns=[
                        "momentum equals force",
                        "momentum is the force of impact",
                        "greater momentum, greater force"
                    ],
                    semantic_patterns=[
                        "momentum force equivalence",
                        "momentum impact confusion"
                    ],
                    reasoning_patterns=[
                        "conflating momentum and force",
                        "misunderstanding impulse"
                    ],
                    typical_contexts=["collision problems", "impulse calculations"],
                    severity_indicators={
                        "same as": MisconceptionSeverity.MAJOR,
                        "related to": MisconceptionSeverity.MODERATE,
                        "like": MisconceptionSeverity.MINOR
                    },
                    intervention_templates=[
                        "Momentum is mass times velocity, while force causes changes in momentum",
                        "Think about the impulse-momentum theorem: force times time equals change in momentum",
                        "Consider how force and momentum are related but distinct concepts"
                    ],
                    prerequisite_concepts=["momentum definition", "impulse-momentum theorem", "Newton's second law"]
                )
            ]
            
            self.misconception_patterns = patterns
            logger.info(f"📋 Loaded {len(patterns)} misconception patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to load misconception patterns: {e}")
    
    async def _initialize_detection_models(self):
        """Initialize various detection models"""
        try:
            # Keyword-based detector
            self.keyword_detector = KeywordMisconceptionDetector()
            await self.keyword_detector.initialize(self.misconception_patterns)
            
            # Semantic similarity detector
            self.semantic_detector = SemanticMisconceptionDetector()
            if self.embeddings_model:
                await self.semantic_detector.initialize(self.embeddings_model, self.misconception_patterns)
            
            logger.info("🔧 Detection models initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize detection models: {e}")
    
    async def analyze_response(self, response: StudentResponse) -> List[PhysicsMisconception]:
        """Analyze student response for misconceptions"""
        try:
            logger.info(f"🔍 Analyzing response: {response.response_id}")
            
            detected_misconceptions = []
            
            # Extract physics concepts
            concepts = await self.concept_extractor.extract_concepts(response.student_answer)
            
            # Analyze sentiment and confidence
            sentiment_analysis = await self._analyze_sentiment(response.student_answer)
            
            # Pattern-based detection
            if self.keyword_detector:
                keyword_detections = await self.keyword_detector.detect(response.student_answer)
                detected_misconceptions.extend(keyword_detections)
            
            # Semantic detection
            if self.semantic_detector:
                semantic_detections = await self.semantic_detector.detect(response.student_answer)
                detected_misconceptions.extend(semantic_detections)
            
            # Reasoning pattern detection
            reasoning_detections = await self._detect_reasoning_patterns(response.student_answer, concepts)
            detected_misconceptions.extend(reasoning_detections)
            
            # Post-process and rank detections
            final_misconceptions = await self._post_process_detections(
                detected_misconceptions, response, concepts, sentiment_analysis
            )
            
            logger.info(f"✅ Detected {len(final_misconceptions)} misconceptions")
            return final_misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze response: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment and confidence indicators in text"""
        try:
            if not self.sentiment_analyzer:
                return {'confidence': 0.5, 'uncertainty': 0.5}
            
            # Get sentiment scores
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Detect uncertainty indicators
            uncertainty_words = ['maybe', 'perhaps', 'might', 'could', 'think', 'guess', 'probably', 'not sure']
            confidence_words = ['definitely', 'certainly', 'always', 'never', 'absolutely', 'sure', 'know']
            
            text_lower = text.lower()
            uncertainty_count = sum(1 for word in uncertainty_words if word in text_lower)
            confidence_count = sum(1 for word in confidence_words if word in text_lower)
            
            # Calculate confidence score
            word_count = len(text.split())
            uncertainty_ratio = uncertainty_count / max(1, word_count)
            confidence_ratio = confidence_count / max(1, word_count)
            
            confidence_score = max(0.1, min(0.9, 0.5 + confidence_ratio - uncertainty_ratio))
            
            return {
                'sentiment_compound': scores['compound'],
                'confidence': confidence_score,
                'uncertainty': uncertainty_ratio,
                'certainty_indicators': confidence_count,
                'uncertainty_indicators': uncertainty_count
            }
            
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}")
            return {'confidence': 0.5, 'uncertainty': 0.5}
    
    async def _detect_reasoning_patterns(self, text: str, concepts: Dict[str, Any]) -> List[PhysicsMisconception]:
        """Detect misconceptions based on reasoning patterns"""
        try:
            detections = []
            text_lower = text.lower()
            
            # Check for common reasoning errors
            reasoning_errors = []
            
            # Causal reasoning errors
            if any(phrase in text_lower for phrase in ['because of', 'due to', 'caused by']):
                # Check for incorrect causal relationships
                if 'force' in text_lower and 'velocity' in text_lower:
                    if any(phrase in text_lower for phrase in ['force causes velocity', 'velocity needs force']):
                        reasoning_errors.append('incorrect_force_velocity_causation')
            
            # Proportionality errors
            proportional_indicators = ['proportional', 'increases with', 'decreases with', 'depends on']
            if any(indicator in text_lower for indicator in proportional_indicators):
                # Check for incorrect relationships
                if 'kinetic energy' in text_lower and 'velocity' in text_lower:
                    if 'proportional to velocity' in text_lower and 'squared' not in text_lower:
                        reasoning_errors.append('incorrect_ke_velocity_relationship')
            
            # Conservation violations
            conservation_indicators = ['conserved', 'constant', 'stays the same']
            violation_indicators = ['lost', 'gained', 'created', 'destroyed']
            
            if any(indicator in text_lower for indicator in conservation_indicators):
                if any(violation in text_lower for violation in violation_indicators):
                    reasoning_errors.append('conservation_violation')
            
            # Create misconception objects for detected errors
            for error in reasoning_errors:
                misconception = await self._create_reasoning_misconception(error, text)
                if misconception:
                    detections.append(misconception)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Reasoning pattern detection failed: {e}")
            return []
    
    async def _create_reasoning_misconception(self, error_type: str, text: str) -> Optional[PhysicsMisconception]:
        """Create misconception object for reasoning errors"""
        try:
            error_definitions = {
                'incorrect_force_velocity_causation': {
                    'category': MisconceptionCategory.FORCE_MOTION,
                    'severity': MisconceptionSeverity.MODERATE,
                    'description': 'Incorrectly linking force directly to velocity rather than acceleration',
                    'correct_concept': 'Force causes acceleration, not velocity',
                    'explanation': 'Force is related to changes in motion (acceleration), not the motion itself (velocity)',
                    'interventions': ['Review Newton\'s second law', 'Practice with force and acceleration problems']
                },
                'incorrect_ke_velocity_relationship': {
                    'category': MisconceptionCategory.ENERGY_WORK,
                    'severity': MisconceptionSeverity.MODERATE,
                    'description': 'Thinking kinetic energy is proportional to velocity instead of velocity squared',
                    'correct_concept': 'Kinetic energy is proportional to velocity squared (KE = ½mv²)',
                    'explanation': 'Doubling velocity quadruples kinetic energy, not doubles it',
                    'interventions': ['Review kinetic energy formula', 'Practice energy calculations']
                },
                'conservation_violation': {
                    'category': MisconceptionCategory.ENERGY_WORK,
                    'severity': MisconceptionSeverity.MAJOR,
                    'description': 'Violating conservation laws by claiming energy is lost or created',
                    'correct_concept': 'Energy is conserved - it transforms but is never created or destroyed',
                    'explanation': 'Energy may change forms but the total amount remains constant in isolated systems',
                    'interventions': ['Review conservation of energy', 'Practice energy transformation problems']
                }
            }
            
            if error_type not in error_definitions:
                return None
            
            error_def = error_definitions[error_type]
            
            return PhysicsMisconception(
                misconception_id=f"reasoning_{error_type}_{datetime.now().timestamp()}",
                category=error_def['category'],
                severity=error_def['severity'],
                description=error_def['description'],
                student_text=text[:200] + "..." if len(text) > 200 else text,
                confidence_score=0.7,
                correct_concept=error_def['correct_concept'],
                explanation=error_def['explanation'],
                intervention_strategies=error_def['interventions'],
                related_topics=[error_def['category'].value],
                linguistic_indicators=[],
                reasoning_errors=[error_type],
                concept_confusion={}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create reasoning misconception: {e}")
            return None
    
    async def _post_process_detections(self, detections: List[PhysicsMisconception],
                                     response: StudentResponse,
                                     concepts: Dict[str, Any],
                                     sentiment: Dict[str, float]) -> List[PhysicsMisconception]:
        """Post-process and rank detected misconceptions"""
        try:
            if not detections:
                return []
            
            # Remove duplicates
            unique_detections = []
            seen_descriptions = set()
            
            for detection in detections:
                if detection.description not in seen_descriptions:
                    unique_detections.append(detection)
                    seen_descriptions.add(detection.description)
            
            # Adjust confidence based on context
            for detection in unique_detections:
                # Adjust for student confidence
                student_confidence = sentiment.get('confidence', 0.5)
                if student_confidence > 0.7:
                    detection.confidence_score *= 1.2  # More confident students with misconceptions
                elif student_confidence < 0.3:
                    detection.confidence_score *= 0.8  # Less confident might be uncertain
                
                # Adjust for concept density
                concept_density = concepts.get('concept_density', 0.0)
                if concept_density > 0.1:
                    detection.confidence_score *= 1.1  # More physics terms = more reliable detection
                
                # Ensure confidence stays in bounds
                detection.confidence_score = max(0.1, min(0.95, detection.confidence_score))
            
            # Sort by confidence score
            unique_detections.sort(key=lambda x: x.confidence_score, reverse=True)
            
            # Return top 3 most confident detections
            return unique_detections[:3]
            
        except Exception as e:
            logger.error(f"❌ Post-processing failed: {e}")
            return detections[:3]  # Fallback to original list

class KeywordMisconceptionDetector:
    """Keyword-based misconception detection"""
    
    def __init__(self):
        self.pattern_keywords = {}
    
    async def initialize(self, patterns: List[MisconceptionPattern]):
        """Initialize keyword patterns"""
        try:
            for pattern in patterns:
                self.pattern_keywords[pattern.pattern_id] = {
                    'keywords': pattern.keyword_patterns,
                    'phrases': pattern.phrase_patterns,
                    'pattern': pattern
                }
            
            logger.info("🔑 Keyword detector initialized")
            
        except Exception as e:
            logger.error(f"❌ Keyword detector initialization failed: {e}")
    
    async def detect(self, text: str) -> List[PhysicsMisconception]:
        """Detect misconceptions using keyword matching"""
        try:
            detections = []
            text_lower = text.lower()
            
            for pattern_id, pattern_data in self.pattern_keywords.items():
                pattern = pattern_data['pattern']
                match_count = 0
                matched_keywords = []
                
                # Check keyword patterns
                for keyword in pattern_data['keywords']:
                    if keyword.lower() in text_lower:
                        match_count += 1
                        matched_keywords.append(keyword)
                
                # Check phrase patterns
                for phrase in pattern_data['phrases']:
                    if phrase.lower() in text_lower:
                        match_count += 2  # Phrases weighted more heavily
                        matched_keywords.append(phrase)
                
                # Create detection if sufficient matches
                if match_count >= 1:  # At least one match required
                    confidence = min(0.9, match_count * 0.3)  # Scale confidence
                    
                    misconception = PhysicsMisconception(
                        misconception_id=f"keyword_{pattern_id}_{datetime.now().timestamp()}",
                        category=pattern.category,
                        severity=MisconceptionSeverity.MODERATE,  # Default severity
                        description=pattern.description,
                        student_text=text[:200] + "..." if len(text) > 200 else text,
                        confidence_score=confidence,
                        correct_concept=f"Correct understanding of {pattern.category.value}",
                        explanation=pattern.description,
                        intervention_strategies=pattern.intervention_templates,
                        related_topics=[pattern.category.value],
                        linguistic_indicators=matched_keywords,
                        reasoning_errors=[],
                        concept_confusion={}
                    )
                    
                    detections.append(misconception)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Keyword detection failed: {e}")
            return []

class SemanticMisconceptionDetector:
    """Semantic similarity-based misconception detection"""
    
    def __init__(self):
        self.embeddings_model = None
        self.pattern_embeddings = {}
        self.similarity_threshold = 0.7
    
    async def initialize(self, embeddings_model, patterns: List[MisconceptionPattern]):
        """Initialize semantic detector"""
        try:
            self.embeddings_model = embeddings_model
            
            # Create embeddings for misconception patterns
            for pattern in patterns:
                pattern_texts = pattern.keyword_patterns + pattern.phrase_patterns + pattern.semantic_patterns
                if pattern_texts:
                    embeddings = self.embeddings_model.encode(pattern_texts)
                    self.pattern_embeddings[pattern.pattern_id] = {
                        'embeddings': embeddings,
                        'texts': pattern_texts,
                        'pattern': pattern
                    }
            
            logger.info("🧠 Semantic detector initialized")
            
        except Exception as e:
            logger.error(f"❌ Semantic detector initialization failed: {e}")
    
    async def detect(self, text: str) -> List[PhysicsMisconception]:
        """Detect misconceptions using semantic similarity"""
        try:
            if not self.embeddings_model:
                return []
            
            detections = []
            
            # Get text embedding
            text_embedding = self.embeddings_model.encode([text])
            
            # Compare with pattern embeddings
            for pattern_id, pattern_data in self.pattern_embeddings.items():
                pattern_embeddings = pattern_data['embeddings']
                pattern = pattern_data['pattern']
                
                # Calculate similarities
                similarities = cosine_similarity(text_embedding, pattern_embeddings)[0]
                max_similarity = np.max(similarities)
                
                # Check if similarity exceeds threshold
                if max_similarity >= self.similarity_threshold:
                    best_match_idx = np.argmax(similarities)
                    matched_text = pattern_data['texts'][best_match_idx]
                    
                    misconception = PhysicsMisconception(
                        misconception_id=f"semantic_{pattern_id}_{datetime.now().timestamp()}",
                        category=pattern.category,
                        severity=MisconceptionSeverity.MODERATE,
                        description=pattern.description,
                        student_text=text[:200] + "..." if len(text) > 200 else text,
                        confidence_score=float(max_similarity),
                        correct_concept=f"Correct understanding of {pattern.category.value}",
                        explanation=pattern.description,
                        intervention_strategies=pattern.intervention_templates,
                        related_topics=[pattern.category.value],
                        linguistic_indicators=[matched_text],
                        reasoning_errors=[],
                        concept_confusion={}
                    )
                    
                    detections.append(misconception)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Semantic detection failed: {e}")
            return []

class PhysicsMisconceptionNLP:
    """Main NLP engine for physics misconception detection"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.misconception_detector = MisconceptionDetector()
        
        # Analytics
        self.detection_history = []
        self.misconception_trends = defaultdict(int)
        self.intervention_effectiveness = {}
        
        # Configuration
        self.confidence_threshold = 0.5
        self.max_detections_per_response = 3
    
    async def initialize(self):
        """Initialize the physics misconception NLP system"""
        try:
            logger.info("🚀 Initializing Physics Misconception NLP")
            
            # Initialize detector
            await self.misconception_detector.initialize()
            
            # Load historical data if available
            await self._load_historical_data()
            
            logger.info("✅ Physics Misconception NLP initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Physics Misconception NLP: {e}")
            return False
    
    async def analyze_student_response(self, student_id: str, question_context: str,
                                     student_answer: str, topic: str,
                                     response_type: str = "explanation") -> List[PhysicsMisconception]:
        """Analyze student response for misconceptions"""
        try:
            # Create response object
            response = StudentResponse(
                response_id=f"resp_{student_id}_{datetime.now().timestamp()}",
                student_id=student_id,
                question_context=question_context,
                student_answer=student_answer,
                response_type=response_type,
                topic=topic
            )
            
            # Detect misconceptions
            misconceptions = await self.misconception_detector.analyze_response(response)
            
            # Filter by confidence threshold
            filtered_misconceptions = [
                m for m in misconceptions 
                if m.confidence_score >= self.confidence_threshold
            ]
            
            # Limit number of detections
            final_misconceptions = filtered_misconceptions[:self.max_detections_per_response]
            
            # Store results
            await self._store_detection_results(response, final_misconceptions)
            
            # Update trends
            for misconception in final_misconceptions:
                self.misconception_trends[misconception.category.value] += 1
            
            logger.info(f"📊 Analyzed response for {student_id}: {len(final_misconceptions)} misconceptions detected")
            return final_misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze student response: {e}")
            return []
    
    async def get_misconception_trends(self, time_period: timedelta = None) -> Dict[str, Any]:
        """Get misconception trends and analytics"""
        try:
            if time_period is None:
                time_period = timedelta(days=30)
            
            cutoff_date = datetime.now() - time_period
            
            # Filter recent detections
            recent_detections = [
                d for d in self.detection_history 
                if d['timestamp'] >= cutoff_date
            ]
            
            # Calculate trends
            category_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            topic_counts = defaultdict(int)
            
            for detection in recent_detections:
                for misconception in detection['misconceptions']:
                    category_counts[misconception.category.value] += 1
                    severity_counts[misconception.severity.value] += 1
                    for topic in misconception.related_topics:
                        topic_counts[topic] += 1
            
            return {
                'time_period_days': time_period.days,
                'total_detections': len(recent_detections),
                'category_distribution': dict(category_counts),
                'severity_distribution': dict(severity_counts),
                'topic_distribution': dict(topic_counts),
                'most_common_misconceptions': dict(self.misconception_trends),
                'detection_rate': len(recent_detections) / max(1, time_period.days)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get misconception trends: {e}")
            return {}
    
    async def generate_intervention_recommendations(self, misconceptions: List[PhysicsMisconception],
                                                  student_context: Dict[str, Any] = None) -> List[str]:
        """Generate personalized intervention recommendations"""
        try:
            recommendations = []
            
            # Group misconceptions by category
            category_groups = defaultdict(list)
            for misconception in misconceptions:
                category_groups[misconception.category].append(misconception)
            
            # Generate category-specific recommendations
            for category, category_misconceptions in category_groups.items():
                # Get severity level
                max_severity = max(m.severity for m in category_misconceptions)
                
                # Add category-specific interventions
                for misconception in category_misconceptions:
                    recommendations.extend(misconception.intervention_strategies)
                
                # Add category-level recommendations
                if category == MisconceptionCategory.FORCE_MOTION:
                    if max_severity in [MisconceptionSeverity.MAJOR, MisconceptionSeverity.CRITICAL]:
                        recommendations.append("Review Newton's laws with interactive simulations")
                        recommendations.append("Practice force analysis with free body diagrams")
                elif category == MisconceptionCategory.ENERGY_WORK:
                    if max_severity in [MisconceptionSeverity.MAJOR, MisconceptionSeverity.CRITICAL]:
                        recommendations.append("Work through energy conservation examples step-by-step")
                        recommendations.append("Use energy bar charts to visualize transformations")
            
            # Add student-specific recommendations if context available
            if student_context:
                learning_style = student_context.get('learning_style', 'mixed')
                if learning_style == 'visual':
                    recommendations.append("Use visual aids and diagrams to reinforce concepts")
                elif learning_style == 'kinesthetic':
                    recommendations.append("Try hands-on experiments and simulations")
            
            # Remove duplicates and limit
            unique_recommendations = list(dict.fromkeys(recommendations))
            return unique_recommendations[:8]  # Limit to 8 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intervention recommendations: {e}")
            return ["Review fundamental physics concepts and practice problem-solving"]
    
    async def _load_historical_data(self):
        """Load historical misconception data"""
        try:
            # In a real implementation, this would load from database
            logger.info("📚 Historical data loading skipped (would load from database)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical data: {e}")
    
    async def _store_detection_results(self, response: StudentResponse, 
                                     misconceptions: List[PhysicsMisconception]):
        """Store detection results for analytics"""
        try:
            detection_record = {
                'response_id': response.response_id,
                'student_id': response.student_id,
                'timestamp': datetime.now(),
                'misconceptions': misconceptions,
                'response_length': len(response.student_answer),
                'topic': response.topic
            }
            
            self.detection_history.append(detection_record)
            
            # Keep only recent history (last 1000 records)
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            # Store in database if available
            if self.db_manager:
                # Would implement database storage here
                pass
            
        except Exception as e:
            logger.error(f"❌ Failed to store detection results: {e}")

# Testing function
async def test_physics_misconception_nlp():
    """Test physics misconception NLP system"""
    try:
        logger.info("🧪 Testing Physics Misconception NLP")
        
        nlp_system = PhysicsMisconceptionNLP()
        await nlp_system.initialize()
        
        # Test misconception detection
        test_responses = [
            {
                'student_id': 'test_student_1',
                'question': 'Why does a ball eventually stop rolling on the ground?',
                'answer': 'The ball stops because it runs out of force to keep it moving. Objects need continuous force to maintain motion.',
                'topic': 'mechanics'
            },
            {
                'student_id': 'test_student_2',
                'question': 'What happens to energy in a collision?',
                'answer': 'Energy gets lost during the collision and some of it disappears completely.',
                'topic': 'energy'
            }
        ]
        
        for test_response in test_responses:
            misconceptions = await nlp_system.analyze_student_response(
                test_response['student_id'],
                test_response['question'],
                test_response['answer'],
                test_response['topic']
            )
            
            logger.info(f"📊 Student {test_response['student_id']}: {len(misconceptions)} misconceptions detected")
            
            for misconception in misconceptions:
                logger.info(f"  - {misconception.category.value}: {misconception.description} (confidence: {misconception.confidence_score:.2f})")
            
            # Test intervention recommendations
            recommendations = await nlp_system.generate_intervention_recommendations(misconceptions)
            logger.info(f"💡 Generated {len(recommendations)} intervention recommendations")
        
        # Test trends analysis
        trends = await nlp_system.get_misconception_trends()
        logger.info(f"📈 Trends: {trends.get('total_detections', 0)} total detections")
        
        logger.info("✅ Physics Misconception NLP test completed")
        
    except Exception as e:
        logger.error(f"❌ Physics Misconception NLP test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_physics_misconception_nlp())