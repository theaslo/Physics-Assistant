#!/usr/bin/env python3
"""
Advanced Learning Analytics with NLP and Computer Vision for Physics Assistant Phase 6
Implements sophisticated analysis of student explanations, hand-drawn diagrams,
and multimodal learning interactions using state-of-the-art AI techniques.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import pickle
import warnings
import base64
import io

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class AnalysisType(Enum):
    MISCONCEPTION_DETECTION = "misconception_detection"
    EXPLANATION_QUALITY = "explanation_quality"
    DIAGRAM_ANALYSIS = "diagram_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CONCEPT_EXTRACTION = "concept_extraction"
    LEARNING_PROGRESSION = "learning_progression"
    ENGAGEMENT_ANALYSIS = "engagement_analysis"

class MisconceptionCategory(Enum):
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    MATHEMATICAL = "mathematical"
    UNIT_CONVERSION = "unit_conversion"
    VECTOR_OPERATIONS = "vector_operations"
    GRAPH_INTERPRETATION = "graph_interpretation"

@dataclass
class MisconceptionDetection:
    """Detected misconception with details"""
    misconception_id: str
    category: MisconceptionCategory
    description: str
    confidence: float
    evidence_text: str
    correct_concept: str
    remediation_suggestions: List[str]
    severity: str  # 'low', 'medium', 'high'
    frequency: int  # How often this misconception appears

@dataclass
class ExplanationAnalysis:
    """Analysis of student explanation quality"""
    explanation_id: str
    student_id: str
    physics_concept: str
    explanation_text: str
    quality_score: float  # 0-1 scale
    completeness_score: float
    accuracy_score: float
    clarity_score: float
    conceptual_understanding: float
    identified_misconceptions: List[MisconceptionDetection]
    key_concepts_mentioned: List[str]
    missing_concepts: List[str]
    language_complexity: str  # 'basic', 'intermediate', 'advanced'
    sentiment_score: float
    confidence_indicators: List[str]
    uncertainty_indicators: List[str]

@dataclass
class DiagramAnalysis:
    """Analysis of hand-drawn physics diagrams"""
    diagram_id: str
    student_id: str
    image_data: str  # Base64 encoded image
    detected_objects: List[Dict[str, Any]]
    diagram_type: str  # 'force_diagram', 'circuit', 'motion_graph', etc.
    completeness_score: float
    accuracy_score: float
    labeling_quality: float
    detected_errors: List[str]
    missing_elements: List[str]
    physics_principles_shown: List[str]
    extracted_text: str
    mathematical_expressions: List[str]

@dataclass
class LearningProgressionAnalysis:
    """Analysis of learning progression over time"""
    student_id: str
    concept: str
    time_period: str
    progression_score: float  # Overall progression rate
    mastery_trajectory: List[Tuple[datetime, float]]  # Time, mastery level
    explanation_evolution: List[ExplanationAnalysis]
    misconception_resolution: Dict[str, bool]  # Misconception ID -> resolved
    conceptual_breakthroughs: List[datetime]
    learning_plateau_periods: List[Tuple[datetime, datetime]]
    vocabulary_development: Dict[str, float]  # Term -> usage frequency over time

class PhysicsNLPProcessor:
    """NLP processor specialized for physics education"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sentiment_analyzer = None
        self.misconception_detector = None
        self.physics_vocabulary = set()
        self.concept_patterns = {}
        self.misconception_patterns = {}
        
        # Physics-specific stop words and terms
        self.physics_stop_words = set(['force', 'motion', 'energy', 'velocity', 'acceleration'])
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic NLP features")
            self.nlp = None
    
    async def initialize(self):
        """Initialize NLP models and resources"""
        try:
            logger.info("🚀 Initializing Physics NLP Processor")
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize transformers for physics text analysis
            model_name = "distilbert-base-uncased"
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            
            # Load physics vocabulary
            await self._load_physics_vocabulary()
            
            # Initialize misconception patterns
            await self._initialize_misconception_patterns()
            
            # Initialize concept extraction patterns
            await self._initialize_concept_patterns()
            
            logger.info("✅ Physics NLP Processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Physics NLP Processor: {e}")
            return False
    
    async def _load_physics_vocabulary(self):
        """Load physics-specific vocabulary and terminology"""
        try:
            # Physics vocabulary by topic
            self.physics_vocabulary = {
                'kinematics': [
                    'position', 'displacement', 'velocity', 'acceleration', 'time',
                    'distance', 'speed', 'motion', 'trajectory', 'kinematic equations'
                ],
                'forces': [
                    'force', 'newton', 'friction', 'tension', 'normal force', 'weight',
                    'gravity', 'equilibrium', 'net force', 'free body diagram'
                ],
                'energy': [
                    'energy', 'kinetic energy', 'potential energy', 'work', 'power',
                    'conservation', 'joule', 'mechanical energy', 'thermal energy'
                ],
                'momentum': [
                    'momentum', 'impulse', 'collision', 'conservation of momentum',
                    'elastic', 'inelastic', 'center of mass'
                ],
                'waves': [
                    'wave', 'frequency', 'wavelength', 'amplitude', 'period',
                    'interference', 'diffraction', 'reflection', 'refraction'
                ]
            }
            
            # Flatten vocabulary
            all_terms = set()
            for terms in self.physics_vocabulary.values():
                all_terms.update(terms)
            self.physics_vocabulary['all'] = list(all_terms)
            
        except Exception as e:
            logger.error(f"❌ Failed to load physics vocabulary: {e}")
    
    async def _initialize_misconception_patterns(self):
        """Initialize patterns for detecting common physics misconceptions"""
        try:
            self.misconception_patterns = {
                'force_motion_misconception': {
                    'patterns': [
                        r'force.*cause.*motion',
                        r'no force.*no motion',
                        r'force.*direction.*motion',
                        r'heavier.*fall.*faster'
                    ],
                    'category': MisconceptionCategory.CONCEPTUAL,
                    'description': 'Confusion about relationship between force and motion',
                    'correct_concept': 'Force causes acceleration, not motion. Objects in motion can have zero net force.'
                },
                'energy_misconception': {
                    'patterns': [
                        r'energy.*used up',
                        r'energy.*consumed',
                        r'energy.*disappear'
                    ],
                    'category': MisconceptionCategory.CONCEPTUAL,
                    'description': 'Misunderstanding of energy conservation',
                    'correct_concept': 'Energy is conserved; it transforms from one form to another.'
                },
                'vector_addition_error': {
                    'patterns': [
                        r'add.*vectors.*like.*numbers',
                        r'magnitude.*sum.*individual.*magnitudes'
                    ],
                    'category': MisconceptionCategory.MATHEMATICAL,
                    'description': 'Incorrect vector addition',
                    'correct_concept': 'Vectors must be added considering both magnitude and direction.'
                },
                'free_fall_misconception': {
                    'patterns': [
                        r'heavier.*objects.*fall.*faster',
                        r'weight.*affects.*falling.*speed'
                    ],
                    'category': MisconceptionCategory.CONCEPTUAL,
                    'description': 'Misconception about free fall',
                    'correct_concept': 'In vacuum, all objects fall at the same rate regardless of weight.'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize misconception patterns: {e}")
    
    async def _initialize_concept_patterns(self):
        """Initialize patterns for extracting physics concepts from text"""
        try:
            self.concept_patterns = {
                'newton_laws': [
                    r'newton.*first.*law', r'law.*inertia', r'object.*rest.*motion',
                    r'newton.*second.*law', r'f.*=.*ma', r'force.*mass.*acceleration',
                    r'newton.*third.*law', r'action.*reaction', r'equal.*opposite'
                ],
                'energy_conservation': [
                    r'conservation.*energy', r'energy.*conserved', r'total.*energy.*constant',
                    r'kinetic.*potential.*energy', r'mechanical.*energy'
                ],
                'momentum_conservation': [
                    r'conservation.*momentum', r'momentum.*conserved', r'total.*momentum.*constant'
                ],
                'wave_properties': [
                    r'wave.*length', r'frequency', r'amplitude', r'period',
                    r'wave.*speed', r'interference', r'superposition'
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize concept patterns: {e}")
    
    async def analyze_explanation(self, student_id: str, explanation_text: str,
                                physics_concept: str) -> ExplanationAnalysis:
        """Analyze student physics explanation for quality and understanding"""
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(explanation_text)
            
            # Detect misconceptions
            misconceptions = await self._detect_misconceptions(cleaned_text)
            
            # Calculate quality scores
            completeness_score = await self._calculate_completeness_score(cleaned_text, physics_concept)
            accuracy_score = await self._calculate_accuracy_score(cleaned_text, misconceptions)
            clarity_score = await self._calculate_clarity_score(cleaned_text)
            conceptual_understanding = await self._assess_conceptual_understanding(cleaned_text, physics_concept)
            
            # Overall quality score
            quality_score = (completeness_score + accuracy_score + clarity_score + conceptual_understanding) / 4.0
            
            # Extract key concepts mentioned
            key_concepts = await self._extract_mentioned_concepts(cleaned_text)
            missing_concepts = await self._identify_missing_concepts(key_concepts, physics_concept)
            
            # Analyze language complexity
            language_complexity = self._assess_language_complexity(cleaned_text)
            
            # Sentiment analysis
            sentiment_score = self._analyze_sentiment(cleaned_text)
            
            # Confidence and uncertainty indicators
            confidence_indicators = self._detect_confidence_indicators(cleaned_text)
            uncertainty_indicators = self._detect_uncertainty_indicators(cleaned_text)
            
            explanation_id = f"exp_{student_id}_{datetime.now().timestamp()}"
            
            return ExplanationAnalysis(
                explanation_id=explanation_id,
                student_id=student_id,
                physics_concept=physics_concept,
                explanation_text=explanation_text,
                quality_score=quality_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                clarity_score=clarity_score,
                conceptual_understanding=conceptual_understanding,
                identified_misconceptions=misconceptions,
                key_concepts_mentioned=key_concepts,
                missing_concepts=missing_concepts,
                language_complexity=language_complexity,
                sentiment_score=sentiment_score,
                confidence_indicators=confidence_indicators,
                uncertainty_indicators=uncertainty_indicators
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze explanation: {e}")
            return None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep physics notation
            text = re.sub(r'[^\w\s\=\+\-\*\/\(\)\.]', ' ', text)
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess text: {e}")
            return text
    
    async def _detect_misconceptions(self, text: str) -> List[MisconceptionDetection]:
        """Detect physics misconceptions in student text"""
        try:
            detected_misconceptions = []
            
            for misconception_id, pattern_data in self.misconception_patterns.items():
                patterns = pattern_data['patterns']
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    
                    if matches:
                        # Calculate confidence based on pattern strength and context
                        confidence = min(1.0, len(matches) * 0.3 + 0.4)
                        
                        misconception = MisconceptionDetection(
                            misconception_id=misconception_id,
                            category=pattern_data['category'],
                            description=pattern_data['description'],
                            confidence=confidence,
                            evidence_text=' '.join(matches),
                            correct_concept=pattern_data['correct_concept'],
                            remediation_suggestions=[
                                f"Review {pattern_data['description'].lower()}",
                                "Practice with specific examples",
                                "Discuss with instructor"
                            ],
                            severity='medium' if confidence > 0.7 else 'low',
                            frequency=len(matches)
                        )
                        
                        detected_misconceptions.append(misconception)
            
            return detected_misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to detect misconceptions: {e}")
            return []
    
    async def _calculate_completeness_score(self, text: str, physics_concept: str) -> float:
        """Calculate how complete the explanation is"""
        try:
            # Get expected concepts for this physics topic
            expected_concepts = self.physics_vocabulary.get(physics_concept, [])
            
            if not expected_concepts:
                return 0.5  # Default score if concept not found
            
            # Count how many expected concepts are mentioned
            mentioned_count = 0
            for concept in expected_concepts:
                if concept.lower() in text.lower():
                    mentioned_count += 1
            
            # Calculate completeness ratio
            completeness = mentioned_count / len(expected_concepts)
            
            # Apply non-linear scaling to reward comprehensive explanations
            completeness_score = min(1.0, completeness * 1.2)
            
            return completeness_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate completeness score: {e}")
            return 0.5
    
    async def _calculate_accuracy_score(self, text: str, 
                                      misconceptions: List[MisconceptionDetection]) -> float:
        """Calculate accuracy score based on misconceptions detected"""
        try:
            if not misconceptions:
                return 1.0  # No misconceptions = high accuracy
            
            # Calculate penalty based on misconception severity and confidence
            total_penalty = 0.0
            for misconception in misconceptions:
                severity_weight = {'low': 0.1, 'medium': 0.3, 'high': 0.5}
                penalty = severity_weight.get(misconception.severity, 0.3) * misconception.confidence
                total_penalty += penalty
            
            # Convert penalty to accuracy score
            accuracy_score = max(0.0, 1.0 - total_penalty)
            
            return accuracy_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate accuracy score: {e}")
            return 0.5
    
    async def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity score based on language use and structure"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not sentences or not words:
                return 0.0
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Optimal sentence length for clarity (10-20 words)
            length_score = 1.0 - min(1.0, abs(avg_sentence_length - 15) / 15)
            
            # Vocabulary diversity (unique words / total words)
            unique_words = len(set(words))
            diversity_score = min(1.0, unique_words / len(words) * 2)
            
            # Physics terminology usage
            physics_terms = [word for word in words if word in self.physics_vocabulary.get('all', [])]
            terminology_score = min(1.0, len(physics_terms) / len(words) * 10)
            
            # Combine scores
            clarity_score = (length_score + diversity_score + terminology_score) / 3.0
            
            return clarity_score
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate clarity score: {e}")
            return 0.5
    
    async def _assess_conceptual_understanding(self, text: str, physics_concept: str) -> float:
        """Assess depth of conceptual understanding"""
        try:
            understanding_score = 0.0
            
            # Check for concept patterns
            concept_patterns = self.concept_patterns.get(physics_concept, [])
            
            for pattern in concept_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    understanding_score += 0.2
            
            # Check for causal reasoning (because, since, therefore, etc.)
            causal_indicators = ['because', 'since', 'therefore', 'thus', 'hence', 'so', 'due to']
            causal_count = sum(1 for indicator in causal_indicators if indicator in text.lower())
            understanding_score += min(0.3, causal_count * 0.1)
            
            # Check for examples or applications
            example_indicators = ['example', 'for instance', 'such as', 'like when']
            example_count = sum(1 for indicator in example_indicators if indicator in text.lower())
            understanding_score += min(0.2, example_count * 0.1)
            
            # Check for connections between concepts
            connection_indicators = ['related to', 'connected', 'leads to', 'results in', 'affects']
            connection_count = sum(1 for indicator in connection_indicators if indicator in text.lower())
            understanding_score += min(0.3, connection_count * 0.15)
            
            return min(1.0, understanding_score)
            
        except Exception as e:
            logger.error(f"❌ Failed to assess conceptual understanding: {e}")
            return 0.5
    
    async def _extract_mentioned_concepts(self, text: str) -> List[str]:
        """Extract physics concepts mentioned in the text"""
        try:
            mentioned_concepts = []
            
            for concept_category, terms in self.physics_vocabulary.items():
                if concept_category == 'all':
                    continue
                    
                for term in terms:
                    if term.lower() in text.lower():
                        mentioned_concepts.append(term)
            
            return list(set(mentioned_concepts))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Failed to extract mentioned concepts: {e}")
            return []
    
    async def _identify_missing_concepts(self, mentioned_concepts: List[str], 
                                       physics_concept: str) -> List[str]:
        """Identify important concepts that should be mentioned but aren't"""
        try:
            expected_concepts = self.physics_vocabulary.get(physics_concept, [])
            mentioned_lower = [concept.lower() for concept in mentioned_concepts]
            
            missing_concepts = []
            for concept in expected_concepts:
                if concept.lower() not in mentioned_lower:
                    missing_concepts.append(concept)
            
            return missing_concepts
            
        except Exception as e:
            logger.error(f"❌ Failed to identify missing concepts: {e}")
            return []
    
    def _assess_language_complexity(self, text: str) -> str:
        """Assess the complexity level of language used"""
        try:
            words = word_tokenize(text)
            sentences = sent_tokenize(text)
            
            if not words or not sentences:
                return 'basic'
            
            # Average word length
            avg_word_length = np.mean([len(word) for word in words if word.isalpha()])
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Complex word ratio (words with 3+ syllables)
            complex_words = [word for word in words if self._count_syllables(word) >= 3]
            complex_ratio = len(complex_words) / len(words) if words else 0
            
            # Determine complexity level
            if avg_word_length > 6 and avg_sentence_length > 20 and complex_ratio > 0.2:
                return 'advanced'
            elif avg_word_length > 4.5 and avg_sentence_length > 15 and complex_ratio > 0.1:
                return 'intermediate'
            else:
                return 'basic'
                
        except Exception as e:
            logger.error(f"❌ Failed to assess language complexity: {e}")
            return 'basic'
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting for complexity assessment"""
        try:
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Handle silent e
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            return max(1, syllable_count)
            
        except Exception as e:
            return 1
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment to gauge student attitude and confidence"""
        try:
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(text)
                return scores['compound']  # Compound score ranges from -1 to 1
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze sentiment: {e}")
            return 0.0
    
    def _detect_confidence_indicators(self, text: str) -> List[str]:
        """Detect indicators of student confidence in their explanation"""
        try:
            confidence_patterns = [
                r'i am sure', r'i know', r'definitely', r'certainly',
                r'clearly', r'obviously', r'without doubt', r'confident'
            ]
            
            indicators = []
            for pattern in confidence_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                indicators.extend(matches)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to detect confidence indicators: {e}")
            return []
    
    def _detect_uncertainty_indicators(self, text: str) -> List[str]:
        """Detect indicators of student uncertainty"""
        try:
            uncertainty_patterns = [
                r'i think', r'maybe', r'probably', r'might be',
                r'not sure', r'unclear', r'confused', r'unsure'
            ]
            
            indicators = []
            for pattern in uncertainty_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                indicators.extend(matches)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to detect uncertainty indicators: {e}")
            return []

class PhysicsDiagramAnalyzer:
    """Computer vision analyzer for hand-drawn physics diagrams"""
    
    def __init__(self):
        self.object_detector = None
        self.text_recognizer = None
        self.shape_classifier = None
        
        # Physics diagram templates and patterns
        self.diagram_templates = {}
        self.shape_patterns = {}
    
    async def initialize(self):
        """Initialize computer vision models"""
        try:
            logger.info("🚀 Initializing Physics Diagram Analyzer")
            
            # Initialize OCR
            # Configure pytesseract path if needed
            # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
            
            # Initialize diagram templates
            await self._initialize_diagram_templates()
            
            # Initialize shape patterns
            await self._initialize_shape_patterns()
            
            logger.info("✅ Physics Diagram Analyzer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Physics Diagram Analyzer: {e}")
            return False
    
    async def _initialize_diagram_templates(self):
        """Initialize templates for different types of physics diagrams"""
        try:
            self.diagram_templates = {
                'force_diagram': {
                    'required_elements': ['object', 'force_vectors', 'labels'],
                    'optional_elements': ['coordinate_system', 'angles', 'magnitudes'],
                    'common_errors': ['missing_forces', 'wrong_directions', 'no_labels']
                },
                'circuit_diagram': {
                    'required_elements': ['components', 'connections', 'current_flow'],
                    'optional_elements': ['voltage_labels', 'component_values'],
                    'common_errors': ['open_circuits', 'wrong_polarity', 'missing_components']
                },
                'motion_graph': {
                    'required_elements': ['axes', 'curve', 'labels'],
                    'optional_elements': ['units', 'scale', 'grid'],
                    'common_errors': ['wrong_shape', 'missing_labels', 'incorrect_scale']
                },
                'energy_diagram': {
                    'required_elements': ['energy_levels', 'transitions', 'labels'],
                    'optional_elements': ['values', 'arrows', 'conservation_note'],
                    'common_errors': ['wrong_levels', 'missing_conservation', 'no_arrows']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize diagram templates: {e}")
    
    async def _initialize_shape_patterns(self):
        """Initialize patterns for recognizing physics diagram shapes"""
        try:
            self.shape_patterns = {
                'vector_arrow': {
                    'description': 'Arrow representing force, velocity, or field',
                    'key_features': ['line_segment', 'arrowhead', 'direction']
                },
                'free_body_object': {
                    'description': 'Central object in force diagram',
                    'key_features': ['geometric_shape', 'central_position', 'force_attachment_points']
                },
                'inclined_plane': {
                    'description': 'Sloped surface for physics problems',
                    'key_features': ['triangular_shape', 'angle_marking', 'surface_indication']
                },
                'spring': {
                    'description': 'Coiled spring representation',
                    'key_features': ['coiled_pattern', 'compression_extension', 'attachment_points']
                },
                'pulley': {
                    'description': 'Circular pulley with rope/string',
                    'key_features': ['circular_shape', 'rope_path', 'rotation_indication']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize shape patterns: {e}")
    
    async def analyze_diagram(self, student_id: str, image_data: str, 
                            expected_diagram_type: str = None) -> DiagramAnalysis:
        """Analyze hand-drawn physics diagram"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect diagram type if not provided
            if not expected_diagram_type:
                expected_diagram_type = await self._classify_diagram_type(cv_image)
            
            # Extract text using OCR
            extracted_text = await self._extract_text_from_image(image)
            
            # Detect objects and shapes
            detected_objects = await self._detect_physics_objects(cv_image, expected_diagram_type)
            
            # Extract mathematical expressions
            math_expressions = await self._extract_math_expressions(extracted_text)
            
            # Analyze completeness
            completeness_score = await self._analyze_diagram_completeness(
                detected_objects, expected_diagram_type
            )
            
            # Analyze accuracy
            accuracy_score = await self._analyze_diagram_accuracy(
                detected_objects, expected_diagram_type
            )
            
            # Analyze labeling quality
            labeling_quality = await self._analyze_labeling_quality(
                extracted_text, detected_objects
            )
            
            # Detect errors
            detected_errors = await self._detect_diagram_errors(
                detected_objects, expected_diagram_type
            )
            
            # Identify missing elements
            missing_elements = await self._identify_missing_elements(
                detected_objects, expected_diagram_type
            )
            
            # Identify physics principles shown
            physics_principles = await self._identify_physics_principles(
                detected_objects, extracted_text, expected_diagram_type
            )
            
            diagram_id = f"diag_{student_id}_{datetime.now().timestamp()}"
            
            return DiagramAnalysis(
                diagram_id=diagram_id,
                student_id=student_id,
                image_data=image_data,
                detected_objects=detected_objects,
                diagram_type=expected_diagram_type,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                labeling_quality=labeling_quality,
                detected_errors=detected_errors,
                missing_elements=missing_elements,
                physics_principles_shown=physics_principles,
                extracted_text=extracted_text,
                mathematical_expressions=math_expressions
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze diagram: {e}")
            return None
    
    async def _classify_diagram_type(self, cv_image: np.ndarray) -> str:
        """Classify the type of physics diagram"""
        try:
            # Simple heuristic classification based on image features
            # In practice, this would use a trained CNN
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect lines and shapes
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            # Count circular shapes (potential pulleys, objects)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)
            
            # Simple classification logic
            if circles is not None and len(circles[0]) > 0:
                if lines is not None and len(lines) > 10:
                    return 'circuit_diagram'
                else:
                    return 'force_diagram'
            elif lines is not None and len(lines) > 20:
                return 'motion_graph'
            else:
                return 'force_diagram'  # Default
                
        except Exception as e:
            logger.error(f"❌ Failed to classify diagram type: {e}")
            return 'unknown'
    
    async def _extract_text_from_image(self, image: Image) -> str:
        """Extract text from diagram using OCR"""
        try:
            # Use pytesseract for OCR
            extracted_text = pytesseract.image_to_string(image, config='--psm 6')
            
            # Clean extracted text
            cleaned_text = ' '.join(extracted_text.split())
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"❌ Failed to extract text from image: {e}")
            return ""
    
    async def _detect_physics_objects(self, cv_image: np.ndarray, 
                                    diagram_type: str) -> List[Dict[str, Any]]:
        """Detect physics-specific objects in the diagram"""
        try:
            detected_objects = []
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect arrows (force vectors)
            arrows = await self._detect_arrows(gray)
            for arrow in arrows:
                detected_objects.append({
                    'type': 'vector_arrow',
                    'position': arrow['position'],
                    'direction': arrow['direction'],
                    'length': arrow['length'],
                    'confidence': arrow['confidence']
                })
            
            # Detect geometric shapes (objects, masses)
            shapes = await self._detect_geometric_shapes(gray)
            for shape in shapes:
                detected_objects.append({
                    'type': 'geometric_object',
                    'shape': shape['shape'],
                    'position': shape['position'],
                    'size': shape['size'],
                    'confidence': shape['confidence']
                })
            
            # Detect coordinate systems
            coord_systems = await self._detect_coordinate_systems(gray)
            for coord_sys in coord_systems:
                detected_objects.append({
                    'type': 'coordinate_system',
                    'position': coord_sys['position'],
                    'axes': coord_sys['axes'],
                    'confidence': coord_sys['confidence']
                })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"❌ Failed to detect physics objects: {e}")
            return []
    
    async def _detect_arrows(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect arrow shapes representing vectors"""
        try:
            arrows = []
            
            # Detect lines first
            edges = cv2.Canny(gray_image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate direction and length
                    direction = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Simple arrow detection (look for arrowheads)
                    # This is a simplified implementation
                    arrows.append({
                        'position': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'direction': direction,
                        'length': length,
                        'confidence': 0.7  # Simplified confidence
                    })
            
            return arrows
            
        except Exception as e:
            logger.error(f"❌ Failed to detect arrows: {e}")
            return []
    
    async def _detect_geometric_shapes(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect geometric shapes representing physics objects"""
        try:
            shapes = []
            
            # Detect circles
            circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                     param1=50, param2=30, minRadius=5, maxRadius=100)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    shapes.append({
                        'shape': 'circle',
                        'position': (x, y),
                        'size': r * 2,  # diameter
                        'confidence': 0.8
                    })
            
            # Detect rectangles/squares
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangle/square
                    x, y, w, h = cv2.boundingRect(contour)
                    shapes.append({
                        'shape': 'rectangle',
                        'position': (x + w//2, y + h//2),
                        'size': (w, h),
                        'confidence': 0.7
                    })
            
            return shapes
            
        except Exception as e:
            logger.error(f"❌ Failed to detect geometric shapes: {e}")
            return []
    
    async def _detect_coordinate_systems(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect coordinate system axes"""
        try:
            coord_systems = []
            
            # Detect perpendicular lines that could be axes
            edges = cv2.Canny(gray_image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=10)
            
            if lines is not None and len(lines) >= 2:
                # Look for perpendicular line pairs
                for i, line1 in enumerate(lines):
                    for j, line2 in enumerate(lines[i+1:], i+1):
                        x1, y1, x2, y2 = line1[0]
                        x3, y3, x4, y4 = line2[0]
                        
                        # Calculate angles
                        angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        angle2 = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
                        
                        # Check if lines are approximately perpendicular
                        angle_diff = abs(angle1 - angle2)
                        if abs(angle_diff - 90) < 15 or abs(angle_diff - 270) < 15:
                            # Found potential coordinate system
                            coord_systems.append({
                                'position': ((x1 + x2 + x3 + x4) // 4, (y1 + y2 + y3 + y4) // 4),
                                'axes': [line1[0], line2[0]],
                                'confidence': 0.6
                            })
                            break
            
            return coord_systems
            
        except Exception as e:
            logger.error(f"❌ Failed to detect coordinate systems: {e}")
            return []
    
    async def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from OCR text"""
        try:
            math_patterns = [
                r'[a-zA-Z]\s*=\s*[0-9\.]+',  # Variable = number
                r'F\s*=\s*m\s*\*?\s*a',      # F = ma
                r'v\s*=\s*u\s*\+\s*a\s*\*?\s*t',  # v = u + at
                r'E\s*=\s*\d*\s*\/?\s*2\s*\*?\s*m\s*\*?\s*v\^?2',  # Kinetic energy
                r'\d+\s*(N|kg|m\/s|m\/s\^?2)',  # Units
            ]
            
            expressions = []
            for pattern in math_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                expressions.extend(matches)
            
            return expressions
            
        except Exception as e:
            logger.error(f"❌ Failed to extract math expressions: {e}")
            return []
    
    async def _analyze_diagram_completeness(self, detected_objects: List[Dict[str, Any]],
                                          diagram_type: str) -> float:
        """Analyze how complete the diagram is"""
        try:
            template = self.diagram_templates.get(diagram_type, {})
            required_elements = template.get('required_elements', [])
            
            if not required_elements:
                return 0.5  # Default score
            
            # Count how many required elements are present
            present_count = 0
            object_types = [obj.get('type', '') for obj in detected_objects]
            
            for required in required_elements:
                # Map required elements to object types
                if required == 'force_vectors' and 'vector_arrow' in object_types:
                    present_count += 1
                elif required == 'object' and 'geometric_object' in object_types:
                    present_count += 1
                elif required == 'coordinate_system' and 'coordinate_system' in object_types:
                    present_count += 1
                # Add more mappings as needed
            
            completeness = present_count / len(required_elements)
            return min(1.0, completeness)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze diagram completeness: {e}")
            return 0.5
    
    async def _analyze_diagram_accuracy(self, detected_objects: List[Dict[str, Any]],
                                      diagram_type: str) -> float:
        """Analyze accuracy of the diagram elements"""
        try:
            # Simplified accuracy assessment
            # In practice, this would check physics principles
            
            total_objects = len(detected_objects)
            if total_objects == 0:
                return 0.0
            
            accurate_objects = 0
            
            for obj in detected_objects:
                # Basic accuracy checks
                confidence = obj.get('confidence', 0.0)
                
                # Objects with high detection confidence are likely accurate
                if confidence > 0.7:
                    accurate_objects += 1
                elif confidence > 0.5:
                    accurate_objects += 0.5
            
            accuracy = accurate_objects / total_objects
            return min(1.0, accuracy)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze diagram accuracy: {e}")
            return 0.5
    
    async def _analyze_labeling_quality(self, extracted_text: str,
                                      detected_objects: List[Dict[str, Any]]) -> float:
        """Analyze quality of labels in the diagram"""
        try:
            if not extracted_text.strip():
                return 0.0  # No labels found
            
            # Count physics terms in labels
            physics_terms = ['force', 'velocity', 'acceleration', 'mass', 'weight', 
                           'friction', 'normal', 'tension', 'F', 'v', 'a', 'm', 'N']
            
            words = extracted_text.lower().split()
            physics_term_count = sum(1 for word in words if any(term.lower() in word for term in physics_terms))
            
            # Basic labeling quality score
            if len(words) > 0:
                labeling_quality = min(1.0, physics_term_count / len(words) * 3)
            else:
                labeling_quality = 0.0
            
            return labeling_quality
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze labeling quality: {e}")
            return 0.5
    
    async def _detect_diagram_errors(self, detected_objects: List[Dict[str, Any]],
                                   diagram_type: str) -> List[str]:
        """Detect common errors in physics diagrams"""
        try:
            errors = []
            
            template = self.diagram_templates.get(diagram_type, {})
            common_errors = template.get('common_errors', [])
            
            # Check for common errors based on detected objects
            object_types = [obj.get('type', '') for obj in detected_objects]
            
            if 'missing_forces' in common_errors:
                if 'vector_arrow' not in object_types:
                    errors.append('No force vectors detected')
            
            if 'missing_labels' in common_errors:
                # This would be checked against extracted text
                errors.append('Insufficient labeling detected')
            
            if 'wrong_directions' in common_errors:
                # Would need more sophisticated analysis
                pass
            
            return errors
            
        except Exception as e:
            logger.error(f"❌ Failed to detect diagram errors: {e}")
            return []
    
    async def _identify_missing_elements(self, detected_objects: List[Dict[str, Any]],
                                       diagram_type: str) -> List[str]:
        """Identify elements that should be present but are missing"""
        try:
            missing = []
            
            template = self.diagram_templates.get(diagram_type, {})
            required_elements = template.get('required_elements', [])
            
            object_types = [obj.get('type', '') for obj in detected_objects]
            
            for required in required_elements:
                if required == 'force_vectors' and 'vector_arrow' not in object_types:
                    missing.append('Force vectors')
                elif required == 'object' and 'geometric_object' not in object_types:
                    missing.append('Central object')
                elif required == 'coordinate_system' and 'coordinate_system' not in object_types:
                    missing.append('Coordinate system')
            
            return missing
            
        except Exception as e:
            logger.error(f"❌ Failed to identify missing elements: {e}")
            return []
    
    async def _identify_physics_principles(self, detected_objects: List[Dict[str, Any]],
                                         extracted_text: str, diagram_type: str) -> List[str]:
        """Identify physics principles demonstrated in the diagram"""
        try:
            principles = []
            
            # Based on diagram type and detected elements
            if diagram_type == 'force_diagram':
                if any(obj.get('type') == 'vector_arrow' for obj in detected_objects):
                    principles.append('Force representation')
                if 'equilibrium' in extracted_text.lower():
                    principles.append('Force equilibrium')
                if 'newton' in extracted_text.lower():
                    principles.append('Newton\'s laws')
            
            elif diagram_type == 'motion_graph':
                if 'velocity' in extracted_text.lower():
                    principles.append('Kinematics')
                if 'acceleration' in extracted_text.lower():
                    principles.append('Acceleration analysis')
            
            elif diagram_type == 'energy_diagram':
                if 'conservation' in extracted_text.lower():
                    principles.append('Energy conservation')
                if 'kinetic' in extracted_text.lower() or 'potential' in extracted_text.lower():
                    principles.append('Energy transformation')
            
            return principles
            
        except Exception as e:
            logger.error(f"❌ Failed to identify physics principles: {e}")
            return []

class AdvancedLearningAnalyticsEngine:
    """Main engine coordinating all advanced analytics components"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Component analyzers
        self.nlp_processor = PhysicsNLPProcessor()
        self.diagram_analyzer = PhysicsDiagramAnalyzer()
        
        # Analytics storage
        self.explanation_analyses = defaultdict(list)
        self.diagram_analyses = defaultdict(list)
        self.learning_progressions = defaultdict(dict)
        
        # Configuration
        self.config = {
            'min_explanations_for_progression': 5,
            'analysis_retention_days': 90,
            'quality_threshold': 0.7,
            'misconception_threshold': 0.6
        }
    
    async def initialize(self):
        """Initialize the advanced learning analytics engine"""
        try:
            logger.info("🚀 Initializing Advanced Learning Analytics Engine")
            
            # Initialize component analyzers
            await self.nlp_processor.initialize()
            await self.diagram_analyzer.initialize()
            
            # Load historical analyses
            await self._load_historical_analyses()
            
            logger.info("✅ Advanced Learning Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced Learning Analytics Engine: {e}")
            return False
    
    async def _load_historical_analyses(self):
        """Load historical analysis data"""
        try:
            if not self.db_manager:
                return
            
            # Load explanation analyses
            async with self.db_manager.postgres.get_connection() as conn:
                analyses = await conn.fetch("""
                    SELECT * FROM explanation_analyses 
                    WHERE created_at >= $1
                    ORDER BY created_at DESC
                """, datetime.now() - timedelta(days=self.config['analysis_retention_days']))
                
                for analysis in analyses:
                    student_id = str(analysis['student_id'])
                    # Convert database record to ExplanationAnalysis object
                    # Implementation would depend on database schema
                    pass
            
            logger.info(f"📊 Loaded historical analyses for {len(self.explanation_analyses)} students")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical analyses: {e}")
    
    async def analyze_student_explanation(self, student_id: str, explanation_text: str,
                                        physics_concept: str) -> ExplanationAnalysis:
        """Analyze student explanation using NLP"""
        try:
            analysis = await self.nlp_processor.analyze_explanation(
                student_id, explanation_text, physics_concept
            )
            
            if analysis:
                # Store analysis
                self.explanation_analyses[student_id].append(analysis)
                
                # Update learning progression
                await self._update_learning_progression(student_id, analysis)
                
                # Store in database
                await self._store_explanation_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze student explanation: {e}")
            return None
    
    async def analyze_student_diagram(self, student_id: str, image_data: str,
                                    expected_diagram_type: str = None) -> DiagramAnalysis:
        """Analyze student diagram using computer vision"""
        try:
            analysis = await self.diagram_analyzer.analyze_diagram(
                student_id, image_data, expected_diagram_type
            )
            
            if analysis:
                # Store analysis
                self.diagram_analyses[student_id].append(analysis)
                
                # Store in database
                await self._store_diagram_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze student diagram: {e}")
            return None
    
    async def _update_learning_progression(self, student_id: str, 
                                         analysis: ExplanationAnalysis):
        """Update learning progression based on new analysis"""
        try:
            concept = analysis.physics_concept
            
            if concept not in self.learning_progressions[student_id]:
                self.learning_progressions[student_id][concept] = {
                    'mastery_trajectory': [],
                    'explanation_evolution': [],
                    'misconception_resolution': {},
                    'conceptual_breakthroughs': [],
                    'learning_plateau_periods': [],
                    'vocabulary_development': defaultdict(int)
                }
            
            progression = self.learning_progressions[student_id][concept]
            
            # Add to mastery trajectory
            progression['mastery_trajectory'].append(
                (datetime.now(), analysis.conceptual_understanding)
            )
            
            # Add to explanation evolution
            progression['explanation_evolution'].append(analysis)
            
            # Track misconception resolution
            for misconception in analysis.identified_misconceptions:
                misconception_id = misconception.misconception_id
                if misconception_id not in progression['misconception_resolution']:
                    progression['misconception_resolution'][misconception_id] = False
                
                # Check if misconception is being resolved (lower confidence over time)
                # This would require more sophisticated analysis
            
            # Track vocabulary development
            for concept_mentioned in analysis.key_concepts_mentioned:
                progression['vocabulary_development'][concept_mentioned] += 1
            
            # Detect conceptual breakthroughs
            if analysis.conceptual_understanding > 0.8 and analysis.quality_score > 0.7:
                # Check if this is a significant improvement
                if len(progression['mastery_trajectory']) > 1:
                    prev_mastery = progression['mastery_trajectory'][-2][1]
                    if analysis.conceptual_understanding - prev_mastery > 0.3:
                        progression['conceptual_breakthroughs'].append(datetime.now())
            
        except Exception as e:
            logger.error(f"❌ Failed to update learning progression: {e}")
    
    async def generate_learning_progression_analysis(self, student_id: str,
                                                   concept: str) -> LearningProgressionAnalysis:
        """Generate comprehensive learning progression analysis"""
        try:
            if student_id not in self.learning_progressions:
                return None
            
            if concept not in self.learning_progressions[student_id]:
                return None
            
            progression_data = self.learning_progressions[student_id][concept]
            
            # Calculate overall progression score
            mastery_trajectory = progression_data['mastery_trajectory']
            if len(mastery_trajectory) < 2:
                progression_score = 0.5
            else:
                # Calculate slope of mastery over time
                times = [(t - mastery_trajectory[0][0]).days for t, _ in mastery_trajectory]
                masteries = [m for _, m in mastery_trajectory]
                
                if len(times) > 1 and times[-1] > 0:
                    progression_score = (masteries[-1] - masteries[0]) / times[-1]
                    progression_score = max(0.0, min(1.0, progression_score + 0.5))
                else:
                    progression_score = 0.5
            
            return LearningProgressionAnalysis(
                student_id=student_id,
                concept=concept,
                time_period=f"{len(mastery_trajectory)} assessments",
                progression_score=progression_score,
                mastery_trajectory=mastery_trajectory,
                explanation_evolution=progression_data['explanation_evolution'],
                misconception_resolution=progression_data['misconception_resolution'],
                conceptual_breakthroughs=progression_data['conceptual_breakthroughs'],
                learning_plateau_periods=progression_data['learning_plateau_periods'],
                vocabulary_development=dict(progression_data['vocabulary_development'])
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate learning progression analysis: {e}")
            return None
    
    async def _store_explanation_analysis(self, analysis: ExplanationAnalysis):
        """Store explanation analysis in database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO explanation_analyses 
                    (explanation_id, student_id, physics_concept, quality_score, 
                     conceptual_understanding, misconceptions_detected, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, analysis.explanation_id, analysis.student_id, analysis.physics_concept,
                    analysis.quality_score, analysis.conceptual_understanding,
                    json.dumps([m.misconception_id for m in analysis.identified_misconceptions]),
                    datetime.now())
        
        except Exception as e:
            logger.error(f"❌ Failed to store explanation analysis: {e}")
    
    async def _store_diagram_analysis(self, analysis: DiagramAnalysis):
        """Store diagram analysis in database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO diagram_analyses 
                    (diagram_id, student_id, diagram_type, completeness_score, 
                     accuracy_score, detected_errors, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, analysis.diagram_id, analysis.student_id, analysis.diagram_type,
                    analysis.completeness_score, analysis.accuracy_score,
                    json.dumps(analysis.detected_errors), datetime.now())
        
        except Exception as e:
            logger.error(f"❌ Failed to store diagram analysis: {e}")

# Testing function
async def test_advanced_learning_analytics():
    """Test the advanced learning analytics engine"""
    try:
        logger.info("🧪 Testing Advanced Learning Analytics Engine")
        
        engine = AdvancedLearningAnalyticsEngine()
        await engine.initialize()
        
        # Test explanation analysis
        sample_explanation = """
        Force is what makes objects move. When you push something, you apply force to it.
        The harder you push, the faster it moves. Newton's first law says that objects
        at rest stay at rest unless a force acts on them.
        """
        
        explanation_analysis = await engine.analyze_student_explanation(
            "test_student", sample_explanation, "forces"
        )
        
        if explanation_analysis:
            logger.info(f"✅ Explanation analysis completed - Quality: {explanation_analysis.quality_score:.2f}")
            logger.info(f"✅ Misconceptions detected: {len(explanation_analysis.identified_misconceptions)}")
        
        # Test learning progression
        progression_analysis = await engine.generate_learning_progression_analysis(
            "test_student", "forces"
        )
        
        if progression_analysis:
            logger.info(f"✅ Learning progression analysis completed - Score: {progression_analysis.progression_score:.2f}")
        
        logger.info("✅ Advanced Learning Analytics Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Advanced Learning Analytics Engine test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_advanced_learning_analytics())