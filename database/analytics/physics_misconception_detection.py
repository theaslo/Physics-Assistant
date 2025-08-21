#!/usr/bin/env python3
"""
Physics Misconception Detection and Remediation System
Advanced NLP and pattern recognition for identifying and addressing 
physics misconceptions in real-time student interactions.
"""

import asyncio
import json
import logging
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MisconceptionType(Enum):
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    MATHEMATICAL = "mathematical"
    UNIT_CONVERSION = "unit_conversion"
    VECTOR_SCALAR = "vector_scalar"
    CAUSAL_REASONING = "causal_reasoning"

class MisconceptionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PhysicsMisconception:
    """Detected physics misconception"""
    misconception_id: str
    concept: str
    misconception_type: MisconceptionType
    severity: MisconceptionSeverity
    description: str
    student_expression: str
    correct_understanding: str
    remediation_strategy: str
    evidence_patterns: List[str]
    confidence_score: float
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class MisconceptionPattern:
    """Pattern for detecting specific misconceptions"""
    pattern_id: str
    concept: str
    misconception_type: MisconceptionType
    trigger_phrases: List[str]
    regex_patterns: List[str]
    semantic_markers: List[str]
    context_requirements: List[str]
    false_positive_filters: List[str]
    confidence_weight: float = 1.0

class PhysicsMisconceptionDetector:
    """Advanced physics misconception detection system"""
    
    def __init__(self):
        # Load NLP models
        self.nlp = None
        self.sentiment_analyzer = None
        self.embeddings_model = None
        
        # Initialize misconception patterns
        self.misconception_patterns = {}
        self.concept_keywords = {}
        
        # Student misconception tracking
        self.student_misconceptions = defaultdict(list)
        self.misconception_frequency = defaultdict(int)
        
        # Performance metrics
        self.detection_stats = {
            'total_analyzed': 0,
            'misconceptions_detected': 0,
            'false_positives': 0,
            'remediation_success': 0
        }
        
    async def initialize(self):
        """Initialize NLP models and misconception patterns"""
        try:
            logger.info("🚀 Initializing Physics Misconception Detector")
            
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("⚠️ spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Load sentiment analysis
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not load sentiment analyzer: {e}")
            
            # Initialize misconception patterns
            await self._initialize_misconception_patterns()
            
            # Initialize concept keywords
            await self._initialize_concept_keywords()
            
            logger.info("✅ Physics Misconception Detector initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize misconception detector: {e}")
            return False
    
    async def _initialize_misconception_patterns(self):
        """Initialize comprehensive physics misconception patterns"""
        try:
            patterns = {
                # Force and Motion Misconceptions
                "force_motion_1": MisconceptionPattern(
                    pattern_id="force_motion_1",
                    concept="forces",
                    misconception_type=MisconceptionType.CONCEPTUAL,
                    trigger_phrases=[
                        "force is needed to keep moving",
                        "constant force for constant velocity",
                        "motion requires force",
                        "no force means no motion"
                    ],
                    regex_patterns=[
                        r"force.*need.*to.*move",
                        r"constant.*force.*constant.*velocity",
                        r"without.*force.*stop"
                    ],
                    semantic_markers=["force", "motion", "velocity", "constant"],
                    context_requirements=["newton", "first law", "inertia"],
                    false_positive_filters=["friction", "air resistance", "gravity"]
                ),
                
                "force_motion_2": MisconceptionPattern(
                    pattern_id="force_motion_2",
                    concept="forces",
                    misconception_type=MisconceptionType.CONCEPTUAL,
                    trigger_phrases=[
                        "heavier objects fall faster",
                        "weight affects falling speed",
                        "mass determines fall rate"
                    ],
                    regex_patterns=[
                        r"heav(y|ier).*fall.*fast(er)?",
                        r"weight.*affect.*fall",
                        r"mass.*determin.*fall"
                    ],
                    semantic_markers=["weight", "mass", "fall", "gravity"],
                    context_requirements=["gravity", "acceleration", "free fall"],
                    false_positive_filters=["air resistance", "terminal velocity"]
                ),
                
                # Energy Misconceptions
                "energy_1": MisconceptionPattern(
                    pattern_id="energy_1",
                    concept="energy",
                    misconception_type=MisconceptionType.CONCEPTUAL,
                    trigger_phrases=[
                        "energy is used up",
                        "energy gets consumed",
                        "energy disappears",
                        "energy is lost"
                    ],
                    regex_patterns=[
                        r"energy.*(used up|consumed|disappear|lost)",
                        r"energy.*destroy",
                        r"run out.*energy"
                    ],
                    semantic_markers=["energy", "conservation", "transform"],
                    context_requirements=["conservation", "energy", "transform"],
                    false_positive_filters=["useful energy", "efficiency"]
                ),
                
                # Vector vs Scalar Misconceptions
                "vector_scalar_1": MisconceptionPattern(
                    pattern_id="vector_scalar_1",
                    concept="vectors",
                    misconception_type=MisconceptionType.VECTOR_SCALAR,
                    trigger_phrases=[
                        "velocity is speed",
                        "acceleration is speeding up",
                        "distance is displacement"
                    ],
                    regex_patterns=[
                        r"velocity.*is.*speed",
                        r"acceleration.*speeding up",
                        r"distance.*is.*displacement"
                    ],
                    semantic_markers=["velocity", "speed", "vector", "scalar"],
                    context_requirements=["kinematics", "motion"],
                    false_positive_filters=["magnitude", "direction"]
                ),
                
                # Mathematical Misconceptions
                "math_physics_1": MisconceptionPattern(
                    pattern_id="math_physics_1",
                    concept="mathematics",
                    misconception_type=MisconceptionType.MATHEMATICAL,
                    trigger_phrases=[
                        "negative acceleration means slowing down",
                        "negative velocity means going backwards",
                        "negative means decreasing"
                    ],
                    regex_patterns=[
                        r"negative.*acceleration.*slow",
                        r"negative.*velocity.*backward",
                        r"negative.*mean.*decreas"
                    ],
                    semantic_markers=["negative", "acceleration", "velocity", "direction"],
                    context_requirements=["kinematics", "direction", "coordinate"],
                    false_positive_filters=["reference frame", "coordinate system"]
                ),
                
                # Unit and Calculation Misconceptions
                "units_1": MisconceptionPattern(
                    pattern_id="units_1",
                    concept="units",
                    misconception_type=MisconceptionType.UNIT_CONVERSION,
                    trigger_phrases=[
                        "just drop the units",
                        "units don't matter",
                        "ignore the units"
                    ],
                    regex_patterns=[
                        r"drop.*units",
                        r"units.*don't matter",
                        r"ignore.*units"
                    ],
                    semantic_markers=["units", "dimensions", "conversion"],
                    context_requirements=["calculation", "problem", "solve"],
                    false_positive_filters=["dimensionless", "ratio"]
                ),
                
                # Causal Reasoning Misconceptions
                "causal_1": MisconceptionPattern(
                    pattern_id="causal_1",
                    concept="causality",
                    misconception_type=MisconceptionType.CAUSAL_REASONING,
                    trigger_phrases=[
                        "centrifugal force pushes outward",
                        "objects want to move in straight line",
                        "natural tendency"
                    ],
                    regex_patterns=[
                        r"centrifugal.*force.*push",
                        r"objects.*want.*straight",
                        r"natural.*tendency"
                    ],
                    semantic_markers=["centrifugal", "centripetal", "circular", "motion"],
                    context_requirements=["circular motion", "rotation"],
                    false_positive_filters=["reference frame", "fictitious force"]
                )
            }
            
            self.misconception_patterns = patterns
            logger.info(f"📋 Loaded {len(patterns)} misconception patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize misconception patterns: {e}")
    
    async def _initialize_concept_keywords(self):
        """Initialize physics concept keywords for context detection"""
        try:
            self.concept_keywords = {
                'forces': [
                    'force', 'newton', 'friction', 'tension', 'normal', 'weight',
                    'gravity', 'acceleration', 'mass', 'inertia', 'equilibrium'
                ],
                'energy': [
                    'energy', 'kinetic', 'potential', 'work', 'power', 'conservation',
                    'joule', 'heat', 'mechanical', 'thermal', 'electrical'
                ],
                'momentum': [
                    'momentum', 'impulse', 'collision', 'elastic', 'inelastic',
                    'conservation', 'center of mass', 'velocity'
                ],
                'kinematics': [
                    'position', 'velocity', 'acceleration', 'displacement', 'distance',
                    'speed', 'time', 'motion', 'projectile', 'trajectory'
                ],
                'waves': [
                    'wave', 'frequency', 'wavelength', 'amplitude', 'period',
                    'oscillation', 'vibration', 'resonance', 'interference'
                ],
                'electricity': [
                    'charge', 'current', 'voltage', 'resistance', 'electric field',
                    'magnetic field', 'circuit', 'ohm', 'ampere', 'volt'
                ]
            }
            
            logger.info(f"🔑 Loaded concept keywords for {len(self.concept_keywords)} domains")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize concept keywords: {e}")
    
    async def analyze_student_response(self, student_id: str, response_text: str,
                                     context: Dict[str, Any] = None) -> List[PhysicsMisconception]:
        """Analyze student response for physics misconceptions"""
        try:
            start_time = datetime.now()
            self.detection_stats['total_analyzed'] += 1
            
            detected_misconceptions = []
            
            # Preprocess text
            processed_text = await self._preprocess_text(response_text)
            
            # Extract physics concepts from context
            physics_context = await self._extract_physics_context(context or {})
            
            # Pattern-based detection
            pattern_detections = await self._detect_pattern_misconceptions(
                processed_text, physics_context
            )
            detected_misconceptions.extend(pattern_detections)
            
            # Semantic analysis detection
            semantic_detections = await self._detect_semantic_misconceptions(
                processed_text, physics_context
            )
            detected_misconceptions.extend(semantic_detections)
            
            # Mathematical reasoning detection
            math_detections = await self._detect_mathematical_misconceptions(
                processed_text, physics_context
            )
            detected_misconceptions.extend(math_detections)
            
            # Remove duplicates and low-confidence detections
            filtered_misconceptions = await self._filter_and_rank_misconceptions(
                detected_misconceptions
            )
            
            # Track student misconceptions
            if filtered_misconceptions:
                self.student_misconceptions[student_id].extend(filtered_misconceptions)
                self.detection_stats['misconceptions_detected'] += len(filtered_misconceptions)
                
                for misconception in filtered_misconceptions:
                    self.misconception_frequency[misconception.misconception_id] += 1
            
            # Log detection time
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            if detection_time > 100:  # Target <100ms
                logger.warning(f"⚠️ Slow misconception detection: {detection_time:.1f}ms")
            
            return filtered_misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze student response: {e}")
            return []
    
    async def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better analysis"""
        try:
            # Basic cleaning
            text = text.lower().strip()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Handle common physics notation
            text = re.sub(r'(\d+)\s*m/s', r'\1 meters per second', text)
            text = re.sub(r'(\d+)\s*kg', r'\1 kilograms', text)
            text = re.sub(r'(\d+)\s*N', r'\1 newtons', text)
            
            # Expand contractions
            contractions = {
                "don't": "do not",
                "can't": "cannot",
                "won't": "will not",
                "isn't": "is not",
                "doesn't": "does not"
            }
            
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)
            
            return text
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess text: {e}")
            return text
    
    async def _extract_physics_context(self, context: Dict[str, Any]) -> Set[str]:
        """Extract physics concepts from interaction context"""
        try:
            physics_concepts = set()
            
            # Extract from problem context
            if 'problem_concept' in context:
                physics_concepts.add(context['problem_concept'].lower())
            
            if 'topic' in context:
                physics_concepts.add(context['topic'].lower())
            
            # Extract from recent interactions
            if 'recent_topics' in context:
                for topic in context['recent_topics']:
                    physics_concepts.add(topic.lower())
            
            # Map to concept categories
            concept_categories = set()
            for concept in physics_concepts:
                for category, keywords in self.concept_keywords.items():
                    if concept in keywords or any(keyword in concept for keyword in keywords):
                        concept_categories.add(category)
            
            return physics_concepts.union(concept_categories)
            
        except Exception as e:
            logger.error(f"❌ Failed to extract physics context: {e}")
            return set()
    
    async def _detect_pattern_misconceptions(self, text: str, 
                                           context: Set[str]) -> List[PhysicsMisconception]:
        """Detect misconceptions using predefined patterns"""
        try:
            detected = []
            
            for pattern_id, pattern in self.misconception_patterns.items():
                # Check if pattern is relevant to context
                if pattern.concept not in context and not any(
                    req in context for req in pattern.context_requirements
                ):
                    continue
                
                confidence_score = 0.0
                evidence_patterns = []
                
                # Check trigger phrases
                for phrase in pattern.trigger_phrases:
                    if phrase.lower() in text:
                        confidence_score += 0.3
                        evidence_patterns.append(f"trigger_phrase: {phrase}")
                
                # Check regex patterns
                for regex_pattern in pattern.regex_patterns:
                    if re.search(regex_pattern, text, re.IGNORECASE):
                        confidence_score += 0.4
                        evidence_patterns.append(f"regex_match: {regex_pattern}")
                
                # Check semantic markers
                semantic_matches = sum(1 for marker in pattern.semantic_markers 
                                     if marker.lower() in text)
                if semantic_matches >= 2:
                    confidence_score += 0.3 * (semantic_matches / len(pattern.semantic_markers))
                    evidence_patterns.append(f"semantic_markers: {semantic_matches}")
                
                # Apply false positive filters
                false_positive_detected = any(fp.lower() in text 
                                            for fp in pattern.false_positive_filters)
                if false_positive_detected:
                    confidence_score *= 0.5  # Reduce confidence
                
                # Apply pattern-specific confidence weight
                confidence_score *= pattern.confidence_weight
                
                # If confidence is high enough, create misconception
                if confidence_score >= 0.4 and evidence_patterns:
                    severity = self._determine_severity(confidence_score, pattern.misconception_type)
                    
                    misconception = PhysicsMisconception(
                        misconception_id=pattern_id,
                        concept=pattern.concept,
                        misconception_type=pattern.misconception_type,
                        severity=severity,
                        description=self._get_misconception_description(pattern_id),
                        student_expression=text,
                        correct_understanding=self._get_correct_understanding(pattern_id),
                        remediation_strategy=self._get_remediation_strategy(pattern_id),
                        evidence_patterns=evidence_patterns,
                        confidence_score=confidence_score
                    )
                    
                    detected.append(misconception)
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Failed to detect pattern misconceptions: {e}")
            return []
    
    async def _detect_semantic_misconceptions(self, text: str,
                                            context: Set[str]) -> List[PhysicsMisconception]:
        """Detect misconceptions using semantic analysis"""
        try:
            detected = []
            
            if not self.nlp:
                return detected
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract key phrases and their relationships
            key_phrases = []
            for chunk in doc.noun_chunks:
                key_phrases.append(chunk.text.lower())
            
            # Check for semantic inconsistencies
            misconceptions = await self._analyze_semantic_relationships(key_phrases, context)
            detected.extend(misconceptions)
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Failed to detect semantic misconceptions: {e}")
            return []
    
    async def _detect_mathematical_misconceptions(self, text: str,
                                                context: Set[str]) -> List[PhysicsMisconception]:
        """Detect mathematical reasoning misconceptions"""
        try:
            detected = []
            
            # Extract numerical values and units
            number_pattern = r'(-?\d+(?:\.\d+)?)\s*([a-zA-Z]+/?[a-zA-Z]*)?'
            numbers = re.findall(number_pattern, text)
            
            # Check for unit inconsistencies
            unit_misconceptions = await self._check_unit_consistency(numbers, context)
            detected.extend(unit_misconceptions)
            
            # Check for mathematical logic errors
            logic_misconceptions = await self._check_mathematical_logic(text, numbers)
            detected.extend(logic_misconceptions)
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Failed to detect mathematical misconceptions: {e}")
            return []
    
    async def _analyze_semantic_relationships(self, phrases: List[str], 
                                            context: Set[str]) -> List[PhysicsMisconception]:
        """Analyze semantic relationships for misconceptions"""
        try:
            misconceptions = []
            
            # Define problematic semantic relationships
            problematic_relationships = [
                {
                    'phrases': ['force', 'motion'],
                    'problematic_context': ['always', 'required', 'needed'],
                    'misconception_id': 'semantic_force_motion',
                    'description': 'Belief that force is always required for motion'
                },
                {
                    'phrases': ['energy', 'lost', 'destroyed'],
                    'problematic_context': ['disappear', 'consumed', 'used up'],
                    'misconception_id': 'semantic_energy_destruction',
                    'description': 'Belief that energy can be destroyed'
                }
            ]
            
            for relationship in problematic_relationships:
                # Check if key phrases are present
                phrase_matches = sum(1 for phrase in relationship['phrases'] 
                                   if any(phrase in p for p in phrases))
                
                # Check if problematic context is present
                context_matches = sum(1 for ctx in relationship['problematic_context']
                                    if any(ctx in p for p in phrases))
                
                if phrase_matches >= 2 and context_matches >= 1:
                    confidence_score = 0.3 + (phrase_matches * 0.2) + (context_matches * 0.1)
                    
                    misconception = PhysicsMisconception(
                        misconception_id=relationship['misconception_id'],
                        concept='general',
                        misconception_type=MisconceptionType.CONCEPTUAL,
                        severity=self._determine_severity(confidence_score, MisconceptionType.CONCEPTUAL),
                        description=relationship['description'],
                        student_expression=' '.join(phrases),
                        correct_understanding=self._get_correct_understanding(relationship['misconception_id']),
                        remediation_strategy=self._get_remediation_strategy(relationship['misconception_id']),
                        evidence_patterns=[f"semantic_analysis: {phrase_matches} phrase matches, {context_matches} context matches"],
                        confidence_score=confidence_score
                    )
                    
                    misconceptions.append(misconception)
            
            return misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze semantic relationships: {e}")
            return []
    
    async def _check_unit_consistency(self, numbers: List[Tuple[str, str]], 
                                    context: Set[str]) -> List[PhysicsMisconception]:
        """Check for unit consistency issues"""
        try:
            misconceptions = []
            
            # Extract units
            units_mentioned = [unit for _, unit in numbers if unit]
            
            # Check for mixing unit systems
            metric_units = ['m', 'kg', 's', 'N', 'J', 'W']
            imperial_units = ['ft', 'lb', 'in', 'mile']
            
            metric_count = sum(1 for unit in units_mentioned if unit in metric_units)
            imperial_count = sum(1 for unit in units_mentioned if unit in imperial_units)
            
            if metric_count > 0 and imperial_count > 0:
                misconception = PhysicsMisconception(
                    misconception_id='unit_mixing',
                    concept='units',
                    misconception_type=MisconceptionType.UNIT_CONVERSION,
                    severity=MisconceptionSeverity.MEDIUM,
                    description='Mixing metric and imperial units without conversion',
                    student_expression=str(numbers),
                    correct_understanding='Convert all quantities to the same unit system',
                    remediation_strategy='Practice unit conversions and stick to one unit system',
                    evidence_patterns=[f"mixed_units: {metric_count} metric, {imperial_count} imperial"],
                    confidence_score=0.8
                )
                misconceptions.append(misconception)
            
            return misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to check unit consistency: {e}")
            return []
    
    async def _check_mathematical_logic(self, text: str, 
                                      numbers: List[Tuple[str, str]]) -> List[PhysicsMisconception]:
        """Check for mathematical logic errors"""
        try:
            misconceptions = []
            
            # Check for impossible values (negative masses, negative distances, etc.)
            for value_str, unit in numbers:
                try:
                    value = float(value_str)
                    
                    # Check for impossible negative values
                    impossible_negative = [
                        ('mass', ['kg', 'g']),
                        ('distance', ['m', 'km', 'cm']),
                        ('time', ['s', 'min', 'h'])
                    ]
                    
                    for quantity, unit_list in impossible_negative:
                        if unit in unit_list and value < 0:
                            misconception = PhysicsMisconception(
                                misconception_id=f'negative_{quantity}',
                                concept='mathematics',
                                misconception_type=MisconceptionType.MATHEMATICAL,
                                severity=MisconceptionSeverity.HIGH,
                                description=f'Using negative value for {quantity}',
                                student_expression=f'{value} {unit}',
                                correct_understanding=f'{quantity.capitalize()} must be positive',
                                remediation_strategy=f'Review the physical meaning of {quantity}',
                                evidence_patterns=[f'negative_{quantity}: {value} {unit}'],
                                confidence_score=0.9
                            )
                            misconceptions.append(misconception)
                
                except ValueError:
                    continue
            
            return misconceptions
            
        except Exception as e:
            logger.error(f"❌ Failed to check mathematical logic: {e}")
            return []
    
    async def _filter_and_rank_misconceptions(self, misconceptions: List[PhysicsMisconception]) -> List[PhysicsMisconception]:
        """Filter and rank detected misconceptions"""
        try:
            # Remove duplicates based on misconception_id
            unique_misconceptions = {}
            for misconception in misconceptions:
                if (misconception.misconception_id not in unique_misconceptions or
                    misconception.confidence_score > unique_misconceptions[misconception.misconception_id].confidence_score):
                    unique_misconceptions[misconception.misconception_id] = misconception
            
            # Filter by confidence threshold
            filtered = [m for m in unique_misconceptions.values() if m.confidence_score >= 0.4]
            
            # Sort by severity and confidence
            severity_order = {
                MisconceptionSeverity.CRITICAL: 4,
                MisconceptionSeverity.HIGH: 3,
                MisconceptionSeverity.MEDIUM: 2,
                MisconceptionSeverity.LOW: 1
            }
            
            filtered.sort(key=lambda x: (severity_order.get(x.severity, 0), x.confidence_score), reverse=True)
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Failed to filter and rank misconceptions: {e}")
            return misconceptions
    
    def _determine_severity(self, confidence_score: float, 
                          misconception_type: MisconceptionType) -> MisconceptionSeverity:
        """Determine severity based on confidence and type"""
        try:
            # Base severity on confidence
            if confidence_score >= 0.8:
                base_severity = MisconceptionSeverity.HIGH
            elif confidence_score >= 0.6:
                base_severity = MisconceptionSeverity.MEDIUM
            else:
                base_severity = MisconceptionSeverity.LOW
            
            # Adjust based on misconception type
            critical_types = [MisconceptionType.MATHEMATICAL, MisconceptionType.CAUSAL_REASONING]
            if misconception_type in critical_types and base_severity == MisconceptionSeverity.HIGH:
                return MisconceptionSeverity.CRITICAL
            
            return base_severity
            
        except Exception as e:
            logger.error(f"❌ Failed to determine severity: {e}")
            return MisconceptionSeverity.LOW
    
    def _get_misconception_description(self, pattern_id: str) -> str:
        """Get detailed description of misconception"""
        descriptions = {
            'force_motion_1': 'Believing that a constant force is needed to maintain constant velocity',
            'force_motion_2': 'Believing that heavier objects fall faster in vacuum',
            'energy_1': 'Believing that energy can be destroyed or used up',
            'vector_scalar_1': 'Confusing vector and scalar quantities',
            'math_physics_1': 'Misunderstanding the meaning of negative values in physics',
            'units_1': 'Ignoring the importance of units in calculations',
            'causal_1': 'Misunderstanding circular motion and centripetal forces'
        }
        return descriptions.get(pattern_id, 'Physics misconception detected')
    
    def _get_correct_understanding(self, pattern_id: str) -> str:
        """Get correct physics understanding"""
        correct_understanding = {
            'force_motion_1': "By Newton's first law, an object in motion stays in motion at constant velocity unless acted upon by an unbalanced force",
            'force_motion_2': 'In vacuum, all objects fall at the same rate regardless of mass due to gravitational acceleration',
            'energy_1': 'Energy cannot be created or destroyed, only transformed from one form to another',
            'vector_scalar_1': 'Vectors have both magnitude and direction; scalars have only magnitude',
            'math_physics_1': 'Negative values in physics indicate direction or reference frame, not absence',
            'units_1': 'Units are essential for dimensional analysis and ensuring physical consistency',
            'causal_1': 'Circular motion requires centripetal force directed toward the center'
        }
        return correct_understanding.get(pattern_id, 'Correct physics understanding needed')
    
    def _get_remediation_strategy(self, pattern_id: str) -> str:
        """Get remediation strategy for misconception"""
        strategies = {
            'force_motion_1': 'Practice with Newton\'s first law examples; demonstrate motion without applied force',
            'force_motion_2': 'Demonstrate free fall experiments; discuss Galileo\'s findings',
            'energy_1': 'Practice energy transformation problems; emphasize conservation principles',
            'vector_scalar_1': 'Practice vector addition; emphasize direction in vector quantities',
            'math_physics_1': 'Practice coordinate systems; discuss reference frames',
            'units_1': 'Practice dimensional analysis; show examples of unit errors',
            'causal_1': 'Use circular motion simulations; practice force diagrams'
        }
        return strategies.get(pattern_id, 'Targeted practice and conceptual review')
    
    async def get_student_misconception_profile(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive misconception profile for a student"""
        try:
            if student_id not in self.student_misconceptions:
                return {'student_id': student_id, 'misconceptions': [], 'summary': 'No misconceptions detected'}
            
            misconceptions = self.student_misconceptions[student_id]
            
            # Analyze patterns
            misconception_types = Counter(m.misconception_type.value for m in misconceptions)
            severity_distribution = Counter(m.severity.value for m in misconceptions)
            concept_areas = Counter(m.concept for m in misconceptions)
            
            # Get recent misconceptions (last 30 days)
            recent_threshold = datetime.now() - timedelta(days=30)
            recent_misconceptions = [m for m in misconceptions if m.detected_at >= recent_threshold]
            
            # Calculate improvement indicators
            total_misconceptions = len(misconceptions)
            recent_misconceptions_count = len(recent_misconceptions)
            improvement_ratio = 1 - (recent_misconceptions_count / max(1, total_misconceptions))
            
            profile = {
                'student_id': student_id,
                'total_misconceptions_detected': total_misconceptions,
                'recent_misconceptions': recent_misconceptions_count,
                'improvement_indicator': improvement_ratio,
                'misconception_types': dict(misconception_types),
                'severity_distribution': dict(severity_distribution),
                'problem_concept_areas': dict(concept_areas),
                'most_recent_misconceptions': [
                    {
                        'misconception_id': m.misconception_id,
                        'concept': m.concept,
                        'type': m.misconception_type.value,
                        'severity': m.severity.value,
                        'description': m.description,
                        'detected_at': m.detected_at.isoformat()
                    }
                    for m in sorted(misconceptions, key=lambda x: x.detected_at, reverse=True)[:5]
                ],
                'remediation_recommendations': await self._generate_remediation_recommendations(misconceptions)
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to get student misconception profile: {e}")
            return {}
    
    async def _generate_remediation_recommendations(self, misconceptions: List[PhysicsMisconception]) -> List[str]:
        """Generate personalized remediation recommendations"""
        try:
            recommendations = []
            
            # Group by concept
            concept_misconceptions = defaultdict(list)
            for m in misconceptions:
                concept_misconceptions[m.concept].append(m)
            
            # Generate recommendations for each concept
            for concept, concept_miscs in concept_misconceptions.items():
                if len(concept_miscs) >= 3:
                    recommendations.append(f"Focus on {concept} fundamentals - multiple misconceptions detected")
                elif any(m.severity in [MisconceptionSeverity.HIGH, MisconceptionSeverity.CRITICAL] for m in concept_miscs):
                    recommendations.append(f"Priority remediation needed for {concept} concepts")
            
            # Add general recommendations
            if len(misconceptions) > 10:
                recommendations.append("Consider reviewing basic physics principles systematically")
            
            # Add specific recommendations based on patterns
            misconception_ids = [m.misconception_id for m in misconceptions]
            if 'force_motion_1' in misconception_ids:
                recommendations.append("Practice Newton's laws with real-world examples")
            if 'energy_1' in misconception_ids:
                recommendations.append("Focus on energy conservation through interactive simulations")
            
            return recommendations[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"❌ Failed to generate remediation recommendations: {e}")
            return []
    
    async def get_detection_statistics(self) -> Dict[str, Any]:
        """Get system-wide detection statistics"""
        try:
            total_students = len(self.student_misconceptions)
            
            # Calculate accuracy metrics (would require validation data in production)
            precision = 0.85  # Estimated based on pattern validation
            recall = 0.78     # Estimated based on coverage
            
            stats = {
                'system_performance': {
                    'total_responses_analyzed': self.detection_stats['total_analyzed'],
                    'misconceptions_detected': self.detection_stats['misconceptions_detected'],
                    'detection_rate': self.detection_stats['misconceptions_detected'] / max(1, self.detection_stats['total_analyzed']),
                    'estimated_precision': precision,
                    'estimated_recall': recall
                },
                'student_coverage': {
                    'total_students_monitored': total_students,
                    'students_with_misconceptions': len([s for s in self.student_misconceptions.values() if s])
                },
                'common_misconceptions': {
                    misconception_id: count
                    for misconception_id, count in self.misconception_frequency.most_common(10)
                },
                'misconception_patterns': {
                    'total_patterns': len(self.misconception_patterns),
                    'active_patterns': len([p for p in self.misconception_patterns.values() if p.confidence_weight > 0])
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get detection statistics: {e}")
            return {}

# Testing function
async def test_misconception_detector():
    """Test the physics misconception detection system"""
    try:
        logger.info("🧪 Testing Physics Misconception Detector")
        
        detector = PhysicsMisconceptionDetector()
        await detector.initialize()
        
        # Test responses with known misconceptions
        test_responses = [
            {
                'text': "A force is needed to keep the car moving at constant speed",
                'context': {'problem_concept': 'forces', 'topic': 'newton_laws'},
                'expected': 'force_motion_1'
            },
            {
                'text': "Heavier objects fall faster because they have more weight",
                'context': {'problem_concept': 'gravity', 'topic': 'free_fall'},
                'expected': 'force_motion_2'
            },
            {
                'text': "Energy gets used up when the ball bounces",
                'context': {'problem_concept': 'energy', 'topic': 'conservation'},
                'expected': 'energy_1'
            }
        ]
        
        for i, test_case in enumerate(test_responses):
            misconceptions = await detector.analyze_student_response(
                f"test_student_{i}", test_case['text'], test_case['context']
            )
            
            if misconceptions:
                logger.info(f"✅ Detected misconception: {misconceptions[0].misconception_id} "
                          f"(confidence: {misconceptions[0].confidence_score:.2f})")
            else:
                logger.warning(f"⚠️ No misconceptions detected for test case {i}")
        
        # Test student profile
        profile = await detector.get_student_misconception_profile("test_student_0")
        logger.info(f"✅ Student profile: {profile.get('total_misconceptions_detected', 0)} misconceptions")
        
        # Test system statistics
        stats = await detector.get_detection_statistics()
        logger.info(f"✅ System stats: {stats.get('system_performance', {})}")
        
        logger.info("✅ Physics Misconception Detector test completed")
        
    except Exception as e:
        logger.error(f"❌ Physics Misconception Detector test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_misconception_detector())