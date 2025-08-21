#!/usr/bin/env python3
"""
Multi-Modal Learning Analytics - Phase 6
Combines text analysis, diagram understanding, and interaction data to provide
comprehensive insights into student learning patterns and educational effectiveness.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import warnings
import base64
from pathlib import Path
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    TEXT = "text"
    DIAGRAM = "diagram"
    INTERACTION = "interaction"
    AUDIO = "audio"
    VIDEO = "video"

class LearningPhase(Enum):
    INITIAL_LEARNING = "initial_learning"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    REVIEW = "review"
    HELP_SEEKING = "help_seeking"

class EngagementLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISENGAGED = "disengaged"

class ConceptDifficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    VERY_HARD = "very_hard"

@dataclass
class ModalityData:
    """Data from a specific modality"""
    modality_type: ModalityType
    content: Any  # Raw content (text, image array, interaction logs, etc.)
    features: np.ndarray  # Extracted features
    embeddings: np.ndarray  # Learned embeddings
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0

@dataclass
class LearningSession:
    """A complete learning session with multi-modal data"""
    session_id: str
    student_id: str
    topic: str
    start_time: datetime
    end_time: datetime
    modality_data: List[ModalityData] = field(default_factory=list)
    learning_outcomes: Dict[str, float] = field(default_factory=dict)
    engagement_metrics: Dict[str, float] = field(default_factory=dict)
    difficulty_progression: List[float] = field(default_factory=list)

@dataclass
class LearningPattern:
    """Identified learning pattern across modalities"""
    pattern_id: str
    pattern_type: str
    description: str
    modalities_involved: List[ModalityType]
    frequency: float
    effectiveness: float
    student_segments: List[str]
    physics_concepts: List[str]
    recommendations: List[str]

@dataclass
class MultiModalInsight:
    """Insight derived from multi-modal analysis"""
    insight_id: str
    insight_type: str
    description: str
    evidence: Dict[str, Any]
    confidence: float
    actionable_recommendations: List[str]
    affected_students: List[str]
    physics_concepts: List[str]
    temporal_patterns: Dict[str, Any]

class ModalityFeatureExtractor:
    """Extract features from different modalities"""
    
    def __init__(self):
        self.text_tokenizer = None
        self.text_model = None
        self.interaction_features = [
            'click_frequency', 'dwell_time', 'scroll_behavior',
            'help_requests', 'submission_attempts', 'navigation_pattern'
        ]
    
    async def initialize(self):
        """Initialize feature extractors"""
        try:
            logger.info("🔧 Initializing modality feature extractors")
            
            # Initialize text processing
            self.text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.text_model = AutoModel.from_pretrained('distilbert-base-uncased')
            
            logger.info("✅ Feature extractors initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature extractors: {e}")
            return False
    
    def extract_text_features(self, text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from text content"""
        try:
            if not text or not self.text_tokenizer:
                return np.zeros(768), np.zeros(768)
            
            # Tokenize and get embeddings
            inputs = self.text_tokenizer(text, return_tensors='pt', 
                                       max_length=512, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                
            # Use CLS token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            # Extract basic text features
            features = np.array([
                len(text),  # Text length
                len(text.split()),  # Word count
                len([w for w in text.split() if len(w) > 6]),  # Complex words
                text.count('?'),  # Questions
                text.count('!'),  # Exclamations
                len([c for c in text if c.isupper()]) / len(text) if text else 0,  # Uppercase ratio
                text.count('physics'),  # Physics mentions
                text.count('formula'),  # Formula mentions
                text.count('equation'),  # Equation mentions
                len([w for w in text.lower().split() if w in ['difficult', 'hard', 'confused']]),  # Difficulty words
            ])
            
            # Pad features to match embedding dimension
            if len(features) < 768:
                features = np.pad(features, (0, 768 - len(features)), 'constant')
            else:
                features = features[:768]
            
            return features, embeddings
            
        except Exception as e:
            logger.error(f"❌ Text feature extraction failed: {e}")
            return np.zeros(768), np.zeros(768)
    
    def extract_diagram_features(self, image_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from physics diagrams"""
        try:
            if image_data is None or image_data.size == 0:
                return np.zeros(768), np.zeros(768)
            
            # Convert to grayscale if needed
            if len(image_data.shape) == 3:
                gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_data
            
            # Extract basic visual features
            height, width = gray.shape
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Contour analysis
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            num_contours = len(contours)
            
            # Shape analysis
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=0, maxRadius=0)
            num_circles = len(circles[0]) if circles is not None else 0
            
            # Line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                  minLineLength=50, maxLineGap=10)
            num_lines = len(lines) if lines is not None else 0
            
            # Text region detection (simplified)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            text_contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            potential_text_regions = len([c for c in text_contours if cv2.contourArea(c) > 100])
            
            # Complexity metrics
            pixel_variance = np.var(gray)
            brightness_mean = np.mean(gray)
            
            # Physics-specific features
            aspect_ratio = width / height
            diagonal_dominance = self._detect_diagonal_lines(edges)
            symmetry_score = self._calculate_symmetry(gray)
            
            features = np.array([
                width, height, aspect_ratio,
                edge_density, num_contours, num_circles, num_lines,
                potential_text_regions, pixel_variance, brightness_mean,
                diagonal_dominance, symmetry_score
            ])
            
            # Create embeddings using histogram features
            hist_features = []
            
            # Intensity histogram
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist_features.extend(hist.flatten())
            
            # Edge orientation histogram
            if edges.any():
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                orientation = np.arctan2(sobel_y, sobel_x)
                orientation_hist, _ = np.histogram(orientation, bins=32)
                hist_features.extend(orientation_hist)
            else:
                hist_features.extend(np.zeros(32))
            
            embeddings = np.array(hist_features)
            
            # Ensure consistent dimensions
            if len(features) < 768:
                features = np.pad(features, (0, 768 - len(features)), 'constant')
            else:
                features = features[:768]
                
            if len(embeddings) < 768:
                embeddings = np.pad(embeddings, (0, 768 - len(embeddings)), 'constant')
            else:
                embeddings = embeddings[:768]
            
            return features, embeddings
            
        except Exception as e:
            logger.error(f"❌ Diagram feature extraction failed: {e}")
            return np.zeros(768), np.zeros(768)
    
    def extract_interaction_features(self, interaction_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from interaction logs"""
        try:
            if not interaction_data:
                return np.zeros(768), np.zeros(768)
            
            # Extract temporal features
            session_duration = interaction_data.get('session_duration', 0)
            total_clicks = interaction_data.get('total_clicks', 0)
            total_keystrokes = interaction_data.get('total_keystrokes', 0)
            page_views = interaction_data.get('page_views', 0)
            
            # Calculate rates
            click_rate = total_clicks / max(session_duration, 1)
            keystroke_rate = total_keystrokes / max(session_duration, 1)
            
            # Extract navigation patterns
            navigation_events = interaction_data.get('navigation_events', [])
            unique_pages = len(set(nav.get('page', '') for nav in navigation_events))
            navigation_entropy = self._calculate_navigation_entropy(navigation_events)
            
            # Extract help-seeking behavior
            help_requests = interaction_data.get('help_requests', 0)
            hint_usage = interaction_data.get('hint_usage', 0)
            help_rate = help_requests / max(session_duration, 1)
            
            # Extract submission patterns
            submission_attempts = interaction_data.get('submission_attempts', 0)
            successful_submissions = interaction_data.get('successful_submissions', 0)
            success_rate = successful_submissions / max(submission_attempts, 1)
            
            # Extract pause patterns
            pause_events = interaction_data.get('pause_events', [])
            total_pause_time = sum(p.get('duration', 0) for p in pause_events)
            pause_frequency = len(pause_events)
            
            # Extract scroll behavior
            scroll_events = interaction_data.get('scroll_events', [])
            total_scroll_distance = sum(s.get('distance', 0) for s in scroll_events)
            scroll_velocity_variance = np.var([s.get('velocity', 0) for s in scroll_events]) if scroll_events else 0
            
            # Extract focus patterns
            focus_events = interaction_data.get('focus_events', [])
            focus_switches = len(focus_events)
            avg_focus_duration = np.mean([f.get('duration', 0) for f in focus_events]) if focus_events else 0
            
            features = np.array([
                session_duration, total_clicks, total_keystrokes, page_views,
                click_rate, keystroke_rate, unique_pages, navigation_entropy,
                help_requests, hint_usage, help_rate, submission_attempts,
                successful_submissions, success_rate, total_pause_time,
                pause_frequency, total_scroll_distance, scroll_velocity_variance,
                focus_switches, avg_focus_duration
            ])
            
            # Create temporal embeddings
            temporal_features = self._extract_temporal_patterns(interaction_data)
            
            # Combine features for embeddings
            embeddings = np.concatenate([features, temporal_features])
            
            # Ensure consistent dimensions
            if len(features) < 768:
                features = np.pad(features, (0, 768 - len(features)), 'constant')
            else:
                features = features[:768]
                
            if len(embeddings) < 768:
                embeddings = np.pad(embeddings, (0, 768 - len(embeddings)), 'constant')
            else:
                embeddings = embeddings[:768]
            
            return features, embeddings
            
        except Exception as e:
            logger.error(f"❌ Interaction feature extraction failed: {e}")
            return np.zeros(768), np.zeros(768)
    
    def _detect_diagonal_lines(self, edges: np.ndarray) -> float:
        """Detect diagonal lines in image (useful for physics diagrams)"""
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                  minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return 0.0
            
            diagonal_count = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Check if line is diagonal (not horizontal or vertical)
                if 20 < abs(angle) < 70 or 110 < abs(angle) < 160:
                    diagonal_count += 1
            
            return diagonal_count / len(lines)
            
        except Exception:
            return 0.0
    
    def _calculate_symmetry(self, image: np.ndarray) -> float:
        """Calculate image symmetry score"""
        try:
            height, width = image.shape
            
            # Vertical symmetry
            left_half = image[:, :width//2]
            right_half = np.fliplr(image[:, width//2:])
            
            # Resize to match if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            vertical_symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            
            # Horizontal symmetry
            top_half = image[:height//2, :]
            bottom_half = np.flipud(image[height//2:, :])
            
            min_height = min(top_half.shape[0], bottom_half.shape[0])
            top_half = top_half[:min_height, :]
            bottom_half = bottom_half[:min_height, :]
            
            horizontal_symmetry = np.corrcoef(top_half.flatten(), bottom_half.flatten())[0, 1]
            
            # Return average symmetry, handling NaN values
            symmetries = [s for s in [vertical_symmetry, horizontal_symmetry] if not np.isnan(s)]
            return np.mean(symmetries) if symmetries else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_navigation_entropy(self, navigation_events: List[Dict]) -> float:
        """Calculate entropy of navigation patterns"""
        try:
            if not navigation_events:
                return 0.0
            
            pages = [nav.get('page', 'unknown') for nav in navigation_events]
            page_counts = Counter(pages)
            total_events = len(navigation_events)
            
            entropy = 0.0
            for count in page_counts.values():
                probability = count / total_events
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def _extract_temporal_patterns(self, interaction_data: Dict[str, Any]) -> np.ndarray:
        """Extract temporal patterns from interactions"""
        try:
            # Extract timestamps from various events
            all_events = []
            
            for event_type in ['clicks', 'keystrokes', 'navigation_events', 'help_requests']:
                events = interaction_data.get(event_type, [])
                if isinstance(events, list):
                    for event in events:
                        if isinstance(event, dict) and 'timestamp' in event:
                            all_events.append(event['timestamp'])
            
            if not all_events:
                return np.zeros(748)  # 768 - 20 basic features
            
            # Convert to time differences
            all_events.sort()
            time_diffs = np.diff(all_events) if len(all_events) > 1 else [0]
            
            # Calculate temporal features
            temporal_features = [
                np.mean(time_diffs),
                np.std(time_diffs),
                np.min(time_diffs) if time_diffs else 0,
                np.max(time_diffs) if time_diffs else 0,
                len(all_events),
                np.median(time_diffs) if time_diffs else 0
            ]
            
            # Create time series features (activity over time buckets)
            if len(all_events) > 0:
                session_start = min(all_events)
                session_end = max(all_events)
                session_duration = session_end - session_start
                
                if session_duration > 0:
                    num_buckets = 20
                    bucket_duration = session_duration / num_buckets
                    
                    activity_histogram = np.zeros(num_buckets)
                    for event_time in all_events:
                        bucket_idx = min(int((event_time - session_start) / bucket_duration), num_buckets - 1)
                        activity_histogram[bucket_idx] += 1
                    
                    temporal_features.extend(activity_histogram.tolist())
                else:
                    temporal_features.extend(np.zeros(20).tolist())
            else:
                temporal_features.extend(np.zeros(20).tolist())
            
            # Pad to ensure consistent length
            while len(temporal_features) < 748:
                temporal_features.append(0.0)
            
            return np.array(temporal_features[:748])
            
        except Exception as e:
            logger.error(f"❌ Temporal pattern extraction failed: {e}")
            return np.zeros(748)

class CrossModalFusionNetwork(nn.Module):
    """Neural network for fusing multi-modal features"""
    
    def __init__(self, modality_dims: Dict[str, int], fusion_dim: int = 256):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Individual modality encoders
        self.text_encoder = nn.Sequential(
            nn.Linear(modality_dims.get('text', 768), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fusion_dim),
            nn.ReLU()
        )
        
        self.diagram_encoder = nn.Sequential(
            nn.Linear(modality_dims.get('diagram', 768), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fusion_dim),
            nn.ReLU()
        )
        
        self.interaction_encoder = nn.Sequential(
            nn.Linear(modality_dims.get('interaction', 768), 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, fusion_dim),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=8, dropout=0.1)
        
        # Fusion layers
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Output heads for different tasks
        self.engagement_head = nn.Linear(128, 4)  # 4 engagement levels
        self.learning_outcome_head = nn.Linear(128, 1)  # Regression
        self.concept_difficulty_head = nn.Linear(128, 5)  # 5 difficulty levels
        
    def forward(self, text_features, diagram_features, interaction_features):
        """Forward pass through fusion network"""
        # Encode each modality
        text_encoded = self.text_encoder(text_features)
        diagram_encoded = self.diagram_encoder(diagram_features)
        interaction_encoded = self.interaction_encoder(interaction_features)
        
        # Stack for attention
        modality_stack = torch.stack([text_encoded, diagram_encoded, interaction_encoded], dim=0)
        
        # Apply cross-modal attention
        attended_features, _ = self.attention(modality_stack, modality_stack, modality_stack)
        
        # Flatten and concatenate
        fused_features = attended_features.flatten(start_dim=0)
        
        # Fusion network
        unified_representation = self.fusion_network(fused_features)
        
        # Task-specific outputs
        engagement_pred = self.engagement_head(unified_representation)
        learning_outcome_pred = self.learning_outcome_head(unified_representation)
        concept_difficulty_pred = self.concept_difficulty_head(unified_representation)
        
        return {
            'unified_representation': unified_representation,
            'engagement': engagement_pred,
            'learning_outcome': learning_outcome_pred,
            'concept_difficulty': concept_difficulty_pred
        }

class LearningPatternDetector:
    """Detect learning patterns across multiple modalities"""
    
    def __init__(self):
        self.pattern_templates = {
            'visual_learner': {
                'description': 'Student prefers visual content and diagrams',
                'indicators': {
                    'diagram_engagement': 0.7,
                    'text_to_diagram_ratio': 0.3,
                    'diagram_interaction_time': 0.8
                }
            },
            'help_seeking': {
                'description': 'Student frequently seeks help when struggling',
                'indicators': {
                    'help_request_frequency': 0.6,
                    'difficulty_correlation': 0.7,
                    'pre_submission_help': 0.5
                }
            },
            'rapid_learner': {
                'description': 'Student learns concepts quickly',
                'indicators': {
                    'concept_mastery_speed': 0.8,
                    'low_repetition_rate': 0.3,
                    'high_success_rate': 0.85
                }
            },
            'struggling_learner': {
                'description': 'Student has difficulty with concepts',
                'indicators': {
                    'multiple_attempts': 0.7,
                    'extended_time_per_problem': 0.8,
                    'frequent_help_requests': 0.6
                }
            },
            'concept_hopper': {
                'description': 'Student jumps between concepts without mastery',
                'indicators': {
                    'high_navigation_entropy': 0.7,
                    'low_time_per_concept': 0.4,
                    'incomplete_concept_coverage': 0.6
                }
            }
        }
    
    def detect_patterns(self, sessions: List[LearningSession]) -> List[LearningPattern]:
        """Detect learning patterns from multi-modal session data"""
        try:
            detected_patterns = []
            
            if not sessions:
                return detected_patterns
            
            # Group sessions by student
            student_sessions = defaultdict(list)
            for session in sessions:
                student_sessions[session.student_id].append(session)
            
            # Analyze patterns for each student
            for student_id, student_session_list in student_sessions.items():
                patterns = self._analyze_student_patterns(student_id, student_session_list)
                detected_patterns.extend(patterns)
            
            # Detect population-level patterns
            population_patterns = self._analyze_population_patterns(sessions)
            detected_patterns.extend(population_patterns)
            
            logger.info(f"🔍 Detected {len(detected_patterns)} learning patterns")
            return detected_patterns
            
        except Exception as e:
            logger.error(f"❌ Pattern detection failed: {e}")
            return []
    
    def _analyze_student_patterns(self, student_id: str, sessions: List[LearningSession]) -> List[LearningPattern]:
        """Analyze patterns for individual student"""
        try:
            patterns = []
            
            if not sessions:
                return patterns
            
            # Calculate student metrics across all sessions
            student_metrics = self._calculate_student_metrics(sessions)
            
            # Check each pattern template
            for pattern_name, template in self.pattern_templates.items():
                match_score = self._calculate_pattern_match(student_metrics, template['indicators'])
                
                if match_score > 0.6:  # Threshold for pattern detection
                    pattern = LearningPattern(
                        pattern_id=f"{student_id}_{pattern_name}",
                        pattern_type=pattern_name,
                        description=template['description'],
                        modalities_involved=[ModalityType.TEXT, ModalityType.DIAGRAM, ModalityType.INTERACTION],
                        frequency=match_score,
                        effectiveness=self._calculate_pattern_effectiveness(sessions, pattern_name),
                        student_segments=[student_id],
                        physics_concepts=list(set(session.topic for session in sessions)),
                        recommendations=self._generate_pattern_recommendations(pattern_name, match_score)
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Student pattern analysis failed: {e}")
            return []
    
    def _analyze_population_patterns(self, sessions: List[LearningSession]) -> List[LearningPattern]:
        """Analyze population-level patterns"""
        try:
            patterns = []
            
            # Group by topic/concept
            topic_sessions = defaultdict(list)
            for session in sessions:
                topic_sessions[session.topic].append(session)
            
            # Analyze patterns per topic
            for topic, topic_session_list in topic_sessions.items():
                topic_patterns = self._detect_topic_specific_patterns(topic, topic_session_list)
                patterns.extend(topic_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Population pattern analysis failed: {e}")
            return []
    
    def _calculate_student_metrics(self, sessions: List[LearningSession]) -> Dict[str, float]:
        """Calculate comprehensive metrics for a student"""
        try:
            metrics = {}
            
            if not sessions:
                return metrics
            
            # Text-related metrics
            text_data = [md for session in sessions for md in session.modality_data 
                        if md.modality_type == ModalityType.TEXT]
            
            if text_data:
                avg_text_length = np.mean([len(str(td.content)) for td in text_data])
                metrics['avg_text_length'] = avg_text_length
                metrics['text_engagement'] = np.mean([td.confidence for td in text_data])
            
            # Diagram-related metrics
            diagram_data = [md for session in sessions for md in session.modality_data 
                           if md.modality_type == ModalityType.DIAGRAM]
            
            if diagram_data:
                metrics['diagram_engagement'] = np.mean([dd.confidence for dd in diagram_data])
                metrics['diagram_interaction_time'] = len(diagram_data) / len(sessions)
            
            # Interaction metrics
            interaction_data = [md for session in sessions for md in session.modality_data 
                              if md.modality_type == ModalityType.INTERACTION]
            
            if interaction_data:
                help_requests = []
                navigation_entropies = []
                
                for id_data in interaction_data:
                    if isinstance(id_data.content, dict):
                        help_requests.append(id_data.content.get('help_requests', 0))
                        nav_events = id_data.content.get('navigation_events', [])
                        if nav_events:
                            pages = [nav.get('page', '') for nav in nav_events]
                            page_counts = Counter(pages)
                            total = len(nav_events)
                            entropy = -sum((count/total) * np.log2(count/total) 
                                         for count in page_counts.values() if count > 0)
                            navigation_entropies.append(entropy)
                
                if help_requests:
                    metrics['help_request_frequency'] = np.mean(help_requests)
                if navigation_entropies:
                    metrics['high_navigation_entropy'] = np.mean(navigation_entropies)
            
            # Learning outcome metrics
            all_outcomes = []
            for session in sessions:
                all_outcomes.extend(session.learning_outcomes.values())
            
            if all_outcomes:
                metrics['high_success_rate'] = np.mean(all_outcomes)
                metrics['concept_mastery_speed'] = 1.0 / (np.std(all_outcomes) + 0.1)  # Inverse of variability
            
            # Session-level metrics
            session_durations = [(session.end_time - session.start_time).total_seconds() / 60 
                               for session in sessions]
            if session_durations:
                metrics['extended_time_per_problem'] = np.mean(session_durations) / 30  # Normalized to 30min baseline
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Student metrics calculation failed: {e}")
            return {}
    
    def _calculate_pattern_match(self, student_metrics: Dict[str, float], 
                               pattern_indicators: Dict[str, float]) -> float:
        """Calculate how well student metrics match a pattern"""
        try:
            matches = []
            
            for indicator, threshold in pattern_indicators.items():
                student_value = student_metrics.get(indicator, 0.0)
                
                # Calculate match score (closer to threshold = higher score)
                if indicator.startswith('low_'):
                    # For "low" indicators, lower values are better matches
                    match = max(0, 1.0 - abs(student_value - threshold))
                elif indicator.startswith('high_'):
                    # For "high" indicators, higher values are better matches
                    match = min(1.0, student_value / threshold) if threshold > 0 else 0
                else:
                    # For regular indicators, closeness to threshold matters
                    match = max(0, 1.0 - abs(student_value - threshold))
                
                matches.append(match)
            
            return np.mean(matches) if matches else 0.0
            
        except Exception as e:
            logger.error(f"❌ Pattern match calculation failed: {e}")
            return 0.0
    
    def _calculate_pattern_effectiveness(self, sessions: List[LearningSession], 
                                       pattern_name: str) -> float:
        """Calculate effectiveness of detected pattern"""
        try:
            # Use learning outcomes as effectiveness measure
            all_outcomes = []
            for session in sessions:
                all_outcomes.extend(session.learning_outcomes.values())
            
            if all_outcomes:
                return np.mean(all_outcomes)
            else:
                return 0.5  # Neutral effectiveness
                
        except Exception as e:
            logger.error(f"❌ Pattern effectiveness calculation failed: {e}")
            return 0.5
    
    def _generate_pattern_recommendations(self, pattern_name: str, match_score: float) -> List[str]:
        """Generate recommendations based on detected pattern"""
        try:
            recommendations = []
            
            if pattern_name == 'visual_learner':
                recommendations = [
                    "Provide more visual explanations and diagrams",
                    "Use interactive simulations for concept illustration",
                    "Incorporate concept maps and visual organizers"
                ]
            elif pattern_name == 'help_seeking':
                recommendations = [
                    "Provide immediate feedback and hints",
                    "Create peer support opportunities",
                    "Implement adaptive scaffolding"
                ]
            elif pattern_name == 'rapid_learner':
                recommendations = [
                    "Provide advanced or extension problems",
                    "Encourage peer tutoring opportunities",
                    "Offer accelerated learning paths"
                ]
            elif pattern_name == 'struggling_learner':
                recommendations = [
                    "Break down concepts into smaller steps",
                    "Provide additional practice opportunities",
                    "Implement mastery-based progression"
                ]
            elif pattern_name == 'concept_hopper':
                recommendations = [
                    "Implement prerequisite checking",
                    "Provide guided learning sequences",
                    "Add mastery indicators for concepts"
                ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return ["Provide personalized learning support"]
    
    def _detect_topic_specific_patterns(self, topic: str, sessions: List[LearningSession]) -> List[LearningPattern]:
        """Detect patterns specific to a physics topic"""
        try:
            patterns = []
            
            # Calculate topic-level metrics
            success_rates = []
            engagement_scores = []
            
            for session in sessions:
                if session.learning_outcomes:
                    success_rates.extend(session.learning_outcomes.values())
                if session.engagement_metrics:
                    engagement_scores.extend(session.engagement_metrics.values())
            
            if success_rates:
                avg_success = np.mean(success_rates)
                success_variance = np.var(success_rates)
                
                # Detect if topic is consistently difficult
                if avg_success < 0.6 and success_variance < 0.1:
                    pattern = LearningPattern(
                        pattern_id=f"topic_{topic}_difficult",
                        pattern_type="difficult_topic",
                        description=f"Topic {topic} shows consistent difficulty across students",
                        modalities_involved=[ModalityType.TEXT, ModalityType.DIAGRAM, ModalityType.INTERACTION],
                        frequency=1.0 - avg_success,
                        effectiveness=avg_success,
                        student_segments=list(set(session.student_id for session in sessions)),
                        physics_concepts=[topic],
                        recommendations=[
                            f"Review teaching approach for {topic}",
                            "Provide additional practice materials",
                            "Consider prerequisite concept reinforcement"
                        ]
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Topic-specific pattern detection failed: {e}")
            return []

class MultiModalInsightGenerator:
    """Generate actionable insights from multi-modal analysis"""
    
    def __init__(self):
        self.insight_templates = {
            'engagement_drop': {
                'type': 'engagement',
                'description': 'Significant drop in student engagement detected',
                'threshold': 0.3
            },
            'modality_preference': {
                'type': 'learning_style',
                'description': 'Strong preference for specific learning modality',
                'threshold': 0.7
            },
            'concept_difficulty_spike': {
                'type': 'curriculum',
                'description': 'Concept shows unusually high difficulty',
                'threshold': 0.8
            },
            'successful_learning_path': {
                'type': 'best_practice',
                'description': 'Highly effective learning sequence identified',
                'threshold': 0.85
            }
        }
    
    def generate_insights(self, sessions: List[LearningSession], 
                         patterns: List[LearningPattern]) -> List[MultiModalInsight]:
        """Generate comprehensive insights from sessions and patterns"""
        try:
            insights = []
            
            # Generate engagement insights
            engagement_insights = self._analyze_engagement_trends(sessions)
            insights.extend(engagement_insights)
            
            # Generate modality preference insights
            modality_insights = self._analyze_modality_preferences(sessions)
            insights.extend(modality_insights)
            
            # Generate concept difficulty insights
            difficulty_insights = self._analyze_concept_difficulties(sessions)
            insights.extend(difficulty_insights)
            
            # Generate pattern-based insights
            pattern_insights = self._analyze_pattern_implications(patterns)
            insights.extend(pattern_insights)
            
            # Generate temporal insights
            temporal_insights = self._analyze_temporal_patterns(sessions)
            insights.extend(temporal_insights)
            
            logger.info(f"💡 Generated {len(insights)} multi-modal insights")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {e}")
            return []
    
    def _analyze_engagement_trends(self, sessions: List[LearningSession]) -> List[MultiModalInsight]:
        """Analyze engagement trends across modalities"""
        try:
            insights = []
            
            if not sessions:
                return insights
            
            # Group sessions by time periods
            sessions_by_week = defaultdict(list)
            for session in sessions:
                week = session.start_time.isocalendar()[1]
                sessions_by_week[week].append(session)
            
            # Calculate weekly engagement scores
            weekly_engagement = []
            for week in sorted(sessions_by_week.keys()):
                week_sessions = sessions_by_week[week]
                week_engagement = []
                
                for session in week_sessions:
                    if session.engagement_metrics:
                        week_engagement.extend(session.engagement_metrics.values())
                
                if week_engagement:
                    weekly_engagement.append(np.mean(week_engagement))
            
            # Detect significant drops
            if len(weekly_engagement) >= 2:
                for i in range(1, len(weekly_engagement)):
                    drop = weekly_engagement[i-1] - weekly_engagement[i]
                    if drop > 0.3:  # Significant drop threshold
                        insight = MultiModalInsight(
                            insight_id=f"engagement_drop_week_{i}",
                            insight_type="engagement_drop",
                            description=f"Engagement dropped by {drop:.1%} in week {i+1}",
                            evidence={
                                'previous_engagement': weekly_engagement[i-1],
                                'current_engagement': weekly_engagement[i],
                                'drop_magnitude': drop
                            },
                            confidence=min(0.9, drop * 2),
                            actionable_recommendations=[
                                "Review content difficulty for recent topics",
                                "Increase interactive elements in lessons",
                                "Check for external factors affecting student motivation"
                            ],
                            affected_students=list(set(s.student_id for s in sessions_by_week[i])),
                            physics_concepts=list(set(s.topic for s in sessions_by_week[i])),
                            temporal_patterns={'weekly_trend': weekly_engagement}
                        )
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Engagement trend analysis failed: {e}")
            return []
    
    def _analyze_modality_preferences(self, sessions: List[LearningSession]) -> List[MultiModalInsight]:
        """Analyze learning modality preferences"""
        try:
            insights = []
            
            # Group by student
            student_sessions = defaultdict(list)
            for session in sessions:
                student_sessions[session.student_id].append(session)
            
            for student_id, student_session_list in student_sessions.items():
                # Calculate modality engagement scores
                modality_scores = {
                    ModalityType.TEXT: [],
                    ModalityType.DIAGRAM: [],
                    ModalityType.INTERACTION: []
                }
                
                for session in student_session_list:
                    for modality_data in session.modality_data:
                        if modality_data.modality_type in modality_scores:
                            modality_scores[modality_data.modality_type].append(modality_data.confidence)
                
                # Calculate average scores
                avg_scores = {}
                for modality, scores in modality_scores.items():
                    if scores:
                        avg_scores[modality] = np.mean(scores)
                
                if len(avg_scores) >= 2:
                    # Find dominant modality
                    dominant_modality = max(avg_scores.keys(), key=lambda x: avg_scores[x])
                    dominant_score = avg_scores[dominant_modality]
                    
                    # Check if preference is strong enough
                    other_scores = [score for mod, score in avg_scores.items() if mod != dominant_modality]
                    if other_scores and dominant_score - max(other_scores) > 0.3:
                        insight = MultiModalInsight(
                            insight_id=f"modality_preference_{student_id}",
                            insight_type="modality_preference",
                            description=f"Student shows strong preference for {dominant_modality.value} learning",
                            evidence={
                                'dominant_modality': dominant_modality.value,
                                'dominant_score': dominant_score,
                                'modality_scores': {k.value: v for k, v in avg_scores.items()}
                            },
                            confidence=min(0.9, (dominant_score - max(other_scores)) * 2),
                            actionable_recommendations=[
                                f"Provide more {dominant_modality.value}-based content for this student",
                                "Gradually introduce other modalities to improve flexibility",
                                "Use preferred modality for introducing difficult concepts"
                            ],
                            affected_students=[student_id],
                            physics_concepts=list(set(s.topic for s in student_session_list)),
                            temporal_patterns={}
                        )
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Modality preference analysis failed: {e}")
            return []
    
    def _analyze_concept_difficulties(self, sessions: List[LearningSession]) -> List[MultiModalInsight]:
        """Analyze concept-specific difficulties"""
        try:
            insights = []
            
            # Group by topic/concept
            concept_sessions = defaultdict(list)
            for session in sessions:
                concept_sessions[session.topic].append(session)
            
            for concept, concept_session_list in concept_sessions.items():
                # Calculate success rates and engagement
                success_rates = []
                engagement_scores = []
                help_request_rates = []
                
                for session in concept_session_list:
                    if session.learning_outcomes:
                        success_rates.extend(session.learning_outcomes.values())
                    
                    if session.engagement_metrics:
                        engagement_scores.extend(session.engagement_metrics.values())
                    
                    # Extract help request rates from interaction data
                    for modality_data in session.modality_data:
                        if (modality_data.modality_type == ModalityType.INTERACTION and 
                            isinstance(modality_data.content, dict)):
                            help_requests = modality_data.content.get('help_requests', 0)
                            session_duration = modality_data.content.get('session_duration', 1)
                            help_request_rates.append(help_requests / session_duration)
                
                # Check for difficulty indicators
                if success_rates and engagement_scores:
                    avg_success = np.mean(success_rates)
                    avg_engagement = np.mean(engagement_scores)
                    avg_help_rate = np.mean(help_request_rates) if help_request_rates else 0
                    
                    # Define difficulty threshold
                    if avg_success < 0.5 or avg_engagement < 0.4 or avg_help_rate > 0.5:
                        difficulty_score = 1.0 - ((avg_success + avg_engagement) / 2 - avg_help_rate)
                        
                        insight = MultiModalInsight(
                            insight_id=f"concept_difficulty_{concept}",
                            insight_type="concept_difficulty_spike",
                            description=f"Concept '{concept}' shows high difficulty across multiple indicators",
                            evidence={
                                'success_rate': avg_success,
                                'engagement_score': avg_engagement,
                                'help_request_rate': avg_help_rate,
                                'difficulty_score': difficulty_score,
                                'student_count': len(set(s.student_id for s in concept_session_list))
                            },
                            confidence=min(0.9, difficulty_score),
                            actionable_recommendations=[
                                f"Review and simplify content for '{concept}'",
                                "Provide additional scaffolding and practice opportunities",
                                "Consider prerequisite concept reinforcement",
                                "Implement mastery-based progression for this concept"
                            ],
                            affected_students=list(set(s.student_id for s in concept_session_list)),
                            physics_concepts=[concept],
                            temporal_patterns={}
                        )
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Concept difficulty analysis failed: {e}")
            return []
    
    def _analyze_pattern_implications(self, patterns: List[LearningPattern]) -> List[MultiModalInsight]:
        """Generate insights from detected learning patterns"""
        try:
            insights = []
            
            # Analyze pattern frequency and effectiveness
            pattern_summary = defaultdict(list)
            for pattern in patterns:
                pattern_summary[pattern.pattern_type].append(pattern)
            
            for pattern_type, pattern_list in pattern_summary.items():
                if len(pattern_list) >= 3:  # Significant pattern occurrence
                    avg_effectiveness = np.mean([p.effectiveness for p in pattern_list])
                    all_affected_students = list(set(s for p in pattern_list for s in p.student_segments))
                    all_concepts = list(set(c for p in pattern_list for c in p.physics_concepts))
                    
                    insight = MultiModalInsight(
                        insight_id=f"pattern_insight_{pattern_type}",
                        insight_type="learning_pattern",
                        description=f"'{pattern_type}' pattern detected across {len(pattern_list)} instances",
                        evidence={
                            'pattern_frequency': len(pattern_list),
                            'average_effectiveness': avg_effectiveness,
                            'affected_student_count': len(all_affected_students),
                            'concept_coverage': len(all_concepts)
                        },
                        confidence=min(0.9, len(pattern_list) / 10),  # Scale confidence by frequency
                        actionable_recommendations=pattern_list[0].recommendations,  # Use first pattern's recommendations
                        affected_students=all_affected_students,
                        physics_concepts=all_concepts,
                        temporal_patterns={}
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Pattern implication analysis failed: {e}")
            return []
    
    def _analyze_temporal_patterns(self, sessions: List[LearningSession]) -> List[MultiModalInsight]:
        """Analyze temporal learning patterns"""
        try:
            insights = []
            
            if not sessions:
                return insights
            
            # Group sessions by time of day
            hour_groups = defaultdict(list)
            for session in sessions:
                hour = session.start_time.hour
                hour_groups[hour].append(session)
            
            # Calculate performance by hour
            hourly_performance = {}
            for hour, hour_sessions in hour_groups.items():
                performance_scores = []
                for session in hour_sessions:
                    if session.learning_outcomes:
                        performance_scores.extend(session.learning_outcomes.values())
                
                if performance_scores:
                    hourly_performance[hour] = np.mean(performance_scores)
            
            if len(hourly_performance) >= 3:
                # Find best and worst performance hours
                best_hour = max(hourly_performance.keys(), key=lambda x: hourly_performance[x])
                worst_hour = min(hourly_performance.keys(), key=lambda x: hourly_performance[x])
                
                performance_difference = hourly_performance[best_hour] - hourly_performance[worst_hour]
                
                if performance_difference > 0.2:  # Significant temporal effect
                    insight = MultiModalInsight(
                        insight_id="temporal_performance_pattern",
                        insight_type="temporal_pattern",
                        description=f"Performance varies significantly by time of day",
                        evidence={
                            'best_hour': best_hour,
                            'worst_hour': worst_hour,
                            'performance_difference': performance_difference,
                            'hourly_performance': hourly_performance
                        },
                        confidence=min(0.9, performance_difference * 3),
                        actionable_recommendations=[
                            f"Schedule challenging content during peak hours ({best_hour}:00)",
                            f"Provide additional support during low-performance hours ({worst_hour}:00)",
                            "Consider student time zone preferences for live sessions"
                        ],
                        affected_students=list(set(s.student_id for s in sessions)),
                        physics_concepts=list(set(s.topic for s in sessions)),
                        temporal_patterns={'hourly_performance': hourly_performance}
                    )
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Temporal pattern analysis failed: {e}")
            return []

class MultiModalLearningAnalytics:
    """Main orchestrator for multi-modal learning analytics"""
    
    def __init__(self):
        self.feature_extractor = ModalityFeatureExtractor()
        self.fusion_network = None
        self.pattern_detector = LearningPatternDetector()
        self.insight_generator = MultiModalInsightGenerator()
        
        # Analytics storage
        self.processed_sessions = []
        self.detected_patterns = []
        self.generated_insights = []
        
        # Configuration
        self.config = {
            'embedding_dim': 768,
            'fusion_dim': 256,
            'pattern_detection_threshold': 0.6,
            'insight_confidence_threshold': 0.5
        }
    
    async def initialize(self):
        """Initialize the multi-modal learning analytics system"""
        try:
            logger.info("🚀 Initializing Multi-Modal Learning Analytics")
            
            # Initialize feature extractor
            await self.feature_extractor.initialize()
            
            # Initialize fusion network
            modality_dims = {
                'text': self.config['embedding_dim'],
                'diagram': self.config['embedding_dim'],
                'interaction': self.config['embedding_dim']
            }
            
            self.fusion_network = CrossModalFusionNetwork(
                modality_dims, 
                self.config['fusion_dim']
            )
            
            logger.info("✅ Multi-Modal Learning Analytics initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Multi-Modal Learning Analytics: {e}")
            return False
    
    async def process_learning_session(self, session_data: Dict[str, Any]) -> LearningSession:
        """Process a complete learning session with multi-modal data"""
        try:
            session = LearningSession(
                session_id=session_data['session_id'],
                student_id=session_data['student_id'],
                topic=session_data['topic'],
                start_time=session_data['start_time'],
                end_time=session_data['end_time']
            )
            
            # Process each modality
            for modality_info in session_data.get('modalities', []):
                modality_data = await self._process_modality_data(modality_info)
                if modality_data:
                    session.modality_data.append(modality_data)
            
            # Calculate learning outcomes using fusion network
            if len(session.modality_data) >= 2:
                outcomes = await self._calculate_learning_outcomes(session)
                session.learning_outcomes = outcomes
            
            # Calculate engagement metrics
            engagement = self._calculate_engagement_metrics(session)
            session.engagement_metrics = engagement
            
            # Calculate difficulty progression
            difficulty = self._calculate_difficulty_progression(session)
            session.difficulty_progression = difficulty
            
            self.processed_sessions.append(session)
            logger.info(f"📊 Processed learning session {session.session_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Failed to process learning session: {e}")
            raise
    
    async def _process_modality_data(self, modality_info: Dict[str, Any]) -> Optional[ModalityData]:
        """Process data from a specific modality"""
        try:
            modality_type = ModalityType(modality_info['type'])
            content = modality_info['content']
            
            # Extract features based on modality type
            if modality_type == ModalityType.TEXT:
                features, embeddings = self.feature_extractor.extract_text_features(content)
            elif modality_type == ModalityType.DIAGRAM:
                features, embeddings = self.feature_extractor.extract_diagram_features(content)
            elif modality_type == ModalityType.INTERACTION:
                features, embeddings = self.feature_extractor.extract_interaction_features(content)
            else:
                logger.warning(f"⚠️ Unsupported modality type: {modality_type}")
                return None
            
            modality_data = ModalityData(
                modality_type=modality_type,
                content=content,
                features=features,
                embeddings=embeddings,
                metadata=modality_info.get('metadata', {}),
                confidence=modality_info.get('confidence', 1.0)
            )
            
            return modality_data
            
        except Exception as e:
            logger.error(f"❌ Modality data processing failed: {e}")
            return None
    
    async def _calculate_learning_outcomes(self, session: LearningSession) -> Dict[str, float]:
        """Calculate learning outcomes using fusion network"""
        try:
            outcomes = {}
            
            if not self.fusion_network or len(session.modality_data) < 2:
                return outcomes
            
            # Prepare modality features
            text_features = np.zeros(self.config['embedding_dim'])
            diagram_features = np.zeros(self.config['embedding_dim'])
            interaction_features = np.zeros(self.config['embedding_dim'])
            
            for modality_data in session.modality_data:
                if modality_data.modality_type == ModalityType.TEXT:
                    text_features = modality_data.embeddings
                elif modality_data.modality_type == ModalityType.DIAGRAM:
                    diagram_features = modality_data.embeddings
                elif modality_data.modality_type == ModalityType.INTERACTION:
                    interaction_features = modality_data.embeddings
            
            # Convert to tensors
            text_tensor = torch.FloatTensor(text_features).unsqueeze(0)
            diagram_tensor = torch.FloatTensor(diagram_features).unsqueeze(0)
            interaction_tensor = torch.FloatTensor(interaction_features).unsqueeze(0)
            
            # Forward pass through fusion network
            with torch.no_grad():
                predictions = self.fusion_network(text_tensor, diagram_tensor, interaction_tensor)
            
            # Extract outcomes
            learning_outcome = torch.sigmoid(predictions['learning_outcome']).item()
            engagement_level = torch.softmax(predictions['engagement'], dim=-1).argmax().item()
            concept_difficulty = torch.softmax(predictions['concept_difficulty'], dim=-1).argmax().item()
            
            outcomes = {
                'learning_success': learning_outcome,
                'engagement_level': engagement_level / 3.0,  # Normalize to 0-1
                'perceived_difficulty': concept_difficulty / 4.0  # Normalize to 0-1
            }
            
            return outcomes
            
        except Exception as e:
            logger.error(f"❌ Learning outcome calculation failed: {e}")
            return {}
    
    def _calculate_engagement_metrics(self, session: LearningSession) -> Dict[str, float]:
        """Calculate engagement metrics from multi-modal data"""
        try:
            metrics = {}
            
            # Text engagement
            text_data = [md for md in session.modality_data if md.modality_type == ModalityType.TEXT]
            if text_data:
                text_engagement = np.mean([td.confidence for td in text_data])
                metrics['text_engagement'] = text_engagement
            
            # Diagram engagement
            diagram_data = [md for md in session.modality_data if md.modality_type == ModalityType.DIAGRAM]
            if diagram_data:
                diagram_engagement = np.mean([dd.confidence for dd in diagram_data])
                metrics['diagram_engagement'] = diagram_engagement
            
            # Interaction engagement
            interaction_data = [md for md in session.modality_data if md.modality_type == ModalityType.INTERACTION]
            if interaction_data:
                interaction_scores = []
                for id_data in interaction_data:
                    if isinstance(id_data.content, dict):
                        # Calculate engagement from interaction patterns
                        click_rate = id_data.content.get('total_clicks', 0) / max(id_data.content.get('session_duration', 1), 1)
                        navigation_diversity = len(set(nav.get('page', '') for nav in id_data.content.get('navigation_events', [])))
                        engagement_score = min(1.0, (click_rate + navigation_diversity / 10) / 2)
                        interaction_scores.append(engagement_score)
                
                if interaction_scores:
                    metrics['interaction_engagement'] = np.mean(interaction_scores)
            
            # Overall engagement
            all_engagements = [v for v in metrics.values()]
            if all_engagements:
                metrics['overall_engagement'] = np.mean(all_engagements)
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Engagement metrics calculation failed: {e}")
            return {}
    
    def _calculate_difficulty_progression(self, session: LearningSession) -> List[float]:
        """Calculate difficulty progression throughout session"""
        try:
            progression = []
            
            # Use timestamps to order events
            timestamped_data = []
            for modality_data in session.modality_data:
                timestamped_data.append((modality_data.timestamp, modality_data))
            
            timestamped_data.sort(key=lambda x: x[0])
            
            # Calculate difficulty indicators over time
            window_size = max(1, len(timestamped_data) // 5)  # 5 time windows
            
            for i in range(0, len(timestamped_data), window_size):
                window_data = timestamped_data[i:i + window_size]
                
                # Calculate difficulty metrics for this window
                help_requests = 0
                total_features = []
                
                for _, modality_data in window_data:
                    if modality_data.modality_type == ModalityType.INTERACTION:
                        if isinstance(modality_data.content, dict):
                            help_requests += modality_data.content.get('help_requests', 0)
                    
                    total_features.extend(modality_data.features)
                
                # Difficulty score based on help requests and feature complexity
                if total_features:
                    feature_complexity = np.std(total_features)
                    difficulty_score = min(1.0, (help_requests + feature_complexity) / 2)
                else:
                    difficulty_score = 0.5
                
                progression.append(difficulty_score)
            
            return progression
            
        except Exception as e:
            logger.error(f"❌ Difficulty progression calculation failed: {e}")
            return []
    
    async def analyze_learning_patterns(self, sessions: List[LearningSession] = None) -> List[LearningPattern]:
        """Analyze learning patterns across sessions"""
        try:
            if sessions is None:
                sessions = self.processed_sessions
            
            if not sessions:
                logger.warning("⚠️ No sessions available for pattern analysis")
                return []
            
            patterns = self.pattern_detector.detect_patterns(sessions)
            self.detected_patterns.extend(patterns)
            
            logger.info(f"🔍 Analyzed learning patterns: {len(patterns)} patterns detected")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Learning pattern analysis failed: {e}")
            return []
    
    async def generate_insights(self, sessions: List[LearningSession] = None, 
                              patterns: List[LearningPattern] = None) -> List[MultiModalInsight]:
        """Generate actionable insights from analysis"""
        try:
            if sessions is None:
                sessions = self.processed_sessions
            if patterns is None:
                patterns = self.detected_patterns
            
            insights = self.insight_generator.generate_insights(sessions, patterns)
            
            # Filter by confidence threshold
            high_confidence_insights = [
                insight for insight in insights 
                if insight.confidence >= self.config['insight_confidence_threshold']
            ]
            
            self.generated_insights.extend(high_confidence_insights)
            
            logger.info(f"💡 Generated {len(high_confidence_insights)} high-confidence insights")
            return high_confidence_insights
            
        except Exception as e:
            logger.error(f"❌ Insight generation failed: {e}")
            return []
    
    async def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        try:
            # Calculate summary statistics
            total_sessions = len(self.processed_sessions)
            unique_students = len(set(s.student_id for s in self.processed_sessions))
            unique_topics = len(set(s.topic for s in self.processed_sessions))
            
            # Calculate average metrics
            all_learning_outcomes = []
            all_engagement_scores = []
            
            for session in self.processed_sessions:
                if session.learning_outcomes:
                    all_learning_outcomes.extend(session.learning_outcomes.values())
                if session.engagement_metrics:
                    all_engagement_scores.extend(session.engagement_metrics.values())
            
            avg_learning_success = np.mean(all_learning_outcomes) if all_learning_outcomes else 0
            avg_engagement = np.mean(all_engagement_scores) if all_engagement_scores else 0
            
            # Modality usage statistics
            modality_usage = Counter()
            for session in self.processed_sessions:
                for modality_data in session.modality_data:
                    modality_usage[modality_data.modality_type.value] += 1
            
            # Pattern frequency
            pattern_frequency = Counter(p.pattern_type for p in self.detected_patterns)
            
            # Recent insights
            recent_insights = sorted(
                self.generated_insights, 
                key=lambda x: len(x.affected_students), 
                reverse=True
            )[:10]
            
            dashboard_data = {
                'summary_statistics': {
                    'total_sessions': total_sessions,
                    'unique_students': unique_students,
                    'unique_topics': unique_topics,
                    'average_learning_success': avg_learning_success,
                    'average_engagement': avg_engagement
                },
                'modality_usage': dict(modality_usage),
                'learning_patterns': dict(pattern_frequency),
                'recent_insights': [
                    {
                        'type': insight.insight_type,
                        'description': insight.description,
                        'confidence': insight.confidence,
                        'affected_students_count': len(insight.affected_students),
                        'recommendations': insight.actionable_recommendations[:3]
                    }
                    for insight in recent_insights
                ],
                'temporal_trends': self._calculate_temporal_trends(),
                'concept_difficulty_map': self._generate_concept_difficulty_map(),
                'student_performance_distribution': self._calculate_performance_distribution()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Dashboard data generation failed: {e}")
            return {}
    
    def _calculate_temporal_trends(self) -> Dict[str, Any]:
        """Calculate temporal trends in learning data"""
        try:
            # Group sessions by week
            weekly_data = defaultdict(lambda: {'sessions': 0, 'success': [], 'engagement': []})
            
            for session in self.processed_sessions:
                week_key = session.start_time.strftime('%Y-W%U')
                weekly_data[week_key]['sessions'] += 1
                
                if session.learning_outcomes:
                    weekly_data[week_key]['success'].extend(session.learning_outcomes.values())
                if session.engagement_metrics:
                    weekly_data[week_key]['engagement'].extend(session.engagement_metrics.values())
            
            # Calculate weekly averages
            trends = {}
            for week, data in weekly_data.items():
                trends[week] = {
                    'session_count': data['sessions'],
                    'avg_success': np.mean(data['success']) if data['success'] else 0,
                    'avg_engagement': np.mean(data['engagement']) if data['engagement'] else 0
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Temporal trends calculation failed: {e}")
            return {}
    
    def _generate_concept_difficulty_map(self) -> Dict[str, float]:
        """Generate concept difficulty mapping"""
        try:
            concept_difficulties = {}
            concept_sessions = defaultdict(list)
            
            for session in self.processed_sessions:
                concept_sessions[session.topic].append(session)
            
            for concept, sessions in concept_sessions.items():
                success_rates = []
                help_rates = []
                
                for session in sessions:
                    if session.learning_outcomes:
                        success_rates.extend(session.learning_outcomes.values())
                    
                    # Extract help request rates
                    for modality_data in session.modality_data:
                        if (modality_data.modality_type == ModalityType.INTERACTION and 
                            isinstance(modality_data.content, dict)):
                            help_requests = modality_data.content.get('help_requests', 0)
                            session_duration = modality_data.content.get('session_duration', 1)
                            help_rates.append(help_requests / session_duration)
                
                if success_rates:
                    avg_success = np.mean(success_rates)
                    avg_help_rate = np.mean(help_rates) if help_rates else 0
                    
                    # Calculate difficulty as inverse of success plus help seeking
                    difficulty = (1.0 - avg_success + avg_help_rate) / 2
                    concept_difficulties[concept] = min(1.0, difficulty)
            
            return concept_difficulties
            
        except Exception as e:
            logger.error(f"❌ Concept difficulty mapping failed: {e}")
            return {}
    
    def _calculate_performance_distribution(self) -> Dict[str, int]:
        """Calculate student performance distribution"""
        try:
            student_performances = defaultdict(list)
            
            for session in self.processed_sessions:
                if session.learning_outcomes:
                    avg_performance = np.mean(list(session.learning_outcomes.values()))
                    student_performances[session.student_id].append(avg_performance)
            
            # Calculate overall performance per student
            student_overall_performance = {}
            for student_id, performances in student_performances.items():
                student_overall_performance[student_id] = np.mean(performances)
            
            # Create distribution
            distribution = {
                'excellent': 0,  # > 0.9
                'good': 0,       # 0.7 - 0.9
                'satisfactory': 0, # 0.5 - 0.7
                'needs_improvement': 0  # < 0.5
            }
            
            for performance in student_overall_performance.values():
                if performance > 0.9:
                    distribution['excellent'] += 1
                elif performance > 0.7:
                    distribution['good'] += 1
                elif performance > 0.5:
                    distribution['satisfactory'] += 1
                else:
                    distribution['needs_improvement'] += 1
            
            return distribution
            
        except Exception as e:
            logger.error(f"❌ Performance distribution calculation failed: {e}")
            return {}

# Testing function
async def test_multimodal_learning_analytics():
    """Test the multi-modal learning analytics system"""
    try:
        logger.info("🧪 Testing Multi-Modal Learning Analytics")
        
        # Initialize system
        analytics = MultiModalLearningAnalytics()
        await analytics.initialize()
        
        # Create sample session data
        sample_sessions = []
        
        for i in range(5):
            session_data = {
                'session_id': f'session_{i}',
                'student_id': f'student_{i % 3}',  # 3 students
                'topic': np.random.choice(['mechanics', 'energy', 'waves']),
                'start_time': datetime.now() - timedelta(days=i),
                'end_time': datetime.now() - timedelta(days=i) + timedelta(hours=1),
                'modalities': [
                    {
                        'type': 'text',
                        'content': f'Physics problem solving session {i}. Understanding concepts.',
                        'confidence': np.random.uniform(0.5, 1.0)
                    },
                    {
                        'type': 'interaction',
                        'content': {
                            'session_duration': 3600,
                            'total_clicks': np.random.randint(50, 200),
                            'help_requests': np.random.randint(0, 5),
                            'navigation_events': [
                                {'page': 'concept', 'timestamp': 100},
                                {'page': 'practice', 'timestamp': 200},
                                {'page': 'help', 'timestamp': 300}
                            ]
                        },
                        'confidence': np.random.uniform(0.6, 1.0)
                    }
                ]
            }
            
            # Add diagram modality for some sessions
            if i % 2 == 0:
                # Create a simple diagram (mock image data)
                diagram_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                session_data['modalities'].append({
                    'type': 'diagram',
                    'content': diagram_image,
                    'confidence': np.random.uniform(0.4, 0.9)
                })
            
            sample_sessions.append(session_data)
        
        # Process sessions
        processed_sessions = []
        for session_data in sample_sessions:
            session = await analytics.process_learning_session(session_data)
            processed_sessions.append(session)
            logger.info(f"✅ Processed session {session.session_id}")
        
        # Analyze patterns
        patterns = await analytics.analyze_learning_patterns(processed_sessions)
        logger.info(f"🔍 Detected {len(patterns)} learning patterns")
        
        # Generate insights
        insights = await analytics.generate_insights(processed_sessions, patterns)
        logger.info(f"💡 Generated {len(insights)} insights")
        
        # Generate dashboard data
        dashboard_data = await analytics.get_analytics_dashboard_data()
        logger.info(f"📊 Dashboard data: {len(dashboard_data)} sections")
        
        # Print summary
        logger.info("📈 Analytics Summary:")
        logger.info(f"   - Sessions processed: {len(processed_sessions)}")
        logger.info(f"   - Patterns detected: {len(patterns)}")
        logger.info(f"   - Insights generated: {len(insights)}")
        logger.info(f"   - Students analyzed: {dashboard_data['summary_statistics']['unique_students']}")
        logger.info(f"   - Topics covered: {dashboard_data['summary_statistics']['unique_topics']}")
        
        logger.info("✅ Multi-Modal Learning Analytics test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Multi-Modal Learning Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_multimodal_learning_analytics())