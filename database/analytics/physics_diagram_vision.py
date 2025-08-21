#!/usr/bin/env python3
"""
Computer Vision System for Physics Diagrams - Phase 6
Analyzes hand-drawn physics diagrams to extract concepts, detect errors,
and provide feedback on student diagram-based problem solving.
"""

import asyncio
import json
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import scipy.ndimage as ndimage
from scipy.spatial.distance import euclidean
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import base64
from io import BytesIO
import warnings

# Computer vision libraries
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("⚠️ MediaPipe not available - some features will be limited")

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagramType(Enum):
    FREE_BODY_DIAGRAM = "free_body_diagram"
    FORCE_DIAGRAM = "force_diagram"
    MOTION_DIAGRAM = "motion_diagram"
    ENERGY_DIAGRAM = "energy_diagram"
    CIRCUIT_DIAGRAM = "circuit_diagram"
    RAY_DIAGRAM = "ray_diagram"
    WAVE_DIAGRAM = "wave_diagram"
    FIELD_DIAGRAM = "field_diagram"

class DiagramElement(Enum):
    OBJECT = "object"
    FORCE_VECTOR = "force_vector"
    VELOCITY_VECTOR = "velocity_vector"
    ACCELERATION_VECTOR = "acceleration_vector"
    ARROW = "arrow"
    LINE = "line"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    TEXT_LABEL = "text_label"
    COORDINATE_AXIS = "coordinate_axis"

class AnalysisResult(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    UNCLEAR = "unclear"

@dataclass
class DetectedElement:
    """Detected element in physics diagram"""
    element_id: str
    element_type: DiagramElement
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    center_point: Tuple[float, float]
    properties: Dict[str, Any]
    
    # Physics-specific properties
    magnitude: Optional[float] = None
    direction: Optional[float] = None  # angle in degrees
    label: Optional[str] = None
    color: Optional[str] = None

@dataclass
class DiagramAnalysis:
    """Complete analysis of a physics diagram"""
    analysis_id: str
    student_id: str
    diagram_type: DiagramType
    detected_elements: List[DetectedElement]
    
    # Analysis results
    correctness_score: float
    completeness_score: float
    clarity_score: float
    overall_result: AnalysisResult
    
    # Physics-specific analysis
    force_balance: Dict[str, Any]
    vector_analysis: Dict[str, Any]
    conceptual_errors: List[str]
    missing_elements: List[str]
    
    # Feedback
    feedback_summary: str
    detailed_feedback: List[str]
    improvement_suggestions: List[str]
    
    # Metadata
    original_image_size: Tuple[int, int]
    processing_time_ms: float
    analyzed_at: datetime = field(default_factory=datetime.now)

class ImagePreprocessor:
    """Preprocess physics diagrams for analysis"""
    
    def __init__(self):
        self.target_size = (512, 512)
        self.noise_reduction_kernel = np.ones((3, 3), np.uint8)
    
    async def preprocess_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess image for various analysis tasks"""
        try:
            processed_images = {}
            
            # Original image info
            original_shape = image.shape
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Resize while maintaining aspect ratio
            resized = await self._resize_with_aspect_ratio(image, self.target_size)
            processed_images['resized'] = resized
            
            # Enhance contrast
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            processed_images['enhanced'] = enhanced
            
            # Edge detection
            edges = cv2.Canny(enhanced, 50, 150)
            processed_images['edges'] = edges
            
            # Binary thresholding
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            processed_images['binary'] = binary
            
            # Morphological operations for noise reduction
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.noise_reduction_kernel)
            processed_images['cleaned'] = cleaned
            
            # Skeletonization for line analysis
            skeleton = self._skeletonize(cleaned)
            processed_images['skeleton'] = skeleton
            
            return processed_images
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            return {'original': image}
    
    async def _resize_with_aspect_ratio(self, image: np.ndarray, 
                                      target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            
            # Calculate new dimensions
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas and center image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            if len(resized.shape) == 2:
                canvas = np.zeros((target_h, target_w), dtype=np.uint8)
            
            # Calculate padding
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            
            # Place resized image on canvas
            if len(resized.shape) == 3:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
            
        except Exception as e:
            logger.error(f"❌ Image resizing failed: {e}")
            return image
    
    def _skeletonize(self, binary_image: np.ndarray) -> np.ndarray:
        """Create skeleton of binary image for line analysis"""
        try:
            # Simple skeletonization using morphological operations
            skeleton = np.zeros_like(binary_image)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            
            while True:
                eroded = cv2.erode(binary_image, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(binary_image, temp)
                skeleton = cv2.bitwise_or(skeleton, temp)
                binary_image = eroded.copy()
                
                if cv2.countNonZero(binary_image) == 0:
                    break
            
            return skeleton
            
        except Exception as e:
            logger.error(f"❌ Skeletonization failed: {e}")
            return binary_image

class ShapeDetector:
    """Detect basic shapes in physics diagrams"""
    
    def __init__(self):
        self.min_contour_area = 100
        self.approx_epsilon_factor = 0.02
    
    async def detect_shapes(self, binary_image: np.ndarray) -> List[DetectedElement]:
        """Detect basic shapes like circles, rectangles, lines"""
        try:
            detected_elements = []
            
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Approximate contour shape
                epsilon = self.approx_epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Classify shape
                shape_type, confidence = await self._classify_shape(contour, approx, area)
                
                # Calculate additional properties
                properties = await self._calculate_shape_properties(contour, approx)
                
                element = DetectedElement(
                    element_id=f"shape_{i}",
                    element_type=shape_type,
                    confidence=confidence,
                    bounding_box=(x, y, w, h),
                    center_point=(center_x, center_y),
                    properties=properties
                )
                
                detected_elements.append(element)
            
            return detected_elements
            
        except Exception as e:
            logger.error(f"❌ Shape detection failed: {e}")
            return []
    
    async def _classify_shape(self, contour: np.ndarray, approx: np.ndarray, 
                            area: float) -> Tuple[DiagramElement, float]:
        """Classify detected shape"""
        try:
            num_vertices = len(approx)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Classify based on vertices and circularity
            if circularity > 0.7:
                return DiagramElement.CIRCLE, 0.9
            elif num_vertices == 3:
                return DiagramElement.OBJECT, 0.7  # Triangle (often used for objects)
            elif num_vertices == 4:
                # Check if rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 1
                if 0.8 <= aspect_ratio <= 1.2:
                    return DiagramElement.OBJECT, 0.8  # Square object
                else:
                    return DiagramElement.RECTANGLE, 0.8
            elif num_vertices > 4:
                if circularity > 0.5:
                    return DiagramElement.CIRCLE, 0.6
                else:
                    return DiagramElement.OBJECT, 0.5
            else:
                return DiagramElement.OBJECT, 0.4
            
        except Exception as e:
            logger.error(f"❌ Shape classification failed: {e}")
            return DiagramElement.OBJECT, 0.1
    
    async def _calculate_shape_properties(self, contour: np.ndarray, 
                                        approx: np.ndarray) -> Dict[str, Any]:
        """Calculate additional properties of detected shape"""
        try:
            properties = {}
            
            # Area and perimeter
            properties['area'] = cv2.contourArea(contour)
            properties['perimeter'] = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            properties['aspect_ratio'] = w / h if h > 0 else 1
            
            # Minimum enclosing circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            properties['enclosing_circle_radius'] = radius
            properties['enclosing_circle_center'] = (center_x, center_y)
            
            # Convex hull
            hull = cv2.convexHull(contour)
            properties['convexity'] = cv2.contourArea(contour) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            
            # Orientation
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                properties['orientation'] = ellipse[2]  # Angle
            else:
                properties['orientation'] = 0
            
            return properties
            
        except Exception as e:
            logger.error(f"❌ Shape property calculation failed: {e}")
            return {}

class VectorDetector:
    """Detect force vectors and arrows in physics diagrams"""
    
    def __init__(self):
        self.min_line_length = 20
        self.max_line_gap = 10
        self.angle_threshold = 30  # degrees
    
    async def detect_vectors(self, edge_image: np.ndarray, 
                           skeleton_image: np.ndarray) -> List[DetectedElement]:
        """Detect arrows and force vectors"""
        try:
            detected_vectors = []
            
            # Detect lines using Hough Line Transform
            lines = cv2.HoughLinesP(edge_image, 1, np.pi/180, 
                                  threshold=50, minLineLength=self.min_line_length, 
                                  maxLineGap=self.max_line_gap)
            
            if lines is None:
                return detected_vectors
            
            # Group lines into potential vectors
            vector_groups = await self._group_lines_into_vectors(lines)
            
            # Analyze each vector group
            for i, line_group in enumerate(vector_groups):
                vector_properties = await self._analyze_vector_group(line_group, skeleton_image)
                
                if vector_properties['confidence'] > 0.3:
                    # Calculate bounding box
                    all_points = np.concatenate([line[:2] for line in line_group] + 
                                               [line[2:] for line in line_group])
                    x_coords = all_points[::2]
                    y_coords = all_points[1::2]
                    
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                    
                    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                    
                    # Determine vector type based on properties
                    vector_type = await self._classify_vector_type(vector_properties)
                    
                    element = DetectedElement(
                        element_id=f"vector_{i}",
                        element_type=vector_type,
                        confidence=vector_properties['confidence'],
                        bounding_box=bbox,
                        center_point=center,
                        properties=vector_properties,
                        magnitude=vector_properties.get('magnitude'),
                        direction=vector_properties.get('direction')
                    )
                    
                    detected_vectors.append(element)
            
            return detected_vectors
            
        except Exception as e:
            logger.error(f"❌ Vector detection failed: {e}")
            return []
    
    async def _group_lines_into_vectors(self, lines: np.ndarray) -> List[List[np.ndarray]]:
        """Group detected lines into potential vectors"""
        try:
            if lines is None or len(lines) == 0:
                return []
            
            vector_groups = []
            used_lines = set()
            
            for i, line1 in enumerate(lines):
                if i in used_lines:
                    continue
                
                x1, y1, x2, y2 = line1[0]
                line1_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                line1_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Start a new vector group
                current_group = [line1[0]]
                used_lines.add(i)
                
                # Look for nearby parallel lines (potential arrow components)
                for j, line2 in enumerate(lines):
                    if j in used_lines or j == i:
                        continue
                    
                    x3, y3, x4, y4 = line2[0]
                    line2_angle = np.arctan2(y4 - y3, x4 - x3) * 180 / np.pi
                    
                    # Check if lines are roughly parallel or form arrow shape
                    angle_diff = abs(line1_angle - line2_angle)
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    
                    # Check distance between lines
                    dist = self._line_to_line_distance(line1[0], line2[0])
                    
                    if (angle_diff < self.angle_threshold and dist < 50) or \
                       (45 < angle_diff < 135 and dist < 30):  # Potential arrowhead
                        current_group.append(line2[0])
                        used_lines.add(j)
                
                if len(current_group) >= 1:  # At least one line
                    vector_groups.append(current_group)
            
            return vector_groups
            
        except Exception as e:
            logger.error(f"❌ Line grouping failed: {e}")
            return []
    
    def _line_to_line_distance(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """Calculate minimum distance between two lines"""
        try:
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            
            # Calculate distances between all endpoint combinations
            distances = [
                np.sqrt((x1 - x3)**2 + (y1 - y3)**2),
                np.sqrt((x1 - x4)**2 + (y1 - y4)**2),
                np.sqrt((x2 - x3)**2 + (y2 - y3)**2),
                np.sqrt((x2 - x4)**2 + (y2 - y4)**2)
            ]
            
            return min(distances)
            
        except Exception as e:
            logger.error(f"❌ Line distance calculation failed: {e}")
            return float('inf')
    
    async def _analyze_vector_group(self, line_group: List[np.ndarray], 
                                  skeleton_image: np.ndarray) -> Dict[str, Any]:
        """Analyze a group of lines to determine vector properties"""
        try:
            properties = {}
            
            if not line_group:
                return {'confidence': 0.0}
            
            # Find the main line (longest line)
            main_line = max(line_group, key=lambda line: 
                          np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2))
            
            x1, y1, x2, y2 = main_line
            
            # Calculate vector properties
            properties['length'] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            properties['angle'] = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            properties['direction'] = properties['angle']
            properties['magnitude'] = properties['length'] / 50.0  # Normalize by expected scale
            
            # Check for arrowhead
            has_arrowhead = len(line_group) > 1
            properties['has_arrowhead'] = has_arrowhead
            
            # Calculate confidence based on properties
            confidence = 0.5  # Base confidence
            
            if properties['length'] > 30:
                confidence += 0.2
            if has_arrowhead:
                confidence += 0.3
            if properties['length'] > 50:
                confidence += 0.1
            
            properties['confidence'] = min(0.95, confidence)
            
            return properties
            
        except Exception as e:
            logger.error(f"❌ Vector analysis failed: {e}")
            return {'confidence': 0.0}
    
    async def _classify_vector_type(self, properties: Dict[str, Any]) -> DiagramElement:
        """Classify vector type based on properties"""
        try:
            # Simple classification based on length and context
            length = properties.get('length', 0)
            has_arrowhead = properties.get('has_arrowhead', False)
            
            if has_arrowhead:
                if length > 60:
                    return DiagramElement.FORCE_VECTOR
                elif length > 30:
                    return DiagramElement.VELOCITY_VECTOR
                else:
                    return DiagramElement.ACCELERATION_VECTOR
            else:
                return DiagramElement.ARROW
            
        except Exception as e:
            logger.error(f"❌ Vector classification failed: {e}")
            return DiagramElement.ARROW

class PhysicsAnalyzer:
    """Analyze physics concepts in diagrams"""
    
    def __init__(self):
        self.force_balance_threshold = 0.1
        self.vector_analysis_threshold = 10  # degrees
    
    async def analyze_physics_concepts(self, detected_elements: List[DetectedElement],
                                     diagram_type: DiagramType) -> Dict[str, Any]:
        """Analyze physics concepts based on detected elements"""
        try:
            analysis = {
                'force_balance': {},
                'vector_analysis': {},
                'conceptual_errors': [],
                'missing_elements': [],
                'physics_correctness': 0.0
            }
            
            # Separate elements by type
            objects = [e for e in detected_elements if e.element_type == DiagramElement.OBJECT]
            force_vectors = [e for e in detected_elements if e.element_type == DiagramElement.FORCE_VECTOR]
            velocity_vectors = [e for e in detected_elements if e.element_type == DiagramElement.VELOCITY_VECTOR]
            
            # Analyze based on diagram type
            if diagram_type == DiagramType.FREE_BODY_DIAGRAM:
                analysis = await self._analyze_free_body_diagram(objects, force_vectors, analysis)
            elif diagram_type == DiagramType.FORCE_DIAGRAM:
                analysis = await self._analyze_force_diagram(objects, force_vectors, analysis)
            elif diagram_type == DiagramType.MOTION_DIAGRAM:
                analysis = await self._analyze_motion_diagram(objects, velocity_vectors, analysis)
            
            # General vector analysis
            if force_vectors or velocity_vectors:
                all_vectors = force_vectors + velocity_vectors
                analysis['vector_analysis'] = await self._analyze_vectors(all_vectors)
            
            # Calculate overall physics correctness
            analysis['physics_correctness'] = await self._calculate_physics_correctness(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Physics analysis failed: {e}")
            return {'force_balance': {}, 'vector_analysis': {}, 'conceptual_errors': [], 
                   'missing_elements': [], 'physics_correctness': 0.0}
    
    async def _analyze_free_body_diagram(self, objects: List[DetectedElement],
                                       force_vectors: List[DetectedElement],
                                       analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze free body diagram for physics correctness"""
        try:
            if not objects:
                analysis['conceptual_errors'].append("No object identified in free body diagram")
                return analysis
            
            if len(objects) > 1:
                analysis['conceptual_errors'].append("Free body diagram should show only one object")
            
            main_object = objects[0]
            object_center = main_object.center_point
            
            # Check if forces originate from object
            forces_from_object = []
            for force in force_vectors:
                distance_to_object = np.sqrt(
                    (force.center_point[0] - object_center[0])**2 + 
                    (force.center_point[1] - object_center[1])**2
                )
                
                if distance_to_object < 100:  # Within reasonable distance
                    forces_from_object.append(force)
            
            analysis['force_balance']['forces_on_object'] = len(forces_from_object)
            analysis['force_balance']['total_forces'] = len(force_vectors)
            
            # Check force balance
            if forces_from_object:
                net_force = await self._calculate_net_force(forces_from_object)
                analysis['force_balance']['net_force_magnitude'] = net_force['magnitude']
                analysis['force_balance']['net_force_direction'] = net_force['direction']
                
                if net_force['magnitude'] < self.force_balance_threshold:
                    analysis['force_balance']['is_balanced'] = True
                else:
                    analysis['force_balance']['is_balanced'] = False
            
            # Check for missing common forces
            if len(forces_from_object) < 2:
                analysis['missing_elements'].append("Consider adding more forces (e.g., weight, normal force)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Free body diagram analysis failed: {e}")
            return analysis
    
    async def _analyze_force_diagram(self, objects: List[DetectedElement],
                                   force_vectors: List[DetectedElement],
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze force diagram"""
        try:
            # Similar to free body diagram but may have multiple objects
            analysis['force_balance']['total_objects'] = len(objects)
            analysis['force_balance']['total_forces'] = len(force_vectors)
            
            if not force_vectors:
                analysis['conceptual_errors'].append("No forces shown in force diagram")
                analysis['missing_elements'].append("Add force vectors to show forces acting")
            
            # Analyze force interactions between objects
            if len(objects) > 1:
                # Check for Newton's third law pairs
                analysis['force_balance']['interaction_pairs'] = await self._check_interaction_pairs(
                    objects, force_vectors
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Force diagram analysis failed: {e}")
            return analysis
    
    async def _analyze_motion_diagram(self, objects: List[DetectedElement],
                                    velocity_vectors: List[DetectedElement],
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze motion diagram"""
        try:
            analysis['motion_analysis'] = {}
            
            if not velocity_vectors:
                analysis['conceptual_errors'].append("No velocity vectors shown in motion diagram")
                analysis['missing_elements'].append("Add velocity vectors to show motion")
                return analysis
            
            # Analyze velocity vector progression
            if len(velocity_vectors) > 1:
                # Check for consistent acceleration patterns
                acceleration_consistency = await self._check_acceleration_consistency(velocity_vectors)
                analysis['motion_analysis']['acceleration_consistent'] = acceleration_consistency
                
                if not acceleration_consistency:
                    analysis['conceptual_errors'].append("Velocity vectors don't show consistent acceleration pattern")
            
            # Check vector spacing (should represent equal time intervals)
            spacing_consistency = await self._check_vector_spacing(velocity_vectors)
            analysis['motion_analysis']['spacing_consistent'] = spacing_consistency
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Motion diagram analysis failed: {e}")
            return analysis
    
    async def _calculate_net_force(self, force_vectors: List[DetectedElement]) -> Dict[str, float]:
        """Calculate net force from force vectors"""
        try:
            total_x = 0.0
            total_y = 0.0
            
            for force in force_vectors:
                magnitude = force.magnitude or 1.0
                direction_rad = np.radians(force.direction or 0.0)
                
                total_x += magnitude * np.cos(direction_rad)
                total_y += magnitude * np.sin(direction_rad)
            
            net_magnitude = np.sqrt(total_x**2 + total_y**2)
            net_direction = np.degrees(np.arctan2(total_y, total_x))
            
            return {
                'magnitude': net_magnitude,
                'direction': net_direction,
                'components': {'x': total_x, 'y': total_y}
            }
            
        except Exception as e:
            logger.error(f"❌ Net force calculation failed: {e}")
            return {'magnitude': 0.0, 'direction': 0.0, 'components': {'x': 0.0, 'y': 0.0}}
    
    async def _analyze_vectors(self, vectors: List[DetectedElement]) -> Dict[str, Any]:
        """Analyze vector properties and relationships"""
        try:
            analysis = {
                'total_vectors': len(vectors),
                'average_magnitude': 0.0,
                'direction_spread': 0.0,
                'parallel_vectors': 0,
                'perpendicular_vectors': 0
            }
            
            if not vectors:
                return analysis
            
            # Calculate average magnitude
            magnitudes = [v.magnitude or 1.0 for v in vectors]
            analysis['average_magnitude'] = np.mean(magnitudes)
            
            # Calculate direction spread
            directions = [v.direction or 0.0 for v in vectors]
            analysis['direction_spread'] = np.std(directions)
            
            # Count parallel and perpendicular vectors
            for i, vector1 in enumerate(vectors):
                for j, vector2 in enumerate(vectors[i+1:], i+1):
                    angle_diff = abs((vector1.direction or 0) - (vector2.direction or 0))
                    angle_diff = min(angle_diff, 180 - angle_diff)
                    
                    if angle_diff < self.vector_analysis_threshold:
                        analysis['parallel_vectors'] += 1
                    elif abs(angle_diff - 90) < self.vector_analysis_threshold:
                        analysis['perpendicular_vectors'] += 1
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Vector analysis failed: {e}")
            return analysis
    
    async def _check_interaction_pairs(self, objects: List[DetectedElement],
                                     force_vectors: List[DetectedElement]) -> int:
        """Check for Newton's third law force pairs"""
        try:
            # Simplified check - count forces between object pairs
            pairs_found = 0
            
            for i, obj1 in enumerate(objects):
                for obj2 in objects[i+1:]:
                    # Check if there are forces between these objects
                    forces_between = 0
                    
                    for force in force_vectors:
                        # Check if force is between the two objects
                        dist1 = np.sqrt((force.center_point[0] - obj1.center_point[0])**2 + 
                                       (force.center_point[1] - obj1.center_point[1])**2)
                        dist2 = np.sqrt((force.center_point[0] - obj2.center_point[0])**2 + 
                                       (force.center_point[1] - obj2.center_point[1])**2)
                        
                        if min(dist1, dist2) < 150:  # Force is near one of the objects
                            forces_between += 1
                    
                    if forces_between >= 2:  # Action-reaction pair
                        pairs_found += 1
            
            return pairs_found
            
        except Exception as e:
            logger.error(f"❌ Interaction pair check failed: {e}")
            return 0
    
    async def _check_acceleration_consistency(self, velocity_vectors: List[DetectedElement]) -> bool:
        """Check if velocity vectors show consistent acceleration"""
        try:
            if len(velocity_vectors) < 3:
                return True  # Can't determine with less than 3 vectors
            
            # Sort vectors by position (assuming left to right motion)
            sorted_vectors = sorted(velocity_vectors, key=lambda v: v.center_point[0])
            
            # Check if magnitude changes are consistent
            magnitude_changes = []
            for i in range(len(sorted_vectors) - 1):
                mag1 = sorted_vectors[i].magnitude or 1.0
                mag2 = sorted_vectors[i + 1].magnitude or 1.0
                magnitude_changes.append(mag2 - mag1)
            
            # Check if changes are roughly constant (constant acceleration)
            if len(magnitude_changes) > 1:
                change_consistency = np.std(magnitude_changes) / np.mean(np.abs(magnitude_changes))
                return change_consistency < 0.5  # Relatively consistent
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Acceleration consistency check failed: {e}")
            return False
    
    async def _check_vector_spacing(self, vectors: List[DetectedElement]) -> bool:
        """Check if vectors are spaced consistently (equal time intervals)"""
        try:
            if len(vectors) < 3:
                return True
            
            # Sort vectors by position
            sorted_vectors = sorted(vectors, key=lambda v: v.center_point[0])
            
            # Calculate spacings
            spacings = []
            for i in range(len(sorted_vectors) - 1):
                dist = np.sqrt(
                    (sorted_vectors[i+1].center_point[0] - sorted_vectors[i].center_point[0])**2 +
                    (sorted_vectors[i+1].center_point[1] - sorted_vectors[i].center_point[1])**2
                )
                spacings.append(dist)
            
            # Check consistency
            if spacings:
                spacing_consistency = np.std(spacings) / np.mean(spacings)
                return spacing_consistency < 0.3  # Relatively consistent spacing
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector spacing check failed: {e}")
            return False
    
    async def _calculate_physics_correctness(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall physics correctness score"""
        try:
            score = 1.0
            
            # Penalize for conceptual errors
            error_count = len(analysis.get('conceptual_errors', []))
            score -= error_count * 0.2
            
            # Penalize for missing elements
            missing_count = len(analysis.get('missing_elements', []))
            score -= missing_count * 0.15
            
            # Bonus for balanced forces
            force_balance = analysis.get('force_balance', {})
            if force_balance.get('is_balanced', False):
                score += 0.1
            
            # Ensure score is in valid range
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Physics correctness calculation failed: {e}")
            return 0.5

class PhysicsDiagramVision:
    """Main computer vision system for physics diagrams"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Core components
        self.preprocessor = ImagePreprocessor()
        self.shape_detector = ShapeDetector()
        self.vector_detector = VectorDetector()
        self.physics_analyzer = PhysicsAnalyzer()
        
        # Analysis history
        self.analysis_history = []
        
        # Diagram type classification
        self.diagram_classifier = None
    
    async def initialize(self):
        """Initialize the physics diagram vision system"""
        try:
            logger.info("🚀 Initializing Physics Diagram Vision System")
            
            # Initialize components
            # (Components are already initialized in __init__)
            
            # Load any pre-trained models
            await self._load_models()
            
            logger.info("✅ Physics Diagram Vision System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Physics Diagram Vision System: {e}")
            return False
    
    async def _load_models(self):
        """Load pre-trained models"""
        try:
            # In a real implementation, load diagram classification models
            logger.info("📚 Model loading skipped (would load diagram classification models)")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
    
    async def analyze_diagram(self, image: np.ndarray, 
                            student_id: str,
                            expected_diagram_type: DiagramType = None) -> DiagramAnalysis:
        """Analyze a physics diagram"""
        try:
            start_time = datetime.now()
            logger.info(f"🔍 Analyzing physics diagram for student {student_id}")
            
            # Preprocess image
            processed_images = await self.preprocessor.preprocess_image(image)
            
            # Classify diagram type if not provided
            if expected_diagram_type is None:
                diagram_type = await self._classify_diagram_type(processed_images)
            else:
                diagram_type = expected_diagram_type
            
            # Detect elements
            detected_elements = []
            
            # Detect shapes
            if 'binary' in processed_images:
                shapes = await self.shape_detector.detect_shapes(processed_images['binary'])
                detected_elements.extend(shapes)
            
            # Detect vectors
            if 'edges' in processed_images and 'skeleton' in processed_images:
                vectors = await self.vector_detector.detect_vectors(
                    processed_images['edges'], processed_images['skeleton']
                )
                detected_elements.extend(vectors)
            
            # Analyze physics concepts
            physics_analysis = await self.physics_analyzer.analyze_physics_concepts(
                detected_elements, diagram_type
            )
            
            # Calculate scores
            correctness_score = physics_analysis.get('physics_correctness', 0.0)
            completeness_score = await self._calculate_completeness_score(
                detected_elements, diagram_type
            )
            clarity_score = await self._calculate_clarity_score(detected_elements, processed_images)
            
            # Determine overall result
            overall_result = await self._determine_overall_result(
                correctness_score, completeness_score, clarity_score
            )
            
            # Generate feedback
            feedback = await self._generate_feedback(
                physics_analysis, detected_elements, diagram_type
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create analysis result
            analysis = DiagramAnalysis(
                analysis_id=f"analysis_{student_id}_{datetime.now().timestamp()}",
                student_id=student_id,
                diagram_type=diagram_type,
                detected_elements=detected_elements,
                correctness_score=correctness_score,
                completeness_score=completeness_score,
                clarity_score=clarity_score,
                overall_result=overall_result,
                force_balance=physics_analysis.get('force_balance', {}),
                vector_analysis=physics_analysis.get('vector_analysis', {}),
                conceptual_errors=physics_analysis.get('conceptual_errors', []),
                missing_elements=physics_analysis.get('missing_elements', []),
                feedback_summary=feedback['summary'],
                detailed_feedback=feedback['detailed'],
                improvement_suggestions=feedback['suggestions'],
                original_image_size=image.shape[:2],
                processing_time_ms=processing_time
            )
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            logger.info(f"✅ Analysis completed in {processing_time:.1f}ms - Score: {correctness_score:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Diagram analysis failed: {e}")
            # Return minimal analysis
            return await self._create_fallback_analysis(student_id, image)
    
    async def _classify_diagram_type(self, processed_images: Dict[str, np.ndarray]) -> DiagramType:
        """Classify the type of physics diagram"""
        try:
            # Simple heuristic classification
            # In a real implementation, this would use a trained classifier
            
            # Count different types of elements to infer diagram type
            binary_image = processed_images.get('binary')
            if binary_image is None:
                return DiagramType.FREE_BODY_DIAGRAM  # Default
            
            # Detect basic shapes and lines
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count circular and rectangular shapes
            circles = 0
            rectangles = 0
            lines = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.7:
                    circles += 1
                else:
                    # Check if roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4:
                        rectangles += 1
                    elif len(approx) <= 2:
                        lines += 1
            
            # Simple classification logic
            if circles >= 2 and lines >= 3:
                return DiagramType.CIRCUIT_DIAGRAM
            elif rectangles >= 1 and lines >= 2:
                return DiagramType.FREE_BODY_DIAGRAM
            elif lines >= 3:
                return DiagramType.FORCE_DIAGRAM
            else:
                return DiagramType.FREE_BODY_DIAGRAM
            
        except Exception as e:
            logger.error(f"❌ Diagram classification failed: {e}")
            return DiagramType.FREE_BODY_DIAGRAM
    
    async def _calculate_completeness_score(self, detected_elements: List[DetectedElement],
                                          diagram_type: DiagramType) -> float:
        """Calculate how complete the diagram is"""
        try:
            expected_elements = {
                DiagramType.FREE_BODY_DIAGRAM: {
                    DiagramElement.OBJECT: 1,
                    DiagramElement.FORCE_VECTOR: 2
                },
                DiagramType.FORCE_DIAGRAM: {
                    DiagramElement.OBJECT: 1,
                    DiagramElement.FORCE_VECTOR: 2
                },
                DiagramType.MOTION_DIAGRAM: {
                    DiagramElement.OBJECT: 1,
                    DiagramElement.VELOCITY_VECTOR: 3
                }
            }
            
            expected = expected_elements.get(diagram_type, {})
            if not expected:
                return 0.8  # Default score if no expectations defined
            
            # Count detected elements by type
            detected_counts = defaultdict(int)
            for element in detected_elements:
                detected_counts[element.element_type] += 1
            
            # Calculate completeness for each expected element type
            completeness_scores = []
            for element_type, expected_count in expected.items():
                detected_count = detected_counts[element_type]
                score = min(1.0, detected_count / expected_count)
                completeness_scores.append(score)
            
            # Return average completeness
            return np.mean(completeness_scores) if completeness_scores else 0.0
            
        except Exception as e:
            logger.error(f"❌ Completeness calculation failed: {e}")
            return 0.5
    
    async def _calculate_clarity_score(self, detected_elements: List[DetectedElement],
                                     processed_images: Dict[str, np.ndarray]) -> float:
        """Calculate how clear and well-drawn the diagram is"""
        try:
            score = 1.0
            
            # Penalize for too many overlapping elements
            overlaps = await self._count_overlaps(detected_elements)
            score -= overlaps * 0.1
            
            # Reward for good contrast (if we have processed images)
            if 'enhanced' in processed_images:
                contrast_score = self._calculate_contrast_score(processed_images['enhanced'])
                score = (score + contrast_score) / 2
            
            # Penalize for too few or too many elements
            element_count = len(detected_elements)
            if element_count < 2:
                score -= 0.3
            elif element_count > 15:
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Clarity calculation failed: {e}")
            return 0.7
    
    async def _count_overlaps(self, detected_elements: List[DetectedElement]) -> int:
        """Count overlapping elements"""
        try:
            overlaps = 0
            
            for i, elem1 in enumerate(detected_elements):
                for elem2 in detected_elements[i+1:]:
                    # Check if bounding boxes overlap
                    x1, y1, w1, h1 = elem1.bounding_box
                    x2, y2, w2, h2 = elem2.bounding_box
                    
                    if (x1 < x2 + w2 and x1 + w1 > x2 and 
                        y1 < y2 + h2 and y1 + h1 > y2):
                        overlaps += 1
            
            return overlaps
            
        except Exception as e:
            logger.error(f"❌ Overlap counting failed: {e}")
            return 0
    
    def _calculate_contrast_score(self, gray_image: np.ndarray) -> float:
        """Calculate contrast score of image"""
        try:
            # Calculate standard deviation as a measure of contrast
            contrast = np.std(gray_image) / 255.0
            return min(1.0, contrast * 2)  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"❌ Contrast calculation failed: {e}")
            return 0.5
    
    async def _determine_overall_result(self, correctness: float, 
                                      completeness: float, clarity: float) -> AnalysisResult:
        """Determine overall analysis result"""
        try:
            average_score = (correctness + completeness + clarity) / 3
            
            if average_score >= 0.8:
                return AnalysisResult.CORRECT
            elif average_score >= 0.6:
                return AnalysisResult.INCOMPLETE
            elif average_score >= 0.3:
                return AnalysisResult.INCORRECT
            else:
                return AnalysisResult.UNCLEAR
            
        except Exception as e:
            logger.error(f"❌ Overall result determination failed: {e}")
            return AnalysisResult.UNCLEAR
    
    async def _generate_feedback(self, physics_analysis: Dict[str, Any],
                               detected_elements: List[DetectedElement],
                               diagram_type: DiagramType) -> Dict[str, Any]:
        """Generate feedback for the student"""
        try:
            feedback = {
                'summary': '',
                'detailed': [],
                'suggestions': []
            }
            
            # Generate summary
            correctness = physics_analysis.get('physics_correctness', 0.0)
            if correctness >= 0.8:
                feedback['summary'] = "Great work! Your diagram shows good understanding of physics concepts."
            elif correctness >= 0.6:
                feedback['summary'] = "Good start! There are a few areas that could be improved."
            elif correctness >= 0.3:
                feedback['summary'] = "Your diagram needs some corrections to better represent the physics."
            else:
                feedback['summary'] = "Let's work on improving your diagram to better show the physics concepts."
            
            # Add detailed feedback for errors
            conceptual_errors = physics_analysis.get('conceptual_errors', [])
            for error in conceptual_errors:
                feedback['detailed'].append(f"⚠️ {error}")
            
            # Add suggestions for missing elements
            missing_elements = physics_analysis.get('missing_elements', [])
            for missing in missing_elements:
                feedback['suggestions'].append(f"💡 {missing}")
            
            # Add physics-specific feedback
            force_balance = physics_analysis.get('force_balance', {})
            if 'is_balanced' in force_balance:
                if force_balance['is_balanced']:
                    feedback['detailed'].append("✅ Forces are properly balanced")
                else:
                    feedback['detailed'].append("⚠️ Forces don't appear to be balanced")
                    feedback['suggestions'].append("Check that all forces are included and properly represented")
            
            # Add general suggestions
            if len(detected_elements) < 3:
                feedback['suggestions'].append("Consider adding more detail to your diagram")
            
            # Diagram-specific suggestions
            if diagram_type == DiagramType.FREE_BODY_DIAGRAM:
                if not any(e.element_type == DiagramElement.FORCE_VECTOR for e in detected_elements):
                    feedback['suggestions'].append("Add force vectors showing all forces acting on the object")
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Feedback generation failed: {e}")
            return {
                'summary': 'Analysis completed. Continue working on your diagram.',
                'detailed': [],
                'suggestions': ['Keep practicing with physics diagrams']
            }
    
    async def _create_fallback_analysis(self, student_id: str, 
                                      image: np.ndarray) -> DiagramAnalysis:
        """Create fallback analysis when main process fails"""
        try:
            return DiagramAnalysis(
                analysis_id=f"fallback_{student_id}_{datetime.now().timestamp()}",
                student_id=student_id,
                diagram_type=DiagramType.FREE_BODY_DIAGRAM,
                detected_elements=[],
                correctness_score=0.3,
                completeness_score=0.3,
                clarity_score=0.5,
                overall_result=AnalysisResult.UNCLEAR,
                force_balance={},
                vector_analysis={},
                conceptual_errors=["Unable to analyze diagram automatically"],
                missing_elements=[],
                feedback_summary="Please try uploading a clearer image of your diagram.",
                detailed_feedback=["Image analysis encountered difficulties"],
                improvement_suggestions=["Ensure your diagram is clear and well-lit", "Use darker lines for better visibility"],
                original_image_size=image.shape[:2],
                processing_time_ms=100.0
            )
            
        except Exception as e:
            logger.error(f"❌ Fallback analysis creation failed: {e}")
            raise

# Testing function
async def test_physics_diagram_vision():
    """Test physics diagram vision system"""
    try:
        logger.info("🧪 Testing Physics Diagram Vision System")
        
        vision_system = PhysicsDiagramVision()
        await vision_system.initialize()
        
        # Create a simple test image (white background with black shapes)
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw a simple free body diagram
        # Object (rectangle)
        cv2.rectangle(test_image, (180, 180), (220, 220), (0, 0, 0), 2)
        
        # Force vectors (arrows)
        # Weight (downward)
        cv2.arrowedLine(test_image, (200, 220), (200, 280), (0, 0, 0), 2)
        # Normal force (upward)
        cv2.arrowedLine(test_image, (200, 180), (200, 120), (0, 0, 0), 2)
        # Applied force (right)
        cv2.arrowedLine(test_image, (220, 200), (280, 200), (0, 0, 0), 2)
        
        # Analyze the test diagram
        analysis = await vision_system.analyze_diagram(
            test_image, 
            "test_student",
            DiagramType.FREE_BODY_DIAGRAM
        )
        
        logger.info(f"✅ Analysis completed for test diagram")
        logger.info(f"📊 Detected {len(analysis.detected_elements)} elements")
        logger.info(f"🎯 Correctness score: {analysis.correctness_score:.2f}")
        logger.info(f"📝 Feedback: {analysis.feedback_summary}")
        
        # Display detected elements
        for element in analysis.detected_elements:
            logger.info(f"  - {element.element_type.value}: confidence {element.confidence:.2f}")
        
        logger.info("✅ Physics Diagram Vision System test completed")
        
    except Exception as e:
        logger.error(f"❌ Physics Diagram Vision test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_physics_diagram_vision())