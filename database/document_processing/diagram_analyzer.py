#!/usr/bin/env python3
"""
Physics Diagram Analyzer for Educational Content
Analyzes physics diagrams, free body diagrams, circuit diagrams, and other visual physics content.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiagramElement:
    """Represents an element detected in a physics diagram"""
    element_type: str  # vector, object, label, line, circle, etc.
    coordinates: Tuple[int, int, int, int]  # x, y, width, height
    properties: Dict[str, Any]
    confidence: float
    description: Optional[str] = None

@dataclass
class PhysicsDiagram:
    """Represents a complete analyzed physics diagram"""
    image_path: str
    diagram_type: str  # free_body, circuit, wave, field, etc.
    elements: List[DiagramElement]
    objects: List[str]  # identified physics objects
    vectors: List[Dict[str, Any]]  # force vectors, velocity vectors, etc.
    labels: List[str]  # text labels found
    physics_concepts: List[str]  # identified physics concepts
    complexity_score: int
    analysis_metadata: Dict[str, Any]

class PhysicsDiagramAnalyzer:
    """Advanced analyzer for physics diagrams and visual content"""
    
    def __init__(self):
        # Physics diagram type patterns
        self.diagram_patterns = {
            'free_body': ['force', 'weight', 'normal', 'friction', 'tension', 'mg', 'N', 'f'],
            'circuit': ['resistor', 'capacitor', 'battery', 'wire', 'current', 'voltage', 'V', 'I', 'R'],
            'wave': ['amplitude', 'wavelength', 'frequency', 'sine', 'cosine', 'crest', 'trough'],
            'field': ['field lines', 'electric', 'magnetic', 'flux', 'charge', 'dipole'],
            'kinematics': ['velocity', 'acceleration', 'trajectory', 'motion', 'path', 'v', 'a'],
            'optics': ['ray', 'lens', 'mirror', 'reflection', 'refraction', 'focal'],
            'thermodynamics': ['heat', 'temperature', 'pressure', 'volume', 'gas', 'Q', 'P', 'V']
        }
        
        # Common physics objects to detect
        self.physics_objects = {
            'block': 'rectangular object',
            'sphere': 'circular/spherical object',
            'incline': 'inclined plane',
            'spring': 'spring/elastic object',
            'pulley': 'circular pulley system',
            'pendulum': 'hanging mass system',
            'car': 'vehicle object',
            'projectile': 'launched object'
        }
        
        # Vector detection patterns
        self.vector_indicators = ['→', '↑', '↓', '←', '↗', '↘', '↙', '↖', 'arrow']
        
    def analyze_image(self, image_path: str) -> PhysicsDiagram:
        """Main method to analyze a physics diagram"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            logger.info(f"Analyzing physics diagram: {image_path}")
            
            # Perform different types of analysis
            elements = []
            
            # 1. Detect basic shapes and objects
            shapes = self._detect_shapes(image)
            elements.extend(shapes)
            
            # 2. Detect text and labels
            text_elements = self._detect_text(image)
            elements.extend(text_elements)
            
            # 3. Detect vectors and arrows
            vectors = self._detect_vectors(image)
            elements.extend(vectors)
            
            # 4. Detect lines and connections
            lines = self._detect_lines(image)
            elements.extend(lines)
            
            # Classify diagram type
            diagram_type = self._classify_diagram_type(elements, image_path)
            
            # Extract physics objects and concepts
            objects = self._identify_physics_objects(elements)
            concepts = self._identify_physics_concepts(elements, diagram_type)
            
            # Extract vector information
            vector_info = self._analyze_vectors(vectors, text_elements)
            
            # Extract labels
            labels = [elem.properties.get('text', '') for elem in text_elements if elem.properties.get('text')]
            
            # Calculate complexity
            complexity = self._calculate_diagram_complexity(elements, objects, concepts)
            
            # Create analysis metadata
            metadata = {
                'image_dimensions': image.shape[:2],
                'total_elements': len(elements),
                'processing_timestamp': None,  # Would be set by caller
                'analysis_confidence': self._calculate_overall_confidence(elements)
            }
            
            return PhysicsDiagram(
                image_path=image_path,
                diagram_type=diagram_type,
                elements=elements,
                objects=objects,
                vectors=vector_info,
                labels=labels,
                physics_concepts=concepts,
                complexity_score=complexity,
                analysis_metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error analyzing diagram {image_path}: {str(e)}")
            # Return empty diagram with error info
            return PhysicsDiagram(
                image_path=image_path,
                diagram_type='unknown',
                elements=[],
                objects=[],
                vectors=[],
                labels=[],
                physics_concepts=[],
                complexity_score=0,
                analysis_metadata={'error': str(e)}
            )
    
    def _detect_shapes(self, image: np.ndarray) -> List[DiagramElement]:
        """Detect basic geometric shapes that might represent physics objects"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangles (blocks, boxes)
        rectangles = self._detect_rectangles(gray)
        for rect in rectangles:
            x, y, w, h = rect
            elements.append(DiagramElement(
                element_type='rectangle',
                coordinates=(x, y, w, h),
                properties={'aspect_ratio': w/h, 'area': w*h},
                confidence=0.8
            ))
        
        # Detect circles (spheres, pulleys, wheels)
        circles = self._detect_circles(gray)
        for circle in circles:
            x, y, r = circle
            elements.append(DiagramElement(
                element_type='circle',
                coordinates=(x-r, y-r, 2*r, 2*r),
                properties={'radius': r, 'center': (x, y)},
                confidence=0.8
            ))
        
        # Detect triangles (inclines, wedges)
        triangles = self._detect_triangles(gray)
        for triangle in triangles:
            elements.append(DiagramElement(
                element_type='triangle',
                coordinates=triangle['bounds'],
                properties={'vertices': triangle['vertices']},
                confidence=0.7
            ))
        
        return elements
    
    def _detect_rectangles(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular shapes in the image"""
        rectangles = []
        
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 vertices)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # Filter by size and aspect ratio
                if w > 20 and h > 20 and 0.2 < w/h < 5:
                    rectangles.append((x, y, w, h))
        
        return rectangles
    
    def _detect_circles(self, gray_image: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circular shapes in the image"""
        circles = []
        
        # Use HoughCircles for circle detection
        detected_circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if detected_circles is not None:
            detected_circles = np.round(detected_circles[0, :]).astype("int")
            for (x, y, r) in detected_circles:
                circles.append((x, y, r))
        
        return circles
    
    def _detect_triangles(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect triangular shapes (inclines, wedges)"""
        triangles = []
        
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly triangular (3 vertices)
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(approx)
                if w > 20 and h > 20:  # Minimum size filter
                    triangles.append({
                        'vertices': approx.reshape(-1, 2).tolist(),
                        'bounds': (x, y, w, h)
                    })
        
        return triangles
    
    def _detect_text(self, image: np.ndarray) -> List[DiagramElement]:
        """Detect text labels in the diagram using OCR-like approach"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple text detection using contours
        # This is a basic approach - for production, would use pytesseract or similar
        
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text regions typically have certain aspect ratios
            aspect_ratio = w / h
            if 1 < aspect_ratio < 10 and w > 10 and h > 5:
                # Extract text region for OCR (simplified)
                text_region = gray[y:y+h, x:x+w]
                
                # Basic pattern matching for common physics symbols
                text_content = self._recognize_physics_text(text_region)
                
                elements.append(DiagramElement(
                    element_type='text',
                    coordinates=(x, y, w, h),
                    properties={'text': text_content, 'area': w*h},
                    confidence=0.6
                ))
        
        return elements
    
    def _recognize_physics_text(self, text_region: np.ndarray) -> str:
        """Basic physics text recognition (simplified OCR)"""
        # This is a simplified approach - in production would use proper OCR
        # For now, we'll return placeholder based on region characteristics
        
        height, width = text_region.shape
        
        # Very basic heuristics
        if width > height * 2:
            return "equation_or_label"
        elif width < height:
            return "variable"
        else:
            return "text"
    
    def _detect_vectors(self, image: np.ndarray) -> List[DiagramElement]:
        """Detect arrows and vectors in the diagram"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines first
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                
                # Check if this might be a vector (arrow)
                is_vector = self._is_likely_vector(gray, x1, y1, x2, y2)
                
                element_type = 'vector' if is_vector else 'line'
                
                elements.append(DiagramElement(
                    element_type=element_type,
                    coordinates=(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)),
                    properties={
                        'start_point': (x1, y1),
                        'end_point': (x2, y2),
                        'length': length,
                        'angle': angle,
                        'is_vector': is_vector
                    },
                    confidence=0.7 if is_vector else 0.5
                ))
        
        return elements
    
    def _is_likely_vector(self, gray_image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Determine if a line is likely to be a vector/arrow"""
        # Check for arrowhead at the end point
        # This is a simplified check - would be more sophisticated in practice
        
        # Look for triangular regions near the end point
        end_region_size = 10
        x_start = max(0, x2 - end_region_size)
        y_start = max(0, y2 - end_region_size)
        x_end = min(gray_image.shape[1], x2 + end_region_size)
        y_end = min(gray_image.shape[0], y2 + end_region_size)
        
        end_region = gray_image[y_start:y_end, x_start:x_end]
        
        # Simple heuristic: if there's more concentrated dark pixels near the end
        if end_region.size > 0:
            dark_pixel_ratio = np.sum(end_region < 128) / end_region.size
            return dark_pixel_ratio > 0.3
        
        return False
    
    def _detect_lines(self, image: np.ndarray) -> List[DiagramElement]:
        """Detect general lines that might represent connections or boundaries"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect straight lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Only include longer lines that aren't already classified as vectors
                if length > 25:
                    elements.append(DiagramElement(
                        element_type='line',
                        coordinates=(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)),
                        properties={
                            'start_point': (x1, y1),
                            'end_point': (x2, y2),
                            'length': length
                        },
                        confidence=0.6
                    ))
        
        return elements
    
    def _classify_diagram_type(self, elements: List[DiagramElement], image_path: str) -> str:
        """Classify the type of physics diagram"""
        # Check filename for hints
        filename = os.path.basename(image_path).lower()
        
        for diagram_type, keywords in self.diagram_patterns.items():
            if any(keyword in filename for keyword in keywords):
                return diagram_type
        
        # Analyze elements to classify
        vector_count = sum(1 for elem in elements if elem.element_type == 'vector')
        circle_count = sum(1 for elem in elements if elem.element_type == 'circle')
        rectangle_count = sum(1 for elem in elements if elem.element_type == 'rectangle')
        line_count = sum(1 for elem in elements if elem.element_type == 'line')
        
        # Classification heuristics
        if vector_count > 2 and rectangle_count >= 1:
            return 'free_body'
        elif circle_count > 1 and line_count > 3:
            return 'circuit'
        elif line_count > 5 and vector_count == 0:
            return 'field'
        elif vector_count > 0 and line_count > 2:
            return 'kinematics'
        else:
            return 'general_physics'
    
    def _identify_physics_objects(self, elements: List[DiagramElement]) -> List[str]:
        """Identify physics objects from detected elements"""
        objects = []
        
        for element in elements:
            if element.element_type == 'rectangle':
                # Could be a block, box, or similar
                aspect_ratio = element.properties.get('aspect_ratio', 1)
                if 0.7 < aspect_ratio < 1.3:
                    objects.append('block')
                elif aspect_ratio > 2:
                    objects.append('beam')
                else:
                    objects.append('object')
            
            elif element.element_type == 'circle':
                # Could be sphere, pulley, wheel
                radius = element.properties.get('radius', 0)
                if radius > 30:
                    objects.append('pulley')
                else:
                    objects.append('sphere')
            
            elif element.element_type == 'triangle':
                objects.append('incline')
        
        return list(set(objects))  # Remove duplicates
    
    def _identify_physics_concepts(self, elements: List[DiagramElement], diagram_type: str) -> List[str]:
        """Identify physics concepts represented in the diagram"""
        concepts = []
        
        # Base concepts from diagram type
        concept_mapping = {
            'free_body': ['force', 'equilibrium', 'newton_laws'],
            'circuit': ['current', 'voltage', 'resistance'],
            'wave': ['frequency', 'wavelength', 'amplitude'],
            'field': ['field_lines', 'electric_field', 'magnetic_field'],
            'kinematics': ['velocity', 'acceleration', 'motion'],
            'optics': ['reflection', 'refraction', 'focal_length']
        }
        
        concepts.extend(concept_mapping.get(diagram_type, []))
        
        # Additional concepts based on elements
        vector_count = sum(1 for elem in elements if elem.element_type == 'vector')
        if vector_count > 0:
            concepts.append('vectors')
        
        if any(elem.element_type == 'triangle' for elem in elements):
            concepts.append('inclined_plane')
        
        return list(set(concepts))
    
    def _analyze_vectors(self, vector_elements: List[DiagramElement], text_elements: List[DiagramElement]) -> List[Dict[str, Any]]:
        """Analyze vector properties and try to identify their physical meaning"""
        vectors = []
        
        for vector_elem in vector_elements:
            if vector_elem.element_type == 'vector':
                # Find nearby text that might label this vector
                vector_center = (
                    vector_elem.coordinates[0] + vector_elem.coordinates[2] // 2,
                    vector_elem.coordinates[1] + vector_elem.coordinates[3] // 2
                )
                
                nearby_text = self._find_nearby_text(vector_center, text_elements)
                
                # Classify vector type based on properties
                vector_type = self._classify_vector_type(vector_elem, nearby_text)
                
                vectors.append({
                    'coordinates': vector_elem.coordinates,
                    'properties': vector_elem.properties,
                    'type': vector_type,
                    'label': nearby_text,
                    'magnitude_relative': vector_elem.properties.get('length', 0)
                })
        
        return vectors
    
    def _find_nearby_text(self, point: Tuple[int, int], text_elements: List[DiagramElement], max_distance: int = 50) -> str:
        """Find text elements near a given point"""
        x, y = point
        
        for text_elem in text_elements:
            text_center = (
                text_elem.coordinates[0] + text_elem.coordinates[2] // 2,
                text_elem.coordinates[1] + text_elem.coordinates[3] // 2
            )
            
            distance = np.sqrt((x - text_center[0])**2 + (y - text_center[1])**2)
            if distance < max_distance:
                return text_elem.properties.get('text', '')
        
        return ''
    
    def _classify_vector_type(self, vector_elem: DiagramElement, label: str) -> str:
        """Classify the type of vector based on properties and label"""
        angle = vector_elem.properties.get('angle', 0)
        length = vector_elem.properties.get('length', 0)
        
        # Check label for hints
        label_lower = label.lower()
        if any(force_word in label_lower for force_word in ['f', 'force', 'weight', 'normal', 'friction']):
            return 'force'
        elif any(vel_word in label_lower for vel_word in ['v', 'velocity', 'speed']):
            return 'velocity'
        elif any(acc_word in label_lower for acc_word in ['a', 'acceleration']):
            return 'acceleration'
        
        # Use angle and length for classification
        if -15 < angle < 15 or 165 < abs(angle) < 195:  # Horizontal
            return 'horizontal_force'
        elif 75 < abs(angle) < 105:  # Vertical
            if angle > 0:
                return 'upward_force'
            else:
                return 'downward_force'
        else:
            return 'angled_force'
    
    def _calculate_diagram_complexity(self, elements: List[DiagramElement], objects: List[str], concepts: List[str]) -> int:
        """Calculate complexity score for the diagram"""
        score = 0
        
        # Base score from element count
        score += len(elements)
        
        # Additional score for physics objects
        score += len(objects) * 2
        
        # Additional score for concepts
        score += len(concepts) * 3
        
        # Bonus for vectors (indicate more complex physics)
        vector_count = sum(1 for elem in elements if elem.element_type == 'vector')
        score += vector_count * 2
        
        return score
    
    def _calculate_overall_confidence(self, elements: List[DiagramElement]) -> float:
        """Calculate overall confidence in the analysis"""
        if not elements:
            return 0.0
        
        total_confidence = sum(elem.confidence for elem in elements)
        return total_confidence / len(elements)
    
    def diagram_to_json(self, diagram: PhysicsDiagram) -> str:
        """Convert diagram analysis to JSON format"""
        diagram_dict = {
            'image_path': diagram.image_path,
            'diagram_type': diagram.diagram_type,
            'elements': [
                {
                    'element_type': elem.element_type,
                    'coordinates': elem.coordinates,
                    'properties': elem.properties,
                    'confidence': elem.confidence,
                    'description': elem.description
                }
                for elem in diagram.elements
            ],
            'objects': diagram.objects,
            'vectors': diagram.vectors,
            'labels': diagram.labels,
            'physics_concepts': diagram.physics_concepts,
            'complexity_score': diagram.complexity_score,
            'analysis_metadata': diagram.analysis_metadata
        }
        
        return json.dumps(diagram_dict, indent=2)
    
    def create_annotated_image(self, diagram: PhysicsDiagram, output_path: str) -> bool:
        """Create an annotated version of the image showing detected elements"""
        try:
            # Load original image
            image = cv2.imread(diagram.image_path)
            if image is None:
                return False
            
            # Draw detected elements
            for element in diagram.elements:
                x, y, w, h = element.coordinates
                
                if element.element_type == 'vector':
                    # Draw vector as arrow
                    start_point = element.properties.get('start_point', (x, y))
                    end_point = element.properties.get('end_point', (x+w, y+h))
                    cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 2)
                    cv2.putText(image, 'V', end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                elif element.element_type == 'rectangle':
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(image, 'OBJ', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                elif element.element_type == 'circle':
                    center = element.properties.get('center', (x+w//2, y+h//2))
                    radius = element.properties.get('radius', w//2)
                    cv2.circle(image, center, radius, (0, 0, 255), 2)
                    cv2.putText(image, 'CIR', (center[0]-10, center[1]-radius-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                elif element.element_type == 'text':
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 1)
                    cv2.putText(image, 'TXT', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Add diagram type annotation
            cv2.putText(image, f"Type: {diagram.diagram_type}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Complexity: {diagram.complexity_score}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save annotated image
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            logger.error(f"Error creating annotated image: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    analyzer = PhysicsDiagramAnalyzer()
    
    # Note: This would require actual image files to test
    print("Physics Diagram Analyzer initialized")
    print("Available diagram types:", list(analyzer.diagram_patterns.keys()))
    print("Physics objects detection:", list(analyzer.physics_objects.keys()))