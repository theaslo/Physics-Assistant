#!/usr/bin/env python3
"""
Educational Content Classifier for Physics Materials
Classifies physics educational content into problems, solutions, explanations, definitions, and examples.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import spacy
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of educational content"""
    PROBLEM = "problem"
    SOLUTION = "solution" 
    EXPLANATION = "explanation"
    DEFINITION = "definition"
    EXAMPLE = "example"
    THEOREM = "theorem"
    FORMULA = "formula"
    REFERENCE = "reference"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    """Result of content classification"""
    content_type: ContentType
    confidence_score: float
    sub_type: Optional[str]  # e.g., "word_problem", "numerical_calculation"
    difficulty_indicators: List[str]
    physics_domain: Optional[str]
    pedagogical_features: Dict[str, Any]
    classification_evidence: Dict[str, Any]

@dataclass
class EducationalSegment:
    """A segment of educational content with classification"""
    segment_id: str
    text: str
    classification: ClassificationResult
    mathematical_content: List[str]  # LaTeX equations found
    start_position: int
    end_position: int
    context: Dict[str, Any]

class PhysicsContentClassifier:
    """Advanced classifier for physics educational content"""
    
    def __init__(self):
        # Try to load spaCy model, fallback to basic patterns if not available
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except OSError:
            logger.warning("spaCy model not found, using basic pattern matching")
            self.nlp = None
            self.use_spacy = False
        
        # Problem identification patterns
        self.problem_patterns = {
            'direct_question': [
                r'what\s+is\s+the\s+',
                r'find\s+the\s+',
                r'calculate\s+the\s+',
                r'determine\s+the\s+',
                r'how\s+fast\s+',
                r'how\s+far\s+',
                r'how\s+long\s+',
                r'at\s+what\s+',
            ],
            'scenario_setup': [
                r'a\s+\d+\s*kg\s+',
                r'a\s+car\s+',
                r'a\s+ball\s+',
                r'a\s+block\s+',
                r'a\s+mass\s+',
                r'consider\s+a\s+',
                r'suppose\s+',
                r'given\s+that\s+',
            ],
            'numerical_problem': [
                r'\d+\s*m/s',
                r'\d+\s*kg',
                r'\d+\s*N',
                r'\d+\s*J',
                r'\d+\s*°',
                r'\d+\s*Hz',
                r'\d+\s*m',
            ]
        }
        
        # Solution identification patterns
        self.solution_patterns = {
            'solution_markers': [
                r'solution:?',
                r'answer:?',
                r'step\s+\d+',
                r'first,?\s+',
                r'next,?\s+',
                r'then,?\s+',
                r'finally,?\s+',
                r'therefore,?\s+',
                r'thus,?\s+',
            ],
            'calculation_steps': [
                r'substituting\s+',
                r'using\s+the\s+equation',
                r'from\s+the\s+formula',
                r'applying\s+',
                r'we\s+get\s+',
                r'this\s+gives\s+',
                r'solving\s+for\s+',
            ],
            'result_indicators': [
                r'=\s*\d+',
                r'the\s+answer\s+is\s+',
                r'the\s+result\s+is\s+',
                r'we\s+find\s+that\s+',
            ]
        }
        
        # Explanation identification patterns
        self.explanation_patterns = {
            'concept_explanations': [
                r'explanation:?',
                r'note\s+that\s+',
                r'remember\s+that\s+',
                r'it\s+is\s+important\s+to\s+',
                r'this\s+means\s+that\s+',
                r'in\s+other\s+words\s+',
                r'concept:?\s+',
            ],
            'reasoning': [
                r'because\s+',
                r'since\s+',
                r'due\s+to\s+',
                r'as\s+a\s+result\s+',
                r'consequently\s+',
                r'this\s+is\s+why\s+',
            ],
            'clarifications': [
                r'clarification:?',
                r'to\s+understand\s+',
                r'let\s+us\s+consider\s+',
                r'for\s+example\s+',
                r'imagine\s+',
            ]
        }
        
        # Definition patterns
        self.definition_patterns = [
            r'is\s+defined\s+as\s+',
            r'definition:?\s+',
            r'means\s+',
            r'refers\s+to\s+',
            r'is\s+the\s+',
            r'can\s+be\s+described\s+as\s+',
        ]
        
        # Physics domains and their indicators
        self.physics_domains = {
            'mechanics': [
                'force', 'velocity', 'acceleration', 'momentum', 'energy', 'work', 'power',
                'motion', 'newton', 'friction', 'gravity', 'mass', 'displacement'
            ],
            'waves': [
                'wave', 'frequency', 'wavelength', 'amplitude', 'oscillation', 'harmonic',
                'period', 'resonance', 'interference', 'standing wave'
            ],
            'thermodynamics': [
                'heat', 'temperature', 'entropy', 'gas', 'pressure', 'volume',
                'thermal', 'calorimetry', 'phase', 'ideal gas'
            ],
            'electromagnetism': [
                'electric', 'magnetic', 'field', 'charge', 'current', 'voltage',
                'capacitor', 'resistor', 'electromagnetic', 'induction'
            ],
            'optics': [
                'light', 'lens', 'mirror', 'reflection', 'refraction', 'ray',
                'focal', 'optical', 'spectrum', 'interference'
            ],
            'modern_physics': [
                'quantum', 'relativity', 'photon', 'electron', 'atom', 'nuclear',
                'particle', 'radiation', 'photoelectric'
            ]
        }
        
        # Difficulty indicators
        self.difficulty_indicators = {
            'beginner': [
                'basic', 'simple', 'elementary', 'introductory', 'fundamental'
            ],
            'intermediate': [
                'intermediate', 'moderate', 'standard', 'typical', 'common'
            ],
            'advanced': [
                'advanced', 'complex', 'sophisticated', 'challenging', 'difficult',
                'differential', 'integral', 'tensor', 'relativistic', 'quantum'
            ]
        }
        
        logger.info("Physics Content Classifier initialized")
    
    def classify_content(self, text: str, context: Dict[str, Any] = None) -> ClassificationResult:
        """Main method to classify educational content"""
        context = context or {}
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract features
        features = self._extract_features(cleaned_text, context)
        
        # Classify content type
        content_type, confidence, evidence = self._classify_content_type(features, cleaned_text)
        
        # Determine sub-type
        sub_type = self._determine_sub_type(content_type, features, cleaned_text)
        
        # Identify physics domain
        physics_domain = self._identify_physics_domain(cleaned_text, features)
        
        # Extract pedagogical features
        pedagogical_features = self._extract_pedagogical_features(cleaned_text, content_type)
        
        # Identify difficulty indicators
        difficulty_indicators = self._identify_difficulty_indicators(cleaned_text)
        
        return ClassificationResult(
            content_type=content_type,
            confidence_score=confidence,
            sub_type=sub_type,
            difficulty_indicators=difficulty_indicators,
            physics_domain=physics_domain,
            pedagogical_features=pedagogical_features,
            classification_evidence=evidence
        )
    
    def segment_and_classify(self, document_text: str) -> List[EducationalSegment]:
        """Segment document and classify each segment"""
        segments = self._segment_document(document_text)
        classified_segments = []
        
        for i, (text, start_pos, end_pos) in enumerate(segments):
            # Extract mathematical content
            math_content = self._extract_mathematical_content(text)
            
            # Build context from surrounding segments
            context = {
                'segment_index': i,
                'total_segments': len(segments),
                'has_math': len(math_content) > 0,
                'segment_length': len(text)
            }
            
            # Classify segment
            classification = self.classify_content(text, context)
            
            classified_segments.append(EducationalSegment(
                segment_id=f"seg_{i:03d}",
                text=text,
                classification=classification,
                mathematical_content=math_content,
                start_position=start_pos,
                end_position=end_pos,
                context=context
            ))
        
        logger.info(f"Segmented and classified {len(classified_segments)} content segments")
        return classified_segments
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for classification"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize some common physics notation
        cleaned = re.sub(r'(\d+)\s*×\s*(\d+)', r'\1*\2', cleaned)
        cleaned = re.sub(r'(\d+)\s*÷\s*(\d+)', r'\1/\2', cleaned)
        
        return cleaned
    
    def _extract_features(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for classification"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'numbers': len(re.findall(r'\d+', text)),
            'equations': len(re.findall(r'[=<>]', text)),
            'units': len(re.findall(r'\d+\s*[a-zA-Z]+', text)),
            'parentheses': text.count('('),
            'capital_letters': sum(1 for c in text if c.isupper()),
        }
        
        # Add linguistic features if spaCy is available
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            features.update({
                'verbs': len([token for token in doc if token.pos_ == 'VERB']),
                'nouns': len([token for token in doc if token.pos_ == 'NOUN']),
                'adjectives': len([token for token in doc if token.pos_ == 'ADJ']),
                'entities': len(doc.ents),
            })
        
        # Pattern matching features
        features.update(self._count_pattern_matches(text))
        
        return features
    
    def _count_pattern_matches(self, text: str) -> Dict[str, int]:
        """Count matches for various pattern categories"""
        text_lower = text.lower()
        
        counts = {
            'problem_indicators': 0,
            'solution_indicators': 0,
            'explanation_indicators': 0,
            'definition_indicators': 0,
            'calculation_indicators': 0,
        }
        
        # Count problem patterns
        for category, patterns in self.problem_patterns.items():
            for pattern in patterns:
                counts['problem_indicators'] += len(re.findall(pattern, text_lower))
        
        # Count solution patterns
        for category, patterns in self.solution_patterns.items():
            for pattern in patterns:
                counts['solution_indicators'] += len(re.findall(pattern, text_lower))
        
        # Count explanation patterns
        for category, patterns in self.explanation_patterns.items():
            for pattern in patterns:
                counts['explanation_indicators'] += len(re.findall(pattern, text_lower))
        
        # Count definition patterns
        for pattern in self.definition_patterns:
            counts['definition_indicators'] += len(re.findall(pattern, text_lower))
        
        # Count calculation indicators
        calc_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'=\s*\d+', r'substituting', r'solving']
        for pattern in calc_patterns:
            counts['calculation_indicators'] += len(re.findall(pattern, text_lower))
        
        return counts
    
    def _classify_content_type(self, features: Dict[str, Any], text: str) -> Tuple[ContentType, float, Dict[str, Any]]:
        """Classify the content type using rule-based approach"""
        scores = {content_type: 0.0 for content_type in ContentType}
        evidence = {}
        
        # Rule-based scoring
        
        # Problem detection
        problem_score = 0
        if features['question_marks'] > 0:
            problem_score += 0.3
        if features['problem_indicators'] > 0:
            problem_score += 0.4 * min(features['problem_indicators'], 3)
        if features['numbers'] > 2 and features['units'] > 0:
            problem_score += 0.2
        
        scores[ContentType.PROBLEM] = min(problem_score, 1.0)
        evidence['problem_score_breakdown'] = {
            'question_marks': features['question_marks'],
            'problem_indicators': features['problem_indicators'],
            'numerical_content': features['numbers'] > 2 and features['units'] > 0
        }
        
        # Solution detection
        solution_score = 0
        if features['solution_indicators'] > 0:
            solution_score += 0.5 * min(features['solution_indicators'], 2)
        if features['calculation_indicators'] > 0:
            solution_score += 0.3 * min(features['calculation_indicators'], 2)
        if features['equations'] > 1:
            solution_score += 0.2
        
        scores[ContentType.SOLUTION] = min(solution_score, 1.0)
        
        # Explanation detection
        explanation_score = 0
        if features['explanation_indicators'] > 0:
            explanation_score += 0.4 * min(features['explanation_indicators'], 2)
        if features['word_count'] > 50 and features['question_marks'] == 0:
            explanation_score += 0.2
        if 'because' in text.lower() or 'since' in text.lower():
            explanation_score += 0.3
        
        scores[ContentType.EXPLANATION] = min(explanation_score, 1.0)
        
        # Definition detection
        definition_score = 0
        if features['definition_indicators'] > 0:
            definition_score += 0.6 * min(features['definition_indicators'], 2)
        if 'is defined as' in text.lower() or 'means' in text.lower():
            definition_score += 0.4
        
        scores[ContentType.DEFINITION] = min(definition_score, 1.0)
        
        # Formula detection
        formula_score = 0
        if features['equations'] > 0 and features['word_count'] < 20:
            formula_score += 0.5
        if re.search(r'^[A-Za-z]\s*=', text.strip()):
            formula_score += 0.4
        
        scores[ContentType.FORMULA] = min(formula_score, 1.0)
        
        # Example detection
        example_score = 0
        if 'example' in text.lower() or 'for instance' in text.lower():
            example_score += 0.4
        if features['problem_indicators'] > 0 and features['solution_indicators'] > 0:
            example_score += 0.3
        
        scores[ContentType.EXAMPLE] = min(example_score, 1.0)
        
        # Find the highest scoring type
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # If no clear winner, classify as unknown
        if confidence < 0.3:
            best_type = ContentType.UNKNOWN
            confidence = 0.1
        
        evidence['all_scores'] = scores
        evidence['winning_type'] = best_type.value
        evidence['confidence'] = confidence
        
        return best_type, confidence, evidence
    
    def _determine_sub_type(self, content_type: ContentType, features: Dict[str, Any], text: str) -> Optional[str]:
        """Determine sub-type based on content type and features"""
        text_lower = text.lower()
        
        if content_type == ContentType.PROBLEM:
            if features['numbers'] > 3:
                return "numerical_problem"
            elif any(word in text_lower for word in ['explain', 'describe', 'discuss']):
                return "conceptual_problem"
            elif features['question_marks'] > 0:
                return "direct_question"
            else:
                return "word_problem"
        
        elif content_type == ContentType.SOLUTION:
            if features['calculation_indicators'] > 2:
                return "step_by_step_calculation"
            elif 'therefore' in text_lower or 'thus' in text_lower:
                return "logical_derivation"
            else:
                return "general_solution"
        
        elif content_type == ContentType.EXPLANATION:
            if any(word in text_lower for word in ['concept', 'principle', 'law']):
                return "conceptual_explanation"
            elif 'because' in text_lower or 'reason' in text_lower:
                return "causal_explanation"
            else:
                return "descriptive_explanation"
        
        return None
    
    def _identify_physics_domain(self, text: str, features: Dict[str, Any]) -> Optional[str]:
        """Identify the physics domain of the content"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.physics_domains.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score / len(keywords)
        
        if domain_scores:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        
        return None
    
    def _extract_pedagogical_features(self, text: str, content_type: ContentType) -> Dict[str, Any]:
        """Extract pedagogical features from the content"""
        text_lower = text.lower()
        
        features = {
            'has_worked_example': False,
            'has_step_by_step': False,
            'has_conceptual_discussion': False,
            'has_real_world_context': False,
            'has_multiple_approaches': False,
            'interactive_elements': [],
            'learning_objectives': [],
            'prerequisites': []
        }
        
        # Check for worked example
        if content_type == ContentType.EXAMPLE or ('example' in text_lower and 'solution' in text_lower):
            features['has_worked_example'] = True
        
        # Check for step-by-step approach
        step_indicators = ['step 1', 'step 2', 'first,', 'second,', 'then,', 'finally,']
        if any(indicator in text_lower for indicator in step_indicators):
            features['has_step_by_step'] = True
        
        # Check for conceptual discussion
        concept_words = ['understand', 'concept', 'principle', 'why', 'meaning', 'significance']
        if any(word in text_lower for word in concept_words):
            features['has_conceptual_discussion'] = True
        
        # Check for real-world context
        real_world_indicators = ['everyday', 'real life', 'practical', 'application', 'example from']
        if any(indicator in text_lower for indicator in real_world_indicators):
            features['has_real_world_context'] = True
        
        # Check for multiple approaches
        if 'alternatively' in text_lower or 'another way' in text_lower or 'method' in text_lower:
            features['has_multiple_approaches'] = True
        
        # Extract interactive elements
        interactive_patterns = [
            r'try\s+this',
            r'exercise\s*:',
            r'practice\s+problem',
            r'check\s+your\s+understanding',
            r'think\s+about',
        ]
        
        for pattern in interactive_patterns:
            if re.search(pattern, text_lower):
                features['interactive_elements'].append(pattern)
        
        return features
    
    def _identify_difficulty_indicators(self, text: str) -> List[str]:
        """Identify difficulty level indicators in the text"""
        text_lower = text.lower()
        indicators = []
        
        for level, words in self.difficulty_indicators.items():
            for word in words:
                if word in text_lower:
                    indicators.append(f"{level}_{word}")
        
        # Additional complexity indicators
        complexity_patterns = [
            (r'differential\s+equation', 'advanced_mathematics'),
            (r'integral\s+', 'advanced_mathematics'),
            (r'vector\s+', 'intermediate_mathematics'),
            (r'matrix\s+', 'advanced_mathematics'),
            (r'tensor\s+', 'advanced_physics'),
            (r'quantum\s+', 'advanced_physics'),
            (r'relativistic\s+', 'advanced_physics'),
        ]
        
        for pattern, indicator in complexity_patterns:
            if re.search(pattern, text_lower):
                indicators.append(indicator)
        
        return indicators
    
    def _segment_document(self, text: str) -> List[Tuple[str, int, int]]:
        """Segment document into meaningful educational units"""
        segments = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_pos = 0
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                start_pos = text.find(paragraph, current_pos)
                end_pos = start_pos + len(paragraph)
                
                # Further split long paragraphs by sentences if needed
                if len(paragraph) > 500:
                    sentences = re.split(r'[.!?]+', paragraph)
                    sentence_pos = start_pos
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) > 20:
                            sent_start = text.find(sentence, sentence_pos)
                            sent_end = sent_start + len(sentence)
                            segments.append((sentence, sent_start, sent_end))
                            sentence_pos = sent_end
                else:
                    segments.append((paragraph, start_pos, end_pos))
                
                current_pos = end_pos
        
        return segments
    
    def _extract_mathematical_content(self, text: str) -> List[str]:
        """Extract mathematical content from text"""
        math_content = []
        
        # LaTeX patterns
        latex_patterns = [
            r'\$([^$]+)\$',  # Inline math
            r'\$\$([^$]+)\$\$',  # Display math
            r'\\begin\{equation\}(.+?)\\end\{equation\}',  # Equation environment
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            math_content.extend(matches)
        
        # Basic equation patterns
        equation_patterns = [
            r'[A-Za-z]\s*=\s*[^=\n]+',  # Variable = expression
            r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+',  # Arithmetic
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            math_content.extend(matches)
        
        return math_content
    
    def get_classification_summary(self, segments: List[EducationalSegment]) -> Dict[str, Any]:
        """Get summary statistics of classification results"""
        type_counts = Counter(seg.classification.content_type for seg in segments)
        domain_counts = Counter(seg.classification.physics_domain for seg in segments if seg.classification.physics_domain)
        
        avg_confidence = sum(seg.classification.confidence_score for seg in segments) / len(segments) if segments else 0
        
        difficulty_indicators = []
        for seg in segments:
            difficulty_indicators.extend(seg.classification.difficulty_indicators)
        
        difficulty_counts = Counter(difficulty_indicators)
        
        pedagogical_features = {
            'worked_examples': sum(1 for seg in segments if seg.classification.pedagogical_features.get('has_worked_example', False)),
            'step_by_step': sum(1 for seg in segments if seg.classification.pedagogical_features.get('has_step_by_step', False)),
            'conceptual_discussion': sum(1 for seg in segments if seg.classification.pedagogical_features.get('has_conceptual_discussion', False)),
            'real_world_context': sum(1 for seg in segments if seg.classification.pedagogical_features.get('has_real_world_context', False)),
        }
        
        return {
            'total_segments': len(segments),
            'content_type_distribution': dict(type_counts),
            'physics_domain_distribution': dict(domain_counts),
            'average_confidence': avg_confidence,
            'difficulty_indicators': dict(difficulty_counts),
            'pedagogical_features': pedagogical_features,
            'segments_with_math': sum(1 for seg in segments if seg.mathematical_content),
        }
    
    def export_classification_results(self, segments: List[EducationalSegment], output_path: str) -> bool:
        """Export classification results to JSON file"""
        try:
            results = {
                'metadata': {
                    'total_segments': len(segments),
                    'classification_timestamp': None,  # Would be set by caller
                    'classifier_version': '1.0'
                },
                'summary': self.get_classification_summary(segments),
                'segments': [
                    {
                        'segment_id': seg.segment_id,
                        'text_preview': seg.text[:200] + "..." if len(seg.text) > 200 else seg.text,
                        'content_type': seg.classification.content_type.value,
                        'confidence_score': seg.classification.confidence_score,
                        'sub_type': seg.classification.sub_type,
                        'physics_domain': seg.classification.physics_domain,
                        'difficulty_indicators': seg.classification.difficulty_indicators,
                        'pedagogical_features': seg.classification.pedagogical_features,
                        'mathematical_content_count': len(seg.mathematical_content),
                        'position': {
                            'start': seg.start_position,
                            'end': seg.end_position
                        }
                    }
                    for seg in segments
                ]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Classification results exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting classification results: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    classifier = PhysicsContentClassifier()
    
    # Test with sample physics content
    sample_texts = [
        "A 5 kg block slides down a 30° incline. What is the acceleration of the block if the coefficient of friction is 0.2?",
        "Solution: First, we identify the forces acting on the block. The weight mg acts vertically downward...",
        "Newton's second law states that the net force on an object is equal to the mass times acceleration.",
        "Velocity is defined as the rate of change of position with respect to time.",
        "F = ma",
    ]
    
    print("Physics Content Classifier Test Results:")
    print("=" * 50)
    
    for i, text in enumerate(sample_texts, 1):
        result = classifier.classify_content(text)
        print(f"\n{i}. Text: {text[:50]}...")
        print(f"   Type: {result.content_type.value}")
        print(f"   Confidence: {result.confidence_score:.2f}")
        print(f"   Sub-type: {result.sub_type}")
        print(f"   Domain: {result.physics_domain}")
        print(f"   Difficulty indicators: {result.difficulty_indicators}")