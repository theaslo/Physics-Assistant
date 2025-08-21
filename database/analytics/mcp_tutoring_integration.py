#!/usr/bin/env python3
"""
MCP Physics Tools Integration for Phase 6.2 Intelligent Tutoring
Connects adaptive tutoring system with MCP physics calculation tools
for real-time problem generation and validation.
"""

import asyncio
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPPhysicsResult:
    """Result from MCP physics calculation"""
    concept: str
    input_values: Dict[str, float]
    calculated_values: Dict[str, float]
    calculation_steps: List[str]
    equations_used: List[str]
    success: bool
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

@dataclass
class AdaptiveProblemTemplate:
    """Template for generating adaptive physics problems"""
    concept: str
    difficulty_level: float
    problem_type: str  # "calculation", "conceptual", "application"
    variable_ranges: Dict[str, tuple]  # min, max values for variables
    required_calculations: List[str]
    learning_objectives: List[str]
    common_misconceptions: List[str]

class MCPPhysicsTutoringIntegration:
    """Integration layer between intelligent tutoring and MCP physics tools"""
    
    def __init__(self, mcp_base_url: str = "http://localhost:8000"):
        self.mcp_base_url = mcp_base_url
        self.tool_endpoints = {
            'kinematics': '/mcp/kinematics',
            'forces': '/mcp/forces',
            'energy': '/mcp/energy',
            'momentum': '/mcp/momentum',
            'angular_motion': '/mcp/angular_motion',
            'math': '/mcp/math'
        }
        
        # Problem templates for different concepts and difficulty levels
        self.problem_templates = self._initialize_problem_templates()
        
        # Cache for MCP calculation results
        self.calculation_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _initialize_problem_templates(self) -> Dict[str, Dict[str, List[AdaptiveProblemTemplate]]]:
        """Initialize adaptive problem templates for physics concepts"""
        return {
            'kinematics_1d': {
                'beginner': [
                    AdaptiveProblemTemplate(
                        concept='kinematics_1d',
                        difficulty_level=0.3,
                        problem_type='calculation',
                        variable_ranges={
                            'initial_velocity': (0, 20),
                            'acceleration': (1, 5),
                            'time': (2, 10)
                        },
                        required_calculations=['final_velocity', 'displacement'],
                        learning_objectives=['Apply kinematic equations', 'Understand motion with constant acceleration'],
                        common_misconceptions=['Velocity vs acceleration confusion', 'Direction interpretation']
                    )
                ],
                'intermediate': [
                    AdaptiveProblemTemplate(
                        concept='kinematics_1d',
                        difficulty_level=0.6,
                        problem_type='application',
                        variable_ranges={
                            'initial_velocity': (10, 50),
                            'acceleration': (-5, 10),
                            'displacement': (50, 500)
                        },
                        required_calculations=['time', 'final_velocity'],
                        learning_objectives=['Solve multi-step kinematics problems', 'Handle negative acceleration'],
                        common_misconceptions=['Sign conventions', 'Multiple solution scenarios']
                    )
                ]
            },
            'forces': {
                'beginner': [
                    AdaptiveProblemTemplate(
                        concept='forces',
                        difficulty_level=0.4,
                        problem_type='calculation',
                        variable_ranges={
                            'mass': (1, 50),
                            'acceleration': (0.5, 10),
                            'friction_coefficient': (0.1, 0.8)
                        },
                        required_calculations=['net_force', 'friction_force'],
                        learning_objectives=['Apply Newton\'s second law', 'Calculate friction forces'],
                        common_misconceptions=['Force vs acceleration', 'Static vs kinetic friction']
                    )
                ],
                'intermediate': [
                    AdaptiveProblemTemplate(
                        concept='forces',
                        difficulty_level=0.7,
                        problem_type='application',
                        variable_ranges={
                            'mass': (10, 100),
                            'angle': (15, 60),
                            'coefficient_friction': (0.2, 0.6)
                        },
                        required_calculations=['normal_force', 'friction_force', 'acceleration'],
                        learning_objectives=['Analyze forces on inclined planes', 'Apply force component analysis'],
                        common_misconceptions=['Force components', 'Normal force on inclines']
                    )
                ]
            },
            'energy': {
                'intermediate': [
                    AdaptiveProblemTemplate(
                        concept='energy',
                        difficulty_level=0.6,
                        problem_type='application',
                        variable_ranges={
                            'mass': (1, 20),
                            'height': (1, 50),
                            'velocity': (5, 30)
                        },
                        required_calculations=['kinetic_energy', 'potential_energy', 'total_energy'],
                        learning_objectives=['Apply conservation of energy', 'Calculate energy transformations'],
                        common_misconceptions=['Energy types', 'Conservation violations']
                    )
                ]
            }
        }
    
    async def generate_adaptive_physics_problem(self, concept: str, difficulty: float,
                                              student_knowledge: Dict[str, float],
                                              learning_style: str) -> Optional[Dict[str, Any]]:
        """Generate adaptive physics problem using MCP tools"""
        try:
            start_time = time.time()
            
            # Select appropriate template
            template = await self._select_problem_template(concept, difficulty)
            if not template:
                logger.warning(f"No template found for concept {concept} at difficulty {difficulty}")
                return None
            
            # Generate problem variables
            problem_variables = await self._generate_problem_variables(template, difficulty)
            
            # Use MCP tools to calculate solution
            mcp_result = await self._calculate_with_mcp(concept, problem_variables)
            if not mcp_result.success:
                logger.error(f"MCP calculation failed: {mcp_result.error_message}")
                return None
            
            # Generate problem text
            problem_text = await self._generate_problem_text(template, problem_variables)
            
            # Generate adaptive hints based on learning style
            hints = await self._generate_adaptive_hints(template, learning_style, problem_variables)
            
            # Create problem response
            adaptive_problem = {
                'problem_id': f"{concept}_{int(time.time())}",
                'concept': concept,
                'difficulty': difficulty,
                'problem_type': template.problem_type,
                'content': problem_text,
                'variables': problem_variables,
                'solution': mcp_result.calculated_values,
                'calculation_steps': mcp_result.calculation_steps,
                'equations_used': mcp_result.equations_used,
                'hints': hints,
                'learning_objectives': template.learning_objectives,
                'misconceptions_addressed': template.common_misconceptions,
                'estimated_time_minutes': self._estimate_problem_time(template, difficulty),
                'mcp_integration': {
                    'tool_used': concept,
                    'calculation_time_ms': mcp_result.execution_time_ms,
                    'validation_passed': True
                },
                'generation_time_ms': (time.time() - start_time) * 1000
            }
            
            logger.info(f"Generated adaptive problem for {concept} in {adaptive_problem['generation_time_ms']:.1f}ms")
            return adaptive_problem
            
        except Exception as e:
            logger.error(f"❌ Failed to generate adaptive physics problem: {e}")
            return None
    
    async def validate_student_answer(self, problem_data: Dict[str, Any], 
                                    student_answer: str) -> Dict[str, Any]:
        """Validate student answer using MCP physics tools"""
        try:
            start_time = time.time()
            
            concept = problem_data['concept']
            problem_variables = problem_data['variables']
            expected_solution = problem_data['solution']
            
            # Parse student answer
            parsed_answer = await self._parse_student_answer(student_answer, concept)
            
            # Validate using MCP tools
            validation_result = await self._validate_with_mcp(
                concept, problem_variables, parsed_answer, expected_solution
            )
            
            # Analyze misconceptions
            misconceptions_detected = await self._detect_misconceptions(
                concept, parsed_answer, expected_solution, problem_data.get('misconceptions_addressed', [])
            )
            
            # Generate feedback
            feedback = await self._generate_validation_feedback(
                validation_result, misconceptions_detected, concept
            )
            
            validation_response = {
                'is_correct': validation_result['is_correct'],
                'correctness_score': validation_result['score'],
                'student_parsed_answer': parsed_answer,
                'expected_answer': expected_solution,
                'misconceptions_detected': misconceptions_detected,
                'feedback': feedback,
                'partial_credit_areas': validation_result.get('partial_credit', []),
                'next_step_hint': validation_result.get('next_step_hint', ''),
                'validation_time_ms': (time.time() - start_time) * 1000
            }
            
            return validation_response
            
        except Exception as e:
            logger.error(f"❌ Failed to validate student answer: {e}")
            return {
                'is_correct': False,
                'correctness_score': 0.0,
                'error': str(e),
                'validation_time_ms': 0.0
            }
    
    async def _select_problem_template(self, concept: str, difficulty: float) -> Optional[AdaptiveProblemTemplate]:
        """Select appropriate problem template based on concept and difficulty"""
        try:
            concept_templates = self.problem_templates.get(concept, {})
            
            # Determine difficulty level
            if difficulty < 0.4:
                level = 'beginner'
            elif difficulty < 0.7:
                level = 'intermediate'
            else:
                level = 'advanced'
            
            # Get templates for difficulty level
            level_templates = concept_templates.get(level, concept_templates.get('intermediate', []))
            
            if not level_templates:
                return None
            
            # Select template (for now, just pick first one)
            return level_templates[0]
            
        except Exception as e:
            logger.error(f"❌ Failed to select problem template: {e}")
            return None
    
    async def _generate_problem_variables(self, template: AdaptiveProblemTemplate, 
                                        difficulty: float) -> Dict[str, float]:
        """Generate problem variables within template ranges"""
        try:
            import random
            
            variables = {}
            
            for var_name, (min_val, max_val) in template.variable_ranges.items():
                # Adjust range based on difficulty
                range_size = max_val - min_val
                difficulty_adjustment = difficulty * range_size * 0.2
                
                adjusted_min = min_val + difficulty_adjustment * 0.3
                adjusted_max = max_val - difficulty_adjustment * 0.3
                
                # Generate value
                value = random.uniform(adjusted_min, adjusted_max)
                
                # Round appropriately based on variable type
                if var_name in ['mass', 'time']:
                    variables[var_name] = round(value, 1)
                elif var_name in ['angle']:
                    variables[var_name] = round(value)
                else:
                    variables[var_name] = round(value, 2)
            
            return variables
            
        except Exception as e:
            logger.error(f"❌ Failed to generate problem variables: {e}")
            return {}
    
    async def _calculate_with_mcp(self, concept: str, variables: Dict[str, float]) -> MCPPhysicsResult:
        """Perform physics calculations using MCP tools"""
        try:
            start_time = time.time()
            
            endpoint = self.tool_endpoints.get(concept)
            if not endpoint:
                return MCPPhysicsResult(
                    concept=concept,
                    input_values=variables,
                    calculated_values={},
                    calculation_steps=[],
                    equations_used=[],
                    success=False,
                    error_message=f"No MCP endpoint for concept: {concept}"
                )
            
            # Check cache first
            cache_key = f"{concept}_{hash(str(sorted(variables.items())))}"
            if cache_key in self.calculation_cache:
                cached_result = self.calculation_cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    return cached_result['result']
            
            # Prepare MCP request
            mcp_request = {
                'operation': 'calculate',
                'parameters': variables,
                'return_steps': True
            }
            
            # Make request to MCP service
            try:
                response = requests.post(
                    f"{self.mcp_base_url}{endpoint}",
                    json=mcp_request,
                    timeout=5.0
                )
                response.raise_for_status()
                mcp_data = response.json()
                
                # Parse MCP response
                result = MCPPhysicsResult(
                    concept=concept,
                    input_values=variables,
                    calculated_values=mcp_data.get('results', {}),
                    calculation_steps=mcp_data.get('steps', []),
                    equations_used=mcp_data.get('equations', []),
                    success=True,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                
                # Cache result
                self.calculation_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                return result
                
            except requests.RequestException as e:
                logger.error(f"MCP request failed: {e}")
                return MCPPhysicsResult(
                    concept=concept,
                    input_values=variables,
                    calculated_values={},
                    calculation_steps=[],
                    equations_used=[],
                    success=False,
                    error_message=f"MCP service unavailable: {str(e)}"
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate with MCP: {e}")
            return MCPPhysicsResult(
                concept=concept,
                input_values=variables,
                calculated_values={},
                calculation_steps=[],
                equations_used=[],
                success=False,
                error_message=str(e)
            )
    
    async def _generate_problem_text(self, template: AdaptiveProblemTemplate, 
                                   variables: Dict[str, float]) -> str:
        """Generate human-readable problem text"""
        try:
            if template.concept == 'kinematics_1d':
                if 'initial_velocity' in variables and 'acceleration' in variables and 'time' in variables:
                    return f"""A car starts with an initial velocity of {variables['initial_velocity']} m/s and accelerates at {variables['acceleration']} m/s² for {variables['time']} seconds.
                    
Calculate:
1. The final velocity of the car
2. The displacement during this time

Show your work and include units in your answer."""

            elif template.concept == 'forces':
                if 'mass' in variables and 'acceleration' in variables:
                    return f"""A {variables['mass']} kg object is accelerating at {variables['acceleration']} m/s².
                    
Calculate:
1. The net force acting on the object
2. If there is friction with coefficient μ = {variables.get('friction_coefficient', 0.3)}, find the friction force

Show your work and include units in your answer."""

            elif template.concept == 'energy':
                if 'mass' in variables and 'height' in variables:
                    return f"""A {variables['mass']} kg object is at a height of {variables['height']} m above the ground.
                    
Calculate:
1. The gravitational potential energy of the object
2. If the object falls, what will be its kinetic energy just before hitting the ground?
3. What will be its velocity just before impact?

Use g = 9.8 m/s² and show your work."""

            # Fallback generic problem text
            return f"Solve this {template.concept} problem using the given values: {variables}"
            
        except Exception as e:
            logger.error(f"❌ Failed to generate problem text: {e}")
            return f"Physics problem involving {template.concept}"
    
    async def _generate_adaptive_hints(self, template: AdaptiveProblemTemplate, 
                                     learning_style: str, variables: Dict[str, float]) -> List[str]:
        """Generate hints adapted to student learning style"""
        try:
            base_hints = []
            
            if template.concept == 'kinematics_1d':
                base_hints = [
                    "Start by listing what you know and what you need to find",
                    "Choose the appropriate kinematic equation",
                    "Substitute the known values and solve for the unknown"
                ]
            elif template.concept == 'forces':
                base_hints = [
                    "Draw a free body diagram to identify all forces",
                    "Apply Newton's second law: F = ma",
                    "Remember to consider the direction of forces"
                ]
            elif template.concept == 'energy':
                base_hints = [
                    "Identify the types of energy involved",
                    "Apply conservation of energy principle",
                    "Remember: PE = mgh, KE = ½mv²"
                ]
            
            # Adapt hints to learning style
            if learning_style == 'visual':
                base_hints.insert(0, "Try drawing a diagram or sketch to visualize the problem")
            elif learning_style == 'analytical':
                base_hints.insert(0, "Write down all equations that might be relevant")
            elif learning_style == 'kinesthetic':
                base_hints.insert(0, "Imagine the physical scenario described in the problem")
            
            return base_hints
            
        except Exception as e:
            logger.error(f"❌ Failed to generate adaptive hints: {e}")
            return ["Think about the physics principles involved in this problem"]
    
    def _estimate_problem_time(self, template: AdaptiveProblemTemplate, difficulty: float) -> float:
        """Estimate time needed to solve problem"""
        try:
            base_times = {
                'kinematics_1d': 3,
                'forces': 5,
                'energy': 4,
                'momentum': 5,
                'angular_motion': 7
            }
            
            base_time = base_times.get(template.concept, 5)
            difficulty_multiplier = 1 + (difficulty - 0.5)
            
            return max(2, base_time * difficulty_multiplier)
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate problem time: {e}")
            return 5.0
    
    async def _parse_student_answer(self, answer: str, concept: str) -> Dict[str, Any]:
        """Parse student answer into structured format"""
        try:
            # Simple parsing - in production, this would be more sophisticated
            import re
            
            parsed = {
                'raw_answer': answer,
                'numerical_values': [],
                'units_mentioned': [],
                'equations_shown': []
            }
            
            # Extract numerical values
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            parsed['numerical_values'] = [float(n) for n in numbers]
            
            # Extract units
            units = re.findall(r'\b(m/s²?|kg|N|J|W|Hz)\b', answer)
            parsed['units_mentioned'] = units
            
            # Check for equations
            if '=' in answer:
                parsed['equations_shown'] = answer.count('=')
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Failed to parse student answer: {e}")
            return {'raw_answer': answer, 'numerical_values': [], 'units_mentioned': [], 'equations_shown': []}
    
    async def _validate_with_mcp(self, concept: str, variables: Dict[str, float],
                               student_answer: Dict[str, Any], expected: Dict[str, float]) -> Dict[str, Any]:
        """Validate student answer using MCP calculations"""
        try:
            student_values = student_answer['numerical_values']
            expected_values = list(expected.values())
            
            if not student_values or not expected_values:
                return {
                    'is_correct': False,
                    'score': 0.0,
                    'partial_credit': []
                }
            
            # Compare numerical values with tolerance
            tolerance = 0.05  # 5% tolerance
            correct_count = 0
            total_values = len(expected_values)
            
            for i, expected_val in enumerate(expected_values):
                if i < len(student_values):
                    student_val = student_values[i]
                    if abs(student_val - expected_val) / abs(expected_val) <= tolerance:
                        correct_count += 1
            
            score = correct_count / total_values if total_values > 0 else 0.0
            is_correct = score >= 0.8  # 80% threshold for correctness
            
            # Check for partial credit areas
            partial_credit = []
            if len(student_answer['units_mentioned']) > 0:
                partial_credit.append('units_included')
            if student_answer['equations_shown'] > 0:
                partial_credit.append('work_shown')
            
            return {
                'is_correct': is_correct,
                'score': score,
                'partial_credit': partial_credit,
                'next_step_hint': 'Check your calculations and units' if not is_correct else 'Great work!'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to validate with MCP: {e}")
            return {
                'is_correct': False,
                'score': 0.0,
                'partial_credit': []
            }
    
    async def _detect_misconceptions(self, concept: str, student_answer: Dict[str, Any],
                                   expected: Dict[str, float], known_misconceptions: List[str]) -> List[str]:
        """Detect common physics misconceptions in student answer"""
        try:
            detected = []
            
            # Check for unit misconceptions
            if len(student_answer['units_mentioned']) == 0:
                detected.append('missing_units')
            
            # Check for magnitude issues
            student_values = student_answer['numerical_values']
            expected_values = list(expected.values())
            
            if student_values and expected_values:
                for student_val, expected_val in zip(student_values, expected_values):
                    # Check for order of magnitude errors
                    if abs(student_val) > abs(expected_val) * 10:
                        detected.append('order_of_magnitude_error')
                    
                    # Check for sign errors
                    if (student_val > 0) != (expected_val > 0):
                        detected.append('sign_error')
            
            # Concept-specific misconception detection
            if concept == 'kinematics_1d':
                if 'velocity' in student_answer['raw_answer'].lower() and 'acceleration' in student_answer['raw_answer'].lower():
                    # Check for velocity/acceleration confusion
                    pass
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Failed to detect misconceptions: {e}")
            return []
    
    async def _generate_validation_feedback(self, validation_result: Dict[str, Any],
                                          misconceptions: List[str], concept: str) -> str:
        """Generate personalized feedback based on validation results"""
        try:
            if validation_result['is_correct']:
                feedback = "Excellent work! Your answer is correct."
                if 'units_included' in validation_result['partial_credit']:
                    feedback += " Great job including units."
                if 'work_shown' in validation_result['partial_credit']:
                    feedback += " I can see your work clearly."
            else:
                feedback = "Not quite right, but you're on the right track."
                
                if 'missing_units' in misconceptions:
                    feedback += " Remember to include units in your answer."
                
                if 'order_of_magnitude_error' in misconceptions:
                    feedback += " Check your calculation - the magnitude seems off."
                
                if 'sign_error' in misconceptions:
                    feedback += " Pay attention to the direction/sign of your answer."
                
                if validation_result['score'] > 0.5:
                    feedback += " You have the right approach!"
            
            return feedback
            
        except Exception as e:
            logger.error(f"❌ Failed to generate validation feedback: {e}")
            return "Keep working on it!"

# Example usage and testing
async def test_mcp_integration():
    """Test MCP integration functionality"""
    try:
        logger.info("🧪 Testing MCP Physics Tutoring Integration")
        
        integration = MCPPhysicsTutoringIntegration()
        
        # Test problem generation
        problem = await integration.generate_adaptive_physics_problem(
            concept='kinematics_1d',
            difficulty=0.5,
            student_knowledge={'kinematics_1d': 0.3},
            learning_style='visual'
        )
        
        if problem:
            logger.info(f"✅ Generated problem: {problem['problem_id']}")
            
            # Test answer validation
            validation = await integration.validate_student_answer(
                problem, "velocity = 15 m/s, displacement = 100 m"
            )
            
            logger.info(f"✅ Validation result: {validation['is_correct']}")
        
        logger.info("✅ MCP integration test completed")
        
    except Exception as e:
        logger.error(f"❌ MCP integration test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_integration())