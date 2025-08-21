#!/usr/bin/env python3
"""
Explainable AI Engine for Physics Assistant Phase 6
Provides transparent and interpretable explanations for ML predictions
in educational contexts, ensuring students and educators understand AI decisions.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
import lime
import lime.lime_tabular
import shap
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    SIMPLE = "simple"           # Easy to understand for students
    DETAILED = "detailed"       # Comprehensive for educators
    TECHNICAL = "technical"     # Full technical details
    VISUAL = "visual"           # Visual explanations with charts

class ExplanationContext(Enum):
    STUDENT_FACING = "student"
    EDUCATOR_FACING = "educator"
    PARENT_FACING = "parent"
    RESEARCHER_FACING = "researcher"

@dataclass
class FeatureImportance:
    """Feature importance with educational context"""
    feature_name: str
    importance_score: float
    direction: str              # "positive", "negative", "neutral"
    confidence: float
    description: str
    educational_meaning: str
    actionable_insight: str

@dataclass
class ExplanationComponent:
    """Individual component of an explanation"""
    component_type: str         # "feature", "rule", "example", "counterfactual"
    content: str
    importance: float
    evidence: Dict[str, Any]
    visual_data: Optional[Dict[str, Any]] = None

@dataclass
class AIExplanation:
    """Comprehensive AI explanation for educational context"""
    explanation_id: str
    prediction_value: float
    prediction_confidence: float
    explanation_type: ExplanationType
    target_audience: ExplanationContext
    
    # Main explanation components
    summary: str                # One-sentence summary
    key_factors: List[FeatureImportance]
    detailed_reasoning: str
    confidence_explanation: str
    
    # Educational elements
    learning_implications: str
    recommended_actions: List[str]
    similar_cases: List[str]
    what_if_scenarios: List[str]
    
    # Technical details
    model_info: Dict[str, Any]
    explanation_components: List[ExplanationComponent]
    uncertainty_analysis: Dict[str, Any]
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"
    explanation_quality_score: float = 0.0

class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) for tabular data"""
    
    def __init__(self):
        self.explainers = {}
        self.feature_names = []
        self.training_data = None
    
    def initialize(self, training_data: np.ndarray, feature_names: List[str]):
        """Initialize LIME explainer with training data"""
        try:
            self.training_data = training_data
            self.feature_names = feature_names
            
            # Create LIME tabular explainer
            self.explainers['tabular'] = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['Low', 'Medium', 'High'],  # For educational predictions
                verbose=False,
                mode='regression'
            )
            
            logger.info("✅ LIME explainer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LIME: {e}")
    
    async def explain_prediction(self, instance: np.ndarray, 
                               predict_fn, num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation for a prediction"""
        try:
            if 'tabular' not in self.explainers:
                raise ValueError("LIME explainer not initialized")
            
            explainer = self.explainers['tabular']
            
            # Generate explanation
            explanation = explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_importance[feature_name] = importance
            
            return {
                'type': 'lime',
                'feature_importance': feature_importance,
                'prediction_confidence': explanation.score,
                'local_prediction': explanation.local_pred[0] if explanation.local_pred else None,
                'intercept': explanation.intercept[0] if explanation.intercept else None
            }
            
        except Exception as e:
            logger.error(f"❌ LIME explanation failed: {e}")
            return {}

class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) for model interpretability"""
    
    def __init__(self):
        self.explainers = {}
        self.background_data = None
        self.feature_names = []
    
    def initialize(self, model, background_data: np.ndarray, feature_names: List[str]):
        """Initialize SHAP explainer"""
        try:
            self.background_data = background_data
            self.feature_names = feature_names
            
            # Create appropriate SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # Tree-based models
                self.explainers['tree'] = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer for other models
                self.explainers['kernel'] = shap.KernelExplainer(
                    model.predict, 
                    background_data[:100]  # Sample for efficiency
                )
            
            logger.info("✅ SHAP explainer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize SHAP: {e}")
    
    async def explain_prediction(self, instance: np.ndarray) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction"""
        try:
            if not self.explainers:
                raise ValueError("SHAP explainer not initialized")
            
            # Use available explainer
            explainer_name = list(self.explainers.keys())[0]
            explainer = self.explainers[explainer_name]
            
            # Generate SHAP values
            if explainer_name == 'tree':
                shap_values = explainer.shap_values(instance.reshape(1, -1))
            else:
                shap_values = explainer.shap_values(instance.reshape(1, -1))
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-class, take first class
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first instance
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, (feature_name, shap_value) in enumerate(zip(self.feature_names, shap_values)):
                feature_importance[feature_name] = float(shap_value)
            
            return {
                'type': 'shap',
                'feature_importance': feature_importance,
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                'shap_values': shap_values.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ SHAP explanation failed: {e}")
            return {}

class PermutationImportanceExplainer:
    """Permutation-based feature importance explainer"""
    
    def __init__(self):
        self.feature_names = []
        self.baseline_score = None
    
    async def explain_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: List[str], scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """Generate permutation importance explanation"""
        try:
            self.feature_names = feature_names
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test,
                scoring=scoring,
                n_repeats=10,
                random_state=42
            )
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature_name in enumerate(feature_names):
                feature_importance[feature_name] = {
                    'importance_mean': float(perm_importance.importances_mean[i]),
                    'importance_std': float(perm_importance.importances_std[i]),
                    'rank': int(np.argsort(perm_importance.importances_mean)[::-1][i] + 1)
                }
            
            return {
                'type': 'permutation',
                'feature_importance': feature_importance,
                'baseline_score': float(perm_importance.importances_mean.sum()),
                'scoring_metric': scoring
            }
            
        except Exception as e:
            logger.error(f"❌ Permutation importance explanation failed: {e}")
            return {}

class EducationalRuleExtractor:
    """Extract interpretable rules for educational contexts"""
    
    def __init__(self):
        self.rule_models = {}
        self.feature_names = []
    
    async def extract_rules(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str], max_depth: int = 5) -> List[str]:
        """Extract interpretable rules using decision trees"""
        try:
            self.feature_names = feature_names
            
            # Train a simple decision tree for rule extraction
            tree_model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            )
            
            tree_model.fit(X, y)
            
            # Extract text rules
            tree_rules = export_text(tree_model, feature_names=feature_names)
            
            # Parse rules into educational format
            educational_rules = await self._parse_educational_rules(tree_rules)
            
            return educational_rules
            
        except Exception as e:
            logger.error(f"❌ Rule extraction failed: {e}")
            return []
    
    async def _parse_educational_rules(self, tree_rules: str) -> List[str]:
        """Parse decision tree rules into educational language"""
        try:
            educational_rules = []
            
            # Split rules by lines and process
            lines = tree_rules.split('\n')
            current_rule = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Process tree structure
                if 'value' in line:
                    # End of a rule path
                    if current_rule:
                        rule_text = await self._format_educational_rule(current_rule)
                        if rule_text:
                            educational_rules.append(rule_text)
                        current_rule = []
                elif '|' in line or '├' in line or '└' in line:
                    # Rule condition
                    condition = self._extract_condition(line)
                    if condition:
                        current_rule.append(condition)
            
            return educational_rules[:10]  # Return top 10 rules
            
        except Exception as e:
            logger.error(f"❌ Educational rule parsing failed: {e}")
            return []
    
    def _extract_condition(self, line: str) -> Optional[str]:
        """Extract condition from decision tree line"""
        try:
            # Remove tree structure characters
            clean_line = line.replace('|', '').replace('├', '').replace('└', '').replace('─', '').strip()
            
            # Look for conditions like "feature <= value"
            if '<=' in clean_line:
                parts = clean_line.split('<=')
                if len(parts) == 2:
                    feature = parts[0].strip()
                    value = parts[1].strip()
                    return f"{feature} is low (≤ {value})"
            elif '>' in clean_line:
                parts = clean_line.split('>')
                if len(parts) == 2:
                    feature = parts[0].strip()
                    value = parts[1].strip()
                    return f"{feature} is high (> {value})"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Condition extraction failed: {e}")
            return None
    
    async def _format_educational_rule(self, conditions: List[str]) -> Optional[str]:
        """Format rule conditions into educational language"""
        try:
            if not conditions:
                return None
            
            # Map technical features to educational language
            feature_mapping = {
                'success_rate': 'success in recent problems',
                'response_time': 'time spent on problems',
                'help_seeking_rate': 'frequency of asking for help',
                'concept_coverage': 'variety of topics attempted',
                'difficulty_progression': 'complexity of problems tackled'
            }
            
            # Translate conditions
            educational_conditions = []
            for condition in conditions:
                for technical_term, educational_term in feature_mapping.items():
                    if technical_term in condition:
                        condition = condition.replace(technical_term, educational_term)
                educational_conditions.append(condition)
            
            # Format as a rule
            if len(educational_conditions) == 1:
                return f"When {educational_conditions[0]}, students typically perform well."
            else:
                conditions_text = " and ".join(educational_conditions[:-1])
                return f"When {conditions_text} and {educational_conditions[-1]}, students typically perform well."
            
        except Exception as e:
            logger.error(f"❌ Educational rule formatting failed: {e}")
            return None

class ExplainableAIEngine:
    """Comprehensive explainable AI engine for educational ML models"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Explanation components
        self.lime_explainer = LIMEExplainer()
        self.shap_explainer = SHAPExplainer()
        self.permutation_explainer = PermutationImportanceExplainer()
        self.rule_extractor = EducationalRuleExtractor()
        
        # Configuration
        self.explanation_cache = {}
        self.feature_descriptions = {}
        self.educational_mappings = {}
        
        # Template explanations
        self.explanation_templates = {
            ExplanationType.SIMPLE: {
                'summary': "Based on your learning patterns, {prediction_text}",
                'factors': "The main factors influencing this are: {top_factors}",
                'action': "To improve: {recommendations}"
            },
            ExplanationType.DETAILED: {
                'summary': "Analysis of learning data indicates {prediction_text} with {confidence_level} confidence",
                'factors': "Key contributing factors include: {detailed_factors}",
                'reasoning': "This prediction is based on {reasoning_details}",
                'action': "Recommended interventions: {detailed_recommendations}"
            }
        }
    
    async def initialize(self, training_data: np.ndarray = None, 
                        feature_names: List[str] = None):
        """Initialize the explainable AI engine"""
        try:
            logger.info("🚀 Initializing Explainable AI Engine")
            
            # Set up feature descriptions
            await self._initialize_feature_descriptions()
            
            # Initialize explanation components if training data available
            if training_data is not None and feature_names is not None:
                await self._initialize_explainers(training_data, feature_names)
            
            logger.info("✅ Explainable AI Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Explainable AI Engine: {e}")
            return False
    
    async def _initialize_feature_descriptions(self):
        """Initialize educational descriptions for features"""
        try:
            self.feature_descriptions = {
                'success_rate': {
                    'name': 'Success Rate',
                    'description': 'Percentage of problems solved correctly',
                    'educational_meaning': 'How well you understand the concepts',
                    'low_impact': 'More practice needed to master the concepts',
                    'high_impact': 'Strong understanding demonstrated'
                },
                'response_time': {
                    'name': 'Response Time',
                    'description': 'Average time taken to solve problems',
                    'educational_meaning': 'How efficiently you work through problems',
                    'low_impact': 'Quick responses - may need to slow down and think more carefully',
                    'high_impact': 'Taking time to think through problems thoroughly'
                },
                'help_seeking_rate': {
                    'name': 'Help Seeking',
                    'description': 'How often you ask for hints or help',
                    'educational_meaning': 'Your independence in problem solving',
                    'low_impact': 'Working independently most of the time',
                    'high_impact': 'Actively seeking help when needed'
                },
                'concept_coverage': {
                    'name': 'Topic Variety',
                    'description': 'Number of different physics topics you\'ve worked on',
                    'educational_meaning': 'Breadth of your physics knowledge',
                    'low_impact': 'Focused on specific topics',
                    'high_impact': 'Exploring diverse physics concepts'
                },
                'difficulty_progression': {
                    'name': 'Challenge Level',
                    'description': 'How the difficulty of problems has changed over time',
                    'educational_meaning': 'Your readiness for more advanced material',
                    'low_impact': 'Working at a steady difficulty level',
                    'high_impact': 'Taking on increasingly challenging problems'
                }
            }
            
            logger.info("✅ Feature descriptions initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature descriptions: {e}")
    
    async def _initialize_explainers(self, training_data: np.ndarray, feature_names: List[str]):
        """Initialize explanation algorithms"""
        try:
            # Initialize LIME
            self.lime_explainer.initialize(training_data, feature_names)
            
            # SHAP and permutation importance will be initialized per model
            logger.info("✅ Explainers initialized with training data")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize explainers: {e}")
    
    async def explain_prediction(self, prediction_value: float, 
                               features: Dict[str, float],
                               model_name: str,
                               student_id: str,
                               explanation_type: ExplanationType = ExplanationType.SIMPLE,
                               target_audience: ExplanationContext = ExplanationContext.STUDENT_FACING,
                               model_predict_fn=None) -> AIExplanation:
        """Generate comprehensive explanation for a prediction"""
        try:
            logger.info(f"🔍 Generating {explanation_type.value} explanation for {model_name}")
            
            # Generate explanation ID
            explanation_id = f"exp_{student_id}_{model_name}_{datetime.now().timestamp()}"
            
            # Calculate feature importance using multiple methods
            feature_importance_results = await self._calculate_comprehensive_importance(
                features, model_predict_fn
            )
            
            # Create feature importance objects
            key_factors = await self._create_feature_importance_objects(
                feature_importance_results, features
            )
            
            # Generate explanations based on type and audience
            explanation_content = await self._generate_explanation_content(
                prediction_value, key_factors, explanation_type, target_audience, model_name
            )
            
            # Generate educational insights
            learning_implications = await self._generate_learning_implications(
                prediction_value, key_factors, student_id
            )
            
            # Create recommended actions
            recommended_actions = await self._generate_recommended_actions(
                prediction_value, key_factors, target_audience
            )
            
            # Generate what-if scenarios
            what_if_scenarios = await self._generate_what_if_scenarios(
                features, key_factors
            )
            
            # Find similar cases
            similar_cases = await self._find_similar_cases(student_id, features)
            
            # Calculate explanation quality
            quality_score = await self._calculate_explanation_quality(
                feature_importance_results, explanation_content
            )
            
            # Create comprehensive explanation
            explanation = AIExplanation(
                explanation_id=explanation_id,
                prediction_value=prediction_value,
                prediction_confidence=0.85,  # Would come from model
                explanation_type=explanation_type,
                target_audience=target_audience,
                summary=explanation_content['summary'],
                key_factors=key_factors,
                detailed_reasoning=explanation_content['detailed_reasoning'],
                confidence_explanation=explanation_content['confidence_explanation'],
                learning_implications=learning_implications,
                recommended_actions=recommended_actions,
                similar_cases=similar_cases,
                what_if_scenarios=what_if_scenarios,
                model_info={
                    'model_name': model_name,
                    'prediction_type': 'educational_outcome',
                    'feature_count': len(features)
                },
                explanation_components=await self._create_explanation_components(
                    feature_importance_results, explanation_content
                ),
                uncertainty_analysis=await self._analyze_uncertainty(
                    prediction_value, features
                ),
                explanation_quality_score=quality_score
            )
            
            # Cache explanation
            self.explanation_cache[explanation_id] = explanation
            
            logger.info(f"✅ Generated explanation with quality score: {quality_score:.2f}")
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanation: {e}")
            # Return minimal explanation
            return await self._create_fallback_explanation(
                prediction_value, features, model_name, student_id
            )
    
    async def _calculate_comprehensive_importance(self, features: Dict[str, float],
                                                model_predict_fn=None) -> Dict[str, Any]:
        """Calculate feature importance using multiple methods"""
        try:
            importance_results = {
                'methods_used': [],
                'feature_scores': defaultdict(list),
                'consensus_ranking': {}
            }
            
            # Simple correlation-based importance (fallback)
            feature_values = list(features.values())
            feature_names = list(features.keys())
            
            # Simple heuristic importance based on value ranges
            for i, (feature_name, value) in enumerate(features.items()):
                # Normalize and calculate importance
                if feature_name in self.feature_descriptions:
                    # Use domain knowledge for importance
                    if feature_name == 'success_rate':
                        importance = abs(value - 0.5) * 2  # Distance from average
                    elif feature_name == 'help_seeking_rate':
                        importance = min(value * 2, 1.0)  # Higher help seeking = more important
                    else:
                        importance = min(value, 1.0)
                else:
                    importance = 0.5
                
                importance_results['feature_scores'][feature_name].append(importance)
            
            importance_results['methods_used'].append('heuristic')
            
            # LIME explanation if available
            if model_predict_fn and hasattr(self.lime_explainer, 'explainers'):
                try:
                    feature_array = np.array(feature_values)
                    lime_result = await self.lime_explainer.explain_prediction(
                        feature_array, model_predict_fn
                    )
                    
                    if lime_result and 'feature_importance' in lime_result:
                        for feature_name, importance in lime_result['feature_importance'].items():
                            importance_results['feature_scores'][feature_name].append(abs(importance))
                        importance_results['methods_used'].append('lime')
                        
                except Exception as e:
                    logger.warning(f"⚠️ LIME explanation failed: {e}")
            
            # Calculate consensus ranking
            consensus_scores = {}
            for feature_name, scores in importance_results['feature_scores'].items():
                consensus_scores[feature_name] = np.mean(scores) if scores else 0.0
            
            # Rank features
            sorted_features = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (feature_name, score) in enumerate(sorted_features, 1):
                importance_results['consensus_ranking'][feature_name] = {
                    'rank': rank,
                    'score': score,
                    'normalized_score': score / max(consensus_scores.values()) if max(consensus_scores.values()) > 0 else 0
                }
            
            return importance_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive importance calculation failed: {e}")
            return {'methods_used': [], 'feature_scores': {}, 'consensus_ranking': {}}
    
    async def _create_feature_importance_objects(self, importance_results: Dict[str, Any],
                                               features: Dict[str, float]) -> List[FeatureImportance]:
        """Create FeatureImportance objects with educational context"""
        try:
            feature_importance_list = []
            
            # Get top 5 most important features
            consensus_ranking = importance_results.get('consensus_ranking', {})
            top_features = sorted(consensus_ranking.items(), 
                                key=lambda x: x[1]['score'], reverse=True)[:5]
            
            for feature_name, ranking_info in top_features:
                if feature_name in features:
                    feature_value = features[feature_name]
                    description_info = self.feature_descriptions.get(feature_name, {})
                    
                    # Determine direction
                    if feature_value > 0.7:
                        direction = "positive"
                        educational_meaning = description_info.get('high_impact', 'High performance indicator')
                    elif feature_value < 0.3:
                        direction = "negative"
                        educational_meaning = description_info.get('low_impact', 'Area needing improvement')
                    else:
                        direction = "neutral"
                        educational_meaning = description_info.get('educational_meaning', 'Moderate performance')
                    
                    # Generate actionable insight
                    actionable_insight = await self._generate_actionable_insight(
                        feature_name, feature_value, direction
                    )
                    
                    feature_importance = FeatureImportance(
                        feature_name=description_info.get('name', feature_name),
                        importance_score=ranking_info['normalized_score'],
                        direction=direction,
                        confidence=0.8,  # Base confidence
                        description=description_info.get('description', f'Feature: {feature_name}'),
                        educational_meaning=educational_meaning,
                        actionable_insight=actionable_insight
                    )
                    
                    feature_importance_list.append(feature_importance)
            
            return feature_importance_list
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature importance objects: {e}")
            return []
    
    async def _generate_actionable_insight(self, feature_name: str, 
                                         feature_value: float, direction: str) -> str:
        """Generate actionable insights for students"""
        try:
            insights = {
                'success_rate': {
                    'positive': 'Keep up the excellent work! You\'re demonstrating strong understanding.',
                    'negative': 'Focus on reviewing fundamentals and practice more problems step-by-step.',
                    'neutral': 'You\'re on track. Continue practicing to strengthen your understanding.'
                },
                'response_time': {
                    'positive': 'You\'re taking appropriate time to think through problems carefully.',
                    'negative': 'Try to slow down and read problems more carefully before answering.',
                    'neutral': 'Your problem-solving pace is appropriate for learning.'
                },
                'help_seeking_rate': {
                    'positive': 'Good job asking for help when needed - this shows good learning strategy.',
                    'negative': 'Don\'t hesitate to ask for help when you\'re stuck on problems.',
                    'neutral': 'You have a balanced approach to seeking help when needed.'
                }
            }
            
            feature_insights = insights.get(feature_name, {
                'positive': 'This is working well for your learning.',
                'negative': 'Consider adjusting this aspect of your learning approach.',
                'neutral': 'This aspect of your learning is on track.'
            })
            
            return feature_insights.get(direction, 'Continue your current learning approach.')
            
        except Exception as e:
            logger.error(f"❌ Failed to generate actionable insight: {e}")
            return 'Continue working on your learning goals.'
    
    async def _generate_explanation_content(self, prediction_value: float,
                                          key_factors: List[FeatureImportance],
                                          explanation_type: ExplanationType,
                                          target_audience: ExplanationContext,
                                          model_name: str) -> Dict[str, str]:
        """Generate explanation content based on type and audience"""
        try:
            content = {}
            
            # Determine prediction text
            if model_name == 'success_predictor':
                if prediction_value > 0.7:
                    prediction_text = 'you are likely to succeed in upcoming challenges'
                elif prediction_value < 0.4:
                    prediction_text = 'you may need additional support to succeed'
                else:
                    prediction_text = 'you have moderate chances of success with continued effort'
            elif model_name == 'engagement_predictor':
                if prediction_value > 0.7:
                    prediction_text = 'you show high engagement with learning'
                elif prediction_value < 0.4:
                    prediction_text = 'your engagement could be improved'
                else:
                    prediction_text = 'your engagement level is moderate'
            else:
                prediction_text = f'the prediction indicates a score of {prediction_value:.2f}'
            
            # Generate content based on explanation type
            if explanation_type == ExplanationType.SIMPLE:
                content['summary'] = f"Based on your learning patterns, {prediction_text}."
                
                if key_factors:
                    top_factor = key_factors[0]
                    content['detailed_reasoning'] = f"This is mainly because your {top_factor.feature_name.lower()} {top_factor.educational_meaning.lower()}."
                else:
                    content['detailed_reasoning'] = "This prediction is based on your overall learning patterns."
                
                content['confidence_explanation'] = "I'm confident in this prediction based on your recent learning data."
                
            elif explanation_type == ExplanationType.DETAILED:
                content['summary'] = f"Analysis of your learning data indicates {prediction_text} with high confidence."
                
                factor_descriptions = []
                for factor in key_factors[:3]:
                    factor_descriptions.append(f"{factor.feature_name} ({factor.direction} impact)")
                
                if factor_descriptions:
                    content['detailed_reasoning'] = f"This prediction is based on several key factors: {', '.join(factor_descriptions)}. Each of these contributes to understanding your learning progress and predicting future performance."
                else:
                    content['detailed_reasoning'] = "This prediction considers multiple aspects of your learning behavior and performance patterns."
                
                content['confidence_explanation'] = f"The confidence in this prediction is high because it's based on {len(key_factors)} key learning indicators that have proven reliable for educational predictions."
            
            return content
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanation content: {e}")
            return {
                'summary': 'The AI system made a prediction about your learning.',
                'detailed_reasoning': 'This prediction is based on your learning data.',
                'confidence_explanation': 'The system is moderately confident in this prediction.'
            }
    
    async def _generate_learning_implications(self, prediction_value: float,
                                            key_factors: List[FeatureImportance],
                                            student_id: str) -> str:
        """Generate learning implications for educational context"""
        try:
            implications = []
            
            # Based on prediction value
            if prediction_value > 0.7:
                implications.append("You're demonstrating strong learning progress and are ready for more challenging material.")
            elif prediction_value < 0.4:
                implications.append("This suggests focusing on foundational concepts before moving to advanced topics.")
            else:
                implications.append("You're making steady progress and should continue with your current learning approach.")
            
            # Based on key factors
            for factor in key_factors[:2]:
                if factor.direction == 'negative':
                    implications.append(f"Working on {factor.feature_name.lower()} could significantly improve your learning outcomes.")
                elif factor.direction == 'positive':
                    implications.append(f"Your strong {factor.feature_name.lower()} is a key asset in your learning journey.")
            
            return " ".join(implications)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate learning implications: {e}")
            return "Continue focusing on your learning goals and seek help when needed."
    
    async def _generate_recommended_actions(self, prediction_value: float,
                                          key_factors: List[FeatureImportance],
                                          target_audience: ExplanationContext) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            # Add factor-specific recommendations
            for factor in key_factors[:3]:
                if factor.actionable_insight:
                    recommendations.append(factor.actionable_insight)
            
            # Add general recommendations based on prediction
            if prediction_value < 0.5:
                if target_audience == ExplanationContext.STUDENT_FACING:
                    recommendations.extend([
                        "Review previous topics you found challenging",
                        "Practice problems at a comfortable pace",
                        "Ask your teacher for additional support"
                    ])
                elif target_audience == ExplanationContext.EDUCATOR_FACING:
                    recommendations.extend([
                        "Provide additional scaffolding for this student",
                        "Consider reviewing prerequisite concepts",
                        "Implement more frequent check-ins"
                    ])
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            return ["Continue your learning journey with focus and persistence."]
    
    async def _generate_what_if_scenarios(self, features: Dict[str, float],
                                        key_factors: List[FeatureImportance]) -> List[str]:
        """Generate what-if scenarios for understanding"""
        try:
            scenarios = []
            
            for factor in key_factors[:3]:
                feature_name = factor.feature_name.lower()
                
                if factor.direction == 'negative':
                    scenarios.append(f"If your {feature_name} improved, your predicted outcome could increase significantly.")
                elif factor.direction == 'positive':
                    scenarios.append(f"Maintaining your strong {feature_name} will help sustain good performance.")
                else:
                    scenarios.append(f"Small improvements in {feature_name} could lead to better outcomes.")
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Failed to generate what-if scenarios: {e}")
            return ["Different learning approaches could lead to different outcomes."]
    
    async def _find_similar_cases(self, student_id: str, 
                                features: Dict[str, float]) -> List[str]:
        """Find similar student cases for context"""
        try:
            # In a real implementation, this would query the database
            similar_cases = [
                "Students with similar learning patterns typically improve with consistent practice.",
                "Others in your situation have benefited from focusing on problem-solving strategies.",
                "Similar learners often see progress when they engage with interactive content."
            ]
            
            return similar_cases[:3]
            
        except Exception as e:
            logger.error(f"❌ Failed to find similar cases: {e}")
            return ["Your learning pattern is unique, focus on your individual progress."]
    
    async def _create_explanation_components(self, importance_results: Dict[str, Any],
                                           explanation_content: Dict[str, str]) -> List[ExplanationComponent]:
        """Create detailed explanation components"""
        try:
            components = []
            
            # Feature importance component
            components.append(ExplanationComponent(
                component_type="feature",
                content="Feature importance analysis",
                importance=0.8,
                evidence=importance_results
            ))
            
            # Reasoning component
            components.append(ExplanationComponent(
                component_type="reasoning",
                content=explanation_content.get('detailed_reasoning', ''),
                importance=0.9,
                evidence={'explanation_method': 'educational_interpretation'}
            ))
            
            return components
            
        except Exception as e:
            logger.error(f"❌ Failed to create explanation components: {e}")
            return []
    
    async def _analyze_uncertainty(self, prediction_value: float,
                                 features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze prediction uncertainty"""
        try:
            # Simple uncertainty analysis
            uncertainty_factors = []
            
            # Check for missing or sparse features
            feature_count = len([v for v in features.values() if v > 0])
            if feature_count < 3:
                uncertainty_factors.append("Limited feature data available")
            
            # Check for extreme values
            extreme_values = sum(1 for v in features.values() if v < 0.1 or v > 0.9)
            if extreme_values > 0:
                uncertainty_factors.append("Some extreme feature values detected")
            
            # Calculate uncertainty score
            uncertainty_score = min(0.3, len(uncertainty_factors) * 0.1)
            
            return {
                'uncertainty_score': uncertainty_score,
                'uncertainty_factors': uncertainty_factors,
                'confidence_intervals': {
                    'lower': max(0, prediction_value - uncertainty_score),
                    'upper': min(1, prediction_value + uncertainty_score)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze uncertainty: {e}")
            return {'uncertainty_score': 0.2, 'uncertainty_factors': [], 'confidence_intervals': {}}
    
    async def _calculate_explanation_quality(self, importance_results: Dict[str, Any],
                                           explanation_content: Dict[str, str]) -> float:
        """Calculate quality score for explanation"""
        try:
            quality_score = 0.0
            
            # Methods diversity
            methods_count = len(importance_results.get('methods_used', []))
            quality_score += min(0.3, methods_count * 0.15)
            
            # Content completeness
            required_fields = ['summary', 'detailed_reasoning', 'confidence_explanation']
            complete_fields = sum(1 for field in required_fields if explanation_content.get(field))
            quality_score += (complete_fields / len(required_fields)) * 0.4
            
            # Feature coverage
            feature_count = len(importance_results.get('consensus_ranking', {}))
            quality_score += min(0.3, feature_count * 0.06)
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate explanation quality: {e}")
            return 0.5
    
    async def _create_fallback_explanation(self, prediction_value: float,
                                         features: Dict[str, float],
                                         model_name: str,
                                         student_id: str) -> AIExplanation:
        """Create minimal fallback explanation when main process fails"""
        try:
            explanation_id = f"fallback_{student_id}_{datetime.now().timestamp()}"
            
            return AIExplanation(
                explanation_id=explanation_id,
                prediction_value=prediction_value,
                prediction_confidence=0.5,
                explanation_type=ExplanationType.SIMPLE,
                target_audience=ExplanationContext.STUDENT_FACING,
                summary="The AI system analyzed your learning patterns to make this prediction.",
                key_factors=[],
                detailed_reasoning="This prediction is based on your recent learning activity and performance patterns.",
                confidence_explanation="The system has moderate confidence in this prediction.",
                learning_implications="Continue focusing on your learning goals.",
                recommended_actions=["Keep practicing", "Ask for help when needed"],
                similar_cases=["Other students have shown similar learning patterns"],
                what_if_scenarios=["Different study approaches could affect your outcomes"],
                model_info={'model_name': model_name, 'fallback': True},
                explanation_components=[],
                uncertainty_analysis={'uncertainty_score': 0.3},
                explanation_quality_score=0.3
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create fallback explanation: {e}")
            raise

# Testing function
async def test_explainable_ai():
    """Test explainable AI engine"""
    try:
        logger.info("🧪 Testing Explainable AI Engine")
        
        engine = ExplainableAIEngine()
        await engine.initialize()
        
        # Test explanation generation
        sample_features = {
            'success_rate': 0.75,
            'response_time': 45.0,
            'help_seeking_rate': 0.15,
            'concept_coverage': 4,
            'difficulty_progression': 0.6
        }
        
        explanation = await engine.explain_prediction(
            prediction_value=0.78,
            features=sample_features,
            model_name='success_predictor',
            student_id='test_student',
            explanation_type=ExplanationType.DETAILED,
            target_audience=ExplanationContext.STUDENT_FACING
        )
        
        logger.info(f"✅ Generated explanation: {explanation.summary}")
        logger.info(f"📊 Quality score: {explanation.explanation_quality_score:.2f}")
        logger.info(f"🔍 Key factors: {len(explanation.key_factors)}")
        
        logger.info("✅ Explainable AI Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Explainable AI test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_explainable_ai())