#!/usr/bin/env python3
"""
A/B Testing Framework for Educational Recommendations - Phase 6
Implements statistical A/B testing for ML model evaluation, recommendation system
optimization, and educational intervention effectiveness measurement.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
import uuid
import warnings
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
import redis

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    RECOMMENDATION_ALGORITHM = "recommendation_algorithm"
    ML_MODEL_COMPARISON = "ml_model_comparison"
    UI_INTERFACE = "ui_interface"
    LEARNING_INTERVENTION = "learning_intervention"
    CONTENT_DELIVERY = "content_delivery"
    DIFFICULTY_ADAPTATION = "difficulty_adaptation"

class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    CONTROL = "control"
    TREATMENT = "treatment"

class MetricType(Enum):
    CONTINUOUS = "continuous"          # e.g., time spent, score
    BINARY = "binary"                 # e.g., success/failure
    CATEGORICAL = "categorical"       # e.g., choice selection
    COUNT = "count"                  # e.g., number of attempts

class SignificanceLevel(Enum):
    ALPHA_001 = 0.01
    ALPHA_005 = 0.05
    ALPHA_010 = 0.10

@dataclass
class ExperimentVariant:
    """A/B test variant configuration"""
    variant_id: str
    variant_name: str
    variant_type: VariantType
    description: str
    configuration: Dict[str, Any]
    traffic_allocation: float  # Percentage of users (0.0 to 1.0)
    
    # Performance tracking
    user_count: int = 0
    conversion_count: int = 0
    total_interactions: int = 0

@dataclass
class ExperimentMetric:
    """Metric definition for A/B testing"""
    metric_id: str
    metric_name: str
    metric_type: MetricType
    description: str
    
    # Statistical parameters
    expected_effect_size: Optional[float] = None
    minimum_detectable_effect: Optional[float] = None
    baseline_value: Optional[float] = None
    
    # Data collection
    collection_method: str = "automatic"
    calculation_formula: Optional[str] = None

@dataclass
class ExperimentDesign:
    """A/B test experiment design"""
    experiment_id: str
    experiment_name: str
    experiment_type: ExperimentType
    description: str
    hypothesis: str
    
    # Variants and metrics
    variants: List[ExperimentVariant]
    primary_metric: ExperimentMetric
    secondary_metrics: List[ExperimentMetric]
    
    # Statistical parameters
    significance_level: SignificanceLevel
    statistical_power: float  # 1 - β (Type II error rate)
    minimum_sample_size: int
    expected_duration_days: int
    
    # Experiment control
    status: ExperimentStatus
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class UserAssignment:
    """User assignment to experiment variant"""
    assignment_id: str
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    
    # Context
    user_segment: Optional[str] = None
    assignment_method: str = "random"

@dataclass
class ExperimentResult:
    """A/B test statistical results"""
    experiment_id: str
    variant_comparisons: List[Dict[str, Any]]
    primary_metric_results: Dict[str, Any]
    secondary_metric_results: Dict[str, Any]
    
    # Statistical conclusions
    is_statistically_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    practical_significance: bool
    
    # Recommendations
    winning_variant: Optional[str] = None
    recommendation: str = ""
    risk_assessment: str = ""
    
    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    sample_size_achieved: Dict[str, int] = field(default_factory=dict)

class StatisticalAnalyzer:
    """Statistical analysis for A/B tests"""
    
    def __init__(self):
        self.confidence_levels = {
            SignificanceLevel.ALPHA_001: 0.99,
            SignificanceLevel.ALPHA_005: 0.95,
            SignificanceLevel.ALPHA_010: 0.90
        }
    
    async def analyze_experiment(self, experiment: ExperimentDesign,
                               experiment_data: pd.DataFrame) -> ExperimentResult:
        """Perform statistical analysis of A/B test experiment"""
        try:
            logger.info(f"📊 Analyzing experiment: {experiment.experiment_name}")
            
            # Validate data
            if experiment_data.empty:
                raise ValueError("No experiment data provided")
            
            # Analyze primary metric
            primary_results = await self._analyze_metric(
                experiment_data, experiment.primary_metric, experiment.variants,
                experiment.significance_level
            )
            
            # Analyze secondary metrics
            secondary_results = {}
            for metric in experiment.secondary_metrics:
                try:
                    result = await self._analyze_metric(
                        experiment_data, metric, experiment.variants,
                        experiment.significance_level
                    )
                    secondary_results[metric.metric_id] = result
                except Exception as e:
                    logger.warning(f"⚠️ Secondary metric analysis failed for {metric.metric_name}: {e}")
                    secondary_results[metric.metric_id] = {"error": str(e)}
            
            # Determine statistical significance
            is_significant = primary_results.get('p_value', 1.0) < experiment.significance_level.value
            
            # Calculate effect size and practical significance
            effect_size = primary_results.get('effect_size', 0.0)
            practical_significance = await self._assess_practical_significance(
                effect_size, experiment.primary_metric
            )
            
            # Generate variant comparisons
            variant_comparisons = await self._generate_variant_comparisons(
                experiment_data, experiment.variants, experiment.primary_metric
            )
            
            # Determine winning variant
            winning_variant = await self._determine_winning_variant(
                variant_comparisons, is_significant
            )
            
            # Generate recommendations
            recommendation = await self._generate_recommendation(
                is_significant, practical_significance, effect_size, winning_variant
            )
            
            # Risk assessment
            risk_assessment = await self._assess_risks(
                experiment, primary_results, secondary_results
            )
            
            # Sample sizes achieved
            sample_sizes = experiment_data.groupby('variant_id').size().to_dict()
            
            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                variant_comparisons=variant_comparisons,
                primary_metric_results=primary_results,
                secondary_metric_results=secondary_results,
                is_statistically_significant=is_significant,
                p_value=primary_results.get('p_value', 1.0),
                confidence_interval=primary_results.get('confidence_interval', (0.0, 0.0)),
                effect_size=effect_size,
                practical_significance=practical_significance,
                winning_variant=winning_variant,
                recommendation=recommendation,
                risk_assessment=risk_assessment,
                sample_size_achieved=sample_sizes
            )
            
            logger.info(f"✅ Analysis completed - Significant: {is_significant}, Effect: {effect_size:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Statistical analysis failed: {e}")
            raise
    
    async def _analyze_metric(self, data: pd.DataFrame, metric: ExperimentMetric,
                            variants: List[ExperimentVariant],
                            significance_level: SignificanceLevel) -> Dict[str, Any]:
        """Analyze a specific metric"""
        try:
            metric_column = f"metric_{metric.metric_id}"
            if metric_column not in data.columns:
                raise ValueError(f"Metric column {metric_column} not found in data")
            
            # Group data by variant
            variant_data = {}
            for variant in variants:
                variant_mask = data['variant_id'] == variant.variant_id
                variant_values = data[variant_mask][metric_column].dropna()
                variant_data[variant.variant_id] = variant_values
            
            if len(variant_data) < 2:
                raise ValueError("Need at least 2 variants for comparison")
            
            # Get control and treatment groups
            control_variant = next((v for v in variants if v.variant_type == VariantType.CONTROL), None)
            treatment_variants = [v for v in variants if v.variant_type == VariantType.TREATMENT]
            
            if not control_variant or not treatment_variants:
                raise ValueError("Need both control and treatment variants")
            
            control_data = variant_data[control_variant.variant_id]
            
            # Perform statistical test based on metric type
            if metric.metric_type == MetricType.CONTINUOUS:
                results = await self._test_continuous_metric(
                    control_data, variant_data, treatment_variants, significance_level
                )
            elif metric.metric_type == MetricType.BINARY:
                results = await self._test_binary_metric(
                    control_data, variant_data, treatment_variants, significance_level
                )
            elif metric.metric_type == MetricType.COUNT:
                results = await self._test_count_metric(
                    control_data, variant_data, treatment_variants, significance_level
                )
            else:
                raise ValueError(f"Unsupported metric type: {metric.metric_type}")
            
            # Add descriptive statistics
            results['descriptive_stats'] = {}
            for variant_id, values in variant_data.items():
                if len(values) > 0:
                    results['descriptive_stats'][variant_id] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()) if len(values) > 1 else 0.0,
                        'median': float(values.median()),
                        'count': len(values),
                        'min': float(values.min()),
                        'max': float(values.max())
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Metric analysis failed: {e}")
            return {'error': str(e)}
    
    async def _test_continuous_metric(self, control_data: pd.Series,
                                    variant_data: Dict[str, pd.Series],
                                    treatment_variants: List[ExperimentVariant],
                                    significance_level: SignificanceLevel) -> Dict[str, Any]:
        """Test continuous metrics using t-test"""
        try:
            results = {}
            
            # For now, compare control vs primary treatment
            primary_treatment = treatment_variants[0]
            treatment_data = variant_data[primary_treatment.variant_id]
            
            if len(control_data) == 0 or len(treatment_data) == 0:
                raise ValueError("Insufficient data for comparison")
            
            # Check normality (simplified)
            control_normal = len(control_data) > 30 or self._check_normality(control_data)
            treatment_normal = len(treatment_data) > 30 or self._check_normality(treatment_data)
            
            if control_normal and treatment_normal:
                # Use t-test
                statistic, p_value = ttest_ind(control_data, treatment_data, equal_var=False)
                test_used = "welch_t_test"
            else:
                # Use Mann-Whitney U test (non-parametric)
                statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
                test_used = "mann_whitney_u"
            
            # Calculate effect size (Cohen's d for continuous metrics)
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()
            pooled_std = np.sqrt(((len(control_data) - 1) * control_data.var() + 
                                 (len(treatment_data) - 1) * treatment_data.var()) / 
                                (len(control_data) + len(treatment_data) - 2))
            
            effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate confidence interval for mean difference
            mean_diff = treatment_mean - control_mean
            sem_diff = np.sqrt(control_data.var() / len(control_data) + 
                              treatment_data.var() / len(treatment_data))
            
            confidence_level = self.confidence_levels[significance_level]
            t_critical = stats.t.ppf((1 + confidence_level) / 2, 
                                   len(control_data) + len(treatment_data) - 2)
            margin_error = t_critical * sem_diff
            
            ci_lower = mean_diff - margin_error
            ci_upper = mean_diff + margin_error
            
            results.update({
                'test_statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'mean_difference': float(mean_diff),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'test_used': test_used,
                'control_mean': float(control_mean),
                'treatment_mean': float(treatment_mean)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Continuous metric test failed: {e}")
            return {'error': str(e)}
    
    async def _test_binary_metric(self, control_data: pd.Series,
                                variant_data: Dict[str, pd.Series],
                                treatment_variants: List[ExperimentVariant],
                                significance_level: SignificanceLevel) -> Dict[str, Any]:
        """Test binary metrics using chi-square or Fisher's exact test"""
        try:
            results = {}
            
            # For binary metrics, calculate success rates
            primary_treatment = treatment_variants[0]
            treatment_data = variant_data[primary_treatment.variant_id]
            
            # Create contingency table
            control_successes = control_data.sum()
            control_total = len(control_data)
            control_failures = control_total - control_successes
            
            treatment_successes = treatment_data.sum()
            treatment_total = len(treatment_data)
            treatment_failures = treatment_total - treatment_successes
            
            contingency_table = np.array([
                [control_successes, control_failures],
                [treatment_successes, treatment_failures]
            ])
            
            # Choose appropriate test
            min_expected = np.min(contingency_table)
            if min_expected < 5:
                # Use Fisher's exact test for small samples
                from scipy.stats import fisher_exact
                odds_ratio, p_value = fisher_exact(contingency_table)
                test_used = "fisher_exact"
                statistic = odds_ratio
            else:
                # Use chi-square test
                statistic, p_value, dof, expected = chi2_contingency(contingency_table)
                test_used = "chi_square"
            
            # Calculate conversion rates
            control_rate = control_successes / control_total if control_total > 0 else 0
            treatment_rate = treatment_successes / treatment_total if treatment_total > 0 else 0
            
            # Calculate effect size (relative lift)
            relative_lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0
            absolute_lift = treatment_rate - control_rate
            
            # Calculate confidence interval for difference in proportions
            p1, p2 = control_rate, treatment_rate
            n1, n2 = control_total, treatment_total
            
            if n1 > 0 and n2 > 0 and 0 < p1 < 1 and 0 < p2 < 1:
                se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                confidence_level = self.confidence_levels[significance_level]
                z_critical = stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_critical * se_diff
                
                ci_lower = absolute_lift - margin_error
                ci_upper = absolute_lift + margin_error
            else:
                ci_lower, ci_upper = 0.0, 0.0
            
            results.update({
                'test_statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(relative_lift),
                'absolute_lift': float(absolute_lift),
                'relative_lift': float(relative_lift),
                'confidence_interval': (float(ci_lower), float(ci_upper)),
                'test_used': test_used,
                'control_rate': float(control_rate),
                'treatment_rate': float(treatment_rate),
                'contingency_table': contingency_table.tolist()
            })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Binary metric test failed: {e}")
            return {'error': str(e)}
    
    async def _test_count_metric(self, control_data: pd.Series,
                               variant_data: Dict[str, pd.Series],
                               treatment_variants: List[ExperimentVariant],
                               significance_level: SignificanceLevel) -> Dict[str, Any]:
        """Test count metrics using Poisson or negative binomial tests"""
        try:
            # For count data, use Mann-Whitney U test as robust option
            primary_treatment = treatment_variants[0]
            treatment_data = variant_data[primary_treatment.variant_id]
            
            statistic, p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
            
            # Calculate means for effect size
            control_mean = control_data.mean()
            treatment_mean = treatment_data.mean()
            
            # Effect size as percentage change
            effect_size = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0
            
            results = {
                'test_statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'test_used': 'mann_whitney_u',
                'control_mean': float(control_mean),
                'treatment_mean': float(treatment_mean)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Count metric test failed: {e}")
            return {'error': str(e)}
    
    def _check_normality(self, data: pd.Series) -> bool:
        """Check if data is approximately normal"""
        try:
            if len(data) < 8:
                return False  # Too small for reliable test
            
            # Use Shapiro-Wilk test
            statistic, p_value = stats.shapiro(data)
            return p_value > 0.05  # Not significantly non-normal
            
        except Exception:
            return False
    
    async def _assess_practical_significance(self, effect_size: float,
                                           metric: ExperimentMetric) -> bool:
        """Assess if effect size is practically significant"""
        try:
            # Use Cohen's conventions adapted for educational context
            if metric.metric_type == MetricType.CONTINUOUS:
                return abs(effect_size) >= 0.2  # Small to medium effect
            elif metric.metric_type == MetricType.BINARY:
                return abs(effect_size) >= 0.05  # 5% relative improvement
            else:
                return abs(effect_size) >= 0.1   # 10% improvement for other metrics
            
        except Exception as e:
            logger.error(f"❌ Practical significance assessment failed: {e}")
            return False
    
    async def _generate_variant_comparisons(self, data: pd.DataFrame,
                                          variants: List[ExperimentVariant],
                                          primary_metric: ExperimentMetric) -> List[Dict[str, Any]]:
        """Generate pairwise variant comparisons"""
        try:
            comparisons = []
            metric_column = f"metric_{primary_metric.metric_id}"
            
            for variant in variants:
                variant_data = data[data['variant_id'] == variant.variant_id][metric_column]
                
                comparison = {
                    'variant_id': variant.variant_id,
                    'variant_name': variant.variant_name,
                    'variant_type': variant.variant_type.value,
                    'sample_size': len(variant_data),
                    'metric_mean': float(variant_data.mean()) if len(variant_data) > 0 else 0.0,
                    'metric_std': float(variant_data.std()) if len(variant_data) > 1 else 0.0,
                    'confidence_90': self._calculate_confidence_interval(variant_data, 0.90),
                    'confidence_95': self._calculate_confidence_interval(variant_data, 0.95)
                }
                
                comparisons.append(comparison)
            
            return comparisons
            
        except Exception as e:
            logger.error(f"❌ Variant comparison generation failed: {e}")
            return []
    
    def _calculate_confidence_interval(self, data: pd.Series, confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for sample mean"""
        try:
            if len(data) == 0:
                return (0.0, 0.0)
            
            mean = data.mean()
            sem = data.sem()  # Standard error of mean
            
            if len(data) > 30:
                # Use normal distribution for large samples
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                margin_error = z_score * sem
            else:
                # Use t-distribution for small samples
                t_score = stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
                margin_error = t_score * sem
            
            return (float(mean - margin_error), float(mean + margin_error))
            
        except Exception as e:
            logger.error(f"❌ Confidence interval calculation failed: {e}")
            return (0.0, 0.0)
    
    async def _determine_winning_variant(self, comparisons: List[Dict[str, Any]],
                                       is_significant: bool) -> Optional[str]:
        """Determine which variant performed best"""
        try:
            if not is_significant or not comparisons:
                return None
            
            # Find variant with highest mean performance
            best_variant = max(comparisons, key=lambda x: x['metric_mean'])
            return best_variant['variant_id']
            
        except Exception as e:
            logger.error(f"❌ Winning variant determination failed: {e}")
            return None
    
    async def _generate_recommendation(self, is_significant: bool,
                                     practical_significance: bool,
                                     effect_size: float,
                                     winning_variant: Optional[str]) -> str:
        """Generate recommendation based on results"""
        try:
            if not is_significant:
                return "Results are not statistically significant. Consider running the experiment longer or investigating external factors."
            
            if not practical_significance:
                return "While statistically significant, the effect size is too small to be practically meaningful. Consider testing more substantial changes."
            
            if winning_variant:
                return f"Implement variant {winning_variant} - it shows both statistical and practical significance with effect size of {effect_size:.3f}."
            else:
                return "Results are significant but no clear winner emerged. Analyze individual metrics for insights."
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return "Unable to generate recommendation due to analysis error."
    
    async def _assess_risks(self, experiment: ExperimentDesign,
                          primary_results: Dict[str, Any],
                          secondary_results: Dict[str, Any]) -> str:
        """Assess risks of implementing the winning variant"""
        try:
            risks = []
            
            # Check sample size adequacy
            achieved_samples = sum(experiment.variants[0].user_count for _ in experiment.variants)
            if achieved_samples < experiment.minimum_sample_size:
                risks.append("Sample size smaller than planned - results may be less reliable")
            
            # Check effect size consistency across metrics
            primary_effect = primary_results.get('effect_size', 0)
            secondary_effects = [r.get('effect_size', 0) for r in secondary_results.values() 
                               if isinstance(r, dict) and 'effect_size' in r]
            
            if secondary_effects and any(abs(e - primary_effect) > 0.5 for e in secondary_effects):
                risks.append("Inconsistent effects across metrics - monitor for unintended consequences")
            
            # Check for multiple testing
            if len(secondary_results) > 3:
                risks.append("Multiple testing performed - consider Bonferroni correction")
            
            if not risks:
                return "Low risk - results appear robust and consistent"
            else:
                return "; ".join(risks)
            
        except Exception as e:
            logger.error(f"❌ Risk assessment failed: {e}")
            return "Unable to assess risks"

class ExperimentManager:
    """Manage A/B testing experiments"""
    
    def __init__(self, db_manager=None, redis_client=None):
        self.db_manager = db_manager
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Core components
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Experiment storage
        self.active_experiments = {}
        self.user_assignments = {}
        self.experiment_data = defaultdict(list)
        
        # Assignment methods
        self.assignment_methods = {
            'random': self._random_assignment,
            'hash_based': self._hash_based_assignment,
            'stratified': self._stratified_assignment
        }
    
    async def initialize(self):
        """Initialize the A/B testing framework"""
        try:
            logger.info("🚀 Initializing A/B Testing Framework")
            
            # Load active experiments
            await self._load_active_experiments()
            
            # Initialize assignment tracking
            await self._initialize_assignment_tracking()
            
            logger.info("✅ A/B Testing Framework initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize A/B Testing Framework: {e}")
            return False
    
    async def create_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentDesign:
        """Create a new A/B testing experiment"""
        try:
            logger.info(f"🧪 Creating experiment: {experiment_config.get('name', 'Unnamed')}")
            
            # Validate configuration
            await self._validate_experiment_config(experiment_config)
            
            # Generate experiment ID
            experiment_id = str(uuid.uuid4())
            
            # Create variants
            variants = []
            for variant_config in experiment_config['variants']:
                variant = ExperimentVariant(
                    variant_id=str(uuid.uuid4()),
                    variant_name=variant_config['name'],
                    variant_type=VariantType(variant_config['type']),
                    description=variant_config['description'],
                    configuration=variant_config['configuration'],
                    traffic_allocation=variant_config['traffic_allocation']
                )
                variants.append(variant)
            
            # Create metrics
            primary_metric = ExperimentMetric(
                metric_id=str(uuid.uuid4()),
                metric_name=experiment_config['primary_metric']['name'],
                metric_type=MetricType(experiment_config['primary_metric']['type']),
                description=experiment_config['primary_metric']['description'],
                expected_effect_size=experiment_config['primary_metric'].get('expected_effect_size'),
                minimum_detectable_effect=experiment_config['primary_metric'].get('minimum_detectable_effect'),
                baseline_value=experiment_config['primary_metric'].get('baseline_value')
            )
            
            secondary_metrics = []
            for metric_config in experiment_config.get('secondary_metrics', []):
                metric = ExperimentMetric(
                    metric_id=str(uuid.uuid4()),
                    metric_name=metric_config['name'],
                    metric_type=MetricType(metric_config['type']),
                    description=metric_config['description']
                )
                secondary_metrics.append(metric)
            
            # Calculate sample size
            sample_size = await self._calculate_sample_size(
                primary_metric, 
                experiment_config.get('significance_level', 0.05),
                experiment_config.get('statistical_power', 0.8)
            )
            
            # Create experiment design
            experiment = ExperimentDesign(
                experiment_id=experiment_id,
                experiment_name=experiment_config['name'],
                experiment_type=ExperimentType(experiment_config['type']),
                description=experiment_config['description'],
                hypothesis=experiment_config['hypothesis'],
                variants=variants,
                primary_metric=primary_metric,
                secondary_metrics=secondary_metrics,
                significance_level=SignificanceLevel(experiment_config.get('significance_level', 0.05)),
                statistical_power=experiment_config.get('statistical_power', 0.8),
                minimum_sample_size=sample_size,
                expected_duration_days=experiment_config.get('expected_duration_days', 14),
                status=ExperimentStatus.DRAFT,
                created_by=experiment_config.get('created_by', 'system')
            )
            
            # Store experiment
            await self._store_experiment(experiment)
            
            logger.info(f"✅ Experiment created with ID: {experiment_id}")
            return experiment
            
        except Exception as e:
            logger.error(f"❌ Failed to create experiment: {e}")
            raise
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B testing experiment"""
        try:
            if experiment_id not in self.active_experiments:
                # Load experiment if not in memory
                await self._load_experiment(experiment_id)
            
            experiment = self.active_experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.DRAFT:
                raise ValueError(f"Experiment must be in DRAFT status to start, current: {experiment.status}")
            
            # Validate experiment is ready
            await self._validate_experiment_ready(experiment)
            
            # Start experiment
            experiment.status = ExperimentStatus.ACTIVE
            experiment.start_date = datetime.now()
            
            # Store updated experiment
            await self._store_experiment(experiment)
            
            logger.info(f"🚀 Started experiment: {experiment.experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start experiment {experiment_id}: {e}")
            return False
    
    async def assign_user_to_experiment(self, user_id: str, experiment_id: str,
                                      user_context: Dict[str, Any] = None) -> Optional[str]:
        """Assign user to experiment variant"""
        try:
            if experiment_id not in self.active_experiments:
                await self._load_experiment(experiment_id)
            
            experiment = self.active_experiments[experiment_id]
            
            if experiment.status != ExperimentStatus.ACTIVE:
                return None
            
            # Check if user already assigned
            assignment_key = f"{user_id}:{experiment_id}"
            if assignment_key in self.user_assignments:
                return self.user_assignments[assignment_key].variant_id
            
            # Assign user to variant
            variant_id = await self._assign_user_to_variant(user_id, experiment, user_context)
            
            if variant_id:
                # Create assignment record
                assignment = UserAssignment(
                    assignment_id=str(uuid.uuid4()),
                    user_id=user_id,
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    assigned_at=datetime.now(),
                    user_segment=user_context.get('segment') if user_context else None
                )
                
                self.user_assignments[assignment_key] = assignment
                
                # Update variant user count
                for variant in experiment.variants:
                    if variant.variant_id == variant_id:
                        variant.user_count += 1
                        break
                
                # Store assignment
                await self._store_user_assignment(assignment)
                
                logger.info(f"👤 Assigned user {user_id} to variant {variant_id}")
            
            return variant_id
            
        except Exception as e:
            logger.error(f"❌ Failed to assign user to experiment: {e}")
            return None
    
    async def record_metric_event(self, user_id: str, experiment_id: str,
                                metric_id: str, metric_value: Any,
                                context: Dict[str, Any] = None) -> bool:
        """Record a metric event for analysis"""
        try:
            # Get user's variant assignment
            assignment_key = f"{user_id}:{experiment_id}"
            if assignment_key not in self.user_assignments:
                logger.warning(f"⚠️ No assignment found for user {user_id} in experiment {experiment_id}")
                return False
            
            assignment = self.user_assignments[assignment_key]
            
            # Create metric event record
            event_record = {
                'user_id': user_id,
                'experiment_id': experiment_id,
                'variant_id': assignment.variant_id,
                'metric_id': metric_id,
                'metric_value': metric_value,
                'recorded_at': datetime.now(),
                'context': context or {}
            }
            
            # Store event
            self.experiment_data[experiment_id].append(event_record)
            
            # Update variant metrics if this is a conversion event
            if metric_id.endswith('_conversion') and metric_value == 1:
                experiment = self.active_experiments.get(experiment_id)
                if experiment:
                    for variant in experiment.variants:
                        if variant.variant_id == assignment.variant_id:
                            variant.conversion_count += 1
                            break
            
            # Store in persistent storage
            await self._store_metric_event(event_record)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to record metric event: {e}")
            return False
    
    async def analyze_experiment_results(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results"""
        try:
            logger.info(f"📊 Analyzing results for experiment {experiment_id}")
            
            # Load experiment
            if experiment_id not in self.active_experiments:
                await self._load_experiment(experiment_id)
            
            experiment = self.active_experiments[experiment_id]
            
            # Prepare data for analysis
            experiment_data = await self._prepare_analysis_data(experiment_id)
            
            if experiment_data.empty:
                raise ValueError("No data available for analysis")
            
            # Perform statistical analysis
            results = await self.statistical_analyzer.analyze_experiment(experiment, experiment_data)
            
            # Store results
            await self._store_analysis_results(results)
            
            logger.info(f"✅ Analysis completed for {experiment.experiment_name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze experiment results: {e}")
            raise
    
    async def _validate_experiment_config(self, config: Dict[str, Any]):
        """Validate experiment configuration"""
        required_fields = ['name', 'type', 'description', 'hypothesis', 'variants', 'primary_metric']
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate variants
        if len(config['variants']) < 2:
            raise ValueError("Experiment must have at least 2 variants")
        
        # Validate traffic allocation sums to 1.0
        total_allocation = sum(v['traffic_allocation'] for v in config['variants'])
        if abs(total_allocation - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Ensure there's a control variant
        control_variants = [v for v in config['variants'] if v['type'] == 'control']
        if not control_variants:
            raise ValueError("Experiment must have at least one control variant")
    
    async def _calculate_sample_size(self, metric: ExperimentMetric,
                                   alpha: float, power: float) -> int:
        """Calculate required sample size"""
        try:
            # Simplified sample size calculation
            # In practice, this would use power analysis formulas
            
            if metric.metric_type == MetricType.BINARY:
                # For binary metrics, use formula for comparing proportions
                baseline_rate = metric.baseline_value or 0.1
                effect_size = metric.minimum_detectable_effect or 0.05
                
                # Simplified calculation
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(power)
                
                p1 = baseline_rate
                p2 = baseline_rate + effect_size
                p_avg = (p1 + p2) / 2
                
                n = 2 * ((z_alpha + z_beta) ** 2) * p_avg * (1 - p_avg) / ((p2 - p1) ** 2)
                return int(np.ceil(n))
            
            elif metric.metric_type == MetricType.CONTINUOUS:
                # For continuous metrics
                effect_size = metric.minimum_detectable_effect or 0.2
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(power)
                
                n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
                return int(np.ceil(n))
            
            else:
                # Default sample size
                return 1000
                
        except Exception as e:
            logger.error(f"❌ Sample size calculation failed: {e}")
            return 1000  # Conservative default
    
    async def _assign_user_to_variant(self, user_id: str, experiment: ExperimentDesign,
                                    user_context: Dict[str, Any] = None) -> Optional[str]:
        """Assign user to a variant"""
        try:
            # Use hash-based assignment for consistency
            assignment_hash = hashlib.md5(f"{user_id}:{experiment.experiment_id}".encode()).hexdigest()
            assignment_value = int(assignment_hash[:8], 16) / (16**8)  # Convert to 0-1 range
            
            # Determine variant based on traffic allocation
            cumulative_allocation = 0.0
            for variant in experiment.variants:
                cumulative_allocation += variant.traffic_allocation
                if assignment_value <= cumulative_allocation:
                    return variant.variant_id
            
            # Fallback to last variant
            return experiment.variants[-1].variant_id
            
        except Exception as e:
            logger.error(f"❌ User assignment failed: {e}")
            return None
    
    async def _prepare_analysis_data(self, experiment_id: str) -> pd.DataFrame:
        """Prepare data for statistical analysis"""
        try:
            events = self.experiment_data.get(experiment_id, [])
            if not events:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Pivot metrics to columns
            experiment = self.active_experiments[experiment_id]
            all_metrics = [experiment.primary_metric] + experiment.secondary_metrics
            
            analysis_data = []
            
            # Group by user and variant
            for (user_id, variant_id), user_events in df.groupby(['user_id', 'variant_id']):
                user_data = {
                    'user_id': user_id,
                    'variant_id': variant_id
                }
                
                # Extract metric values
                for metric in all_metrics:
                    metric_events = user_events[user_events['metric_id'] == metric.metric_id]
                    if not metric_events.empty:
                        if metric.metric_type == MetricType.BINARY:
                            user_data[f'metric_{metric.metric_id}'] = metric_events['metric_value'].max()
                        elif metric.metric_type == MetricType.COUNT:
                            user_data[f'metric_{metric.metric_id}'] = metric_events['metric_value'].sum()
                        else:
                            user_data[f'metric_{metric.metric_id}'] = metric_events['metric_value'].mean()
                    else:
                        user_data[f'metric_{metric.metric_id}'] = 0
                
                analysis_data.append(user_data)
            
            return pd.DataFrame(analysis_data)
            
        except Exception as e:
            logger.error(f"❌ Data preparation failed: {e}")
            return pd.DataFrame()
    
    # Storage methods (simplified - would integrate with actual database)
    async def _store_experiment(self, experiment: ExperimentDesign):
        """Store experiment in database"""
        # In real implementation, would store in PostgreSQL
        self.active_experiments[experiment.experiment_id] = experiment
    
    async def _load_experiment(self, experiment_id: str):
        """Load experiment from database"""
        # In real implementation, would load from PostgreSQL
        pass
    
    async def _store_user_assignment(self, assignment: UserAssignment):
        """Store user assignment"""
        # In real implementation, would store in PostgreSQL
        pass
    
    async def _store_metric_event(self, event: Dict[str, Any]):
        """Store metric event"""
        # In real implementation, would store in PostgreSQL
        pass
    
    async def _store_analysis_results(self, results: ExperimentResult):
        """Store analysis results"""
        # In real implementation, would store in PostgreSQL
        pass
    
    async def _load_active_experiments(self):
        """Load active experiments from database"""
        # In real implementation, would load from PostgreSQL
        pass
    
    async def _initialize_assignment_tracking(self):
        """Initialize assignment tracking"""
        # In real implementation, would load from PostgreSQL
        pass
    
    async def _validate_experiment_ready(self, experiment: ExperimentDesign):
        """Validate experiment is ready to start"""
        # Additional validation logic
        pass

# Testing function
async def test_ab_testing_framework():
    """Test A/B testing framework"""
    try:
        logger.info("🧪 Testing A/B Testing Framework")
        
        manager = ExperimentManager()
        await manager.initialize()
        
        # Create test experiment
        experiment_config = {
            'name': 'Recommendation Algorithm Test',
            'type': 'recommendation_algorithm',
            'description': 'Test new vs old recommendation algorithm',
            'hypothesis': 'New algorithm will improve student engagement',
            'variants': [
                {
                    'name': 'Control - Old Algorithm',
                    'type': 'control',
                    'description': 'Current recommendation algorithm',
                    'configuration': {'algorithm': 'collaborative_filtering'},
                    'traffic_allocation': 0.5
                },
                {
                    'name': 'Treatment - New Algorithm',
                    'type': 'treatment',
                    'description': 'New hybrid recommendation algorithm',
                    'configuration': {'algorithm': 'hybrid_deep_learning'},
                    'traffic_allocation': 0.5
                }
            ],
            'primary_metric': {
                'name': 'Engagement Score',
                'type': 'continuous',
                'description': 'Average time spent per session',
                'baseline_value': 15.0,
                'minimum_detectable_effect': 2.0
            },
            'secondary_metrics': [
                {
                    'name': 'Click-through Rate',
                    'type': 'binary',
                    'description': 'Whether user clicked on recommendation'
                }
            ],
            'significance_level': 0.05,
            'statistical_power': 0.8,
            'expected_duration_days': 14
        }
        
        experiment = await manager.create_experiment(experiment_config)
        logger.info(f"✅ Created experiment: {experiment.experiment_name}")
        
        # Start experiment
        success = await manager.start_experiment(experiment.experiment_id)
        logger.info(f"✅ Started experiment: {success}")
        
        # Simulate user assignments and metric events
        for i in range(100):
            user_id = f"test_user_{i}"
            
            # Assign user
            variant_id = await manager.assign_user_to_experiment(user_id, experiment.experiment_id)
            
            if variant_id:
                # Simulate engagement metric
                if 'old' in variant_id.lower():
                    engagement_score = np.random.normal(15.0, 3.0)  # Control
                else:
                    engagement_score = np.random.normal(17.0, 3.0)  # Treatment (better)
                
                await manager.record_metric_event(
                    user_id, experiment.experiment_id, 
                    experiment.primary_metric.metric_id,
                    max(0, engagement_score)
                )
                
                # Simulate binary metric
                click_rate = 0.3 if 'old' in variant_id.lower() else 0.35
                clicked = 1 if np.random.random() < click_rate else 0
                
                if experiment.secondary_metrics:
                    await manager.record_metric_event(
                        user_id, experiment.experiment_id,
                        experiment.secondary_metrics[0].metric_id,
                        clicked
                    )
        
        # Analyze results
        results = await manager.analyze_experiment_results(experiment.experiment_id)
        
        logger.info(f"📊 Analysis Results:")
        logger.info(f"  - Statistically Significant: {results.is_statistically_significant}")
        logger.info(f"  - P-value: {results.p_value:.4f}")
        logger.info(f"  - Effect Size: {results.effect_size:.4f}")
        logger.info(f"  - Winning Variant: {results.winning_variant}")
        logger.info(f"  - Recommendation: {results.recommendation}")
        
        logger.info("✅ A/B Testing Framework test completed")
        
    except Exception as e:
        logger.error(f"❌ A/B Testing Framework test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ab_testing_framework())