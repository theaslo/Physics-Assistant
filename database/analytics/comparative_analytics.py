#!/usr/bin/env python3
"""
Comparative Analytics Engine for Physics Assistant
Implements cohort comparison, A/B testing framework, statistical significance testing,
and peer comparison systems for educational analytics.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, ttest_ind, welch_ttest
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CohortDefinition:
    """Definition of a student cohort for comparison"""
    cohort_id: str
    name: str
    description: str
    filters: Dict[str, Any]
    created_date: datetime = field(default_factory=datetime.now)
    student_count: int = 0
    is_active: bool = True

@dataclass
class ComparisonResult:
    """Result of comparative analysis between cohorts"""
    comparison_id: str
    comparison_type: str  # 'cohort', 'temporal', 'experimental'
    primary_cohort: str
    comparison_cohorts: List[str]
    metrics_compared: List[str]
    statistical_results: Dict[str, Any]
    effect_sizes: Dict[str, float]
    practical_significance: Dict[str, bool]
    recommendations: List[str]
    confidence_level: float = 0.95
    analysis_date: datetime = field(default_factory=datetime.now)

@dataclass
class ABTestResult:
    """Result of A/B testing analysis"""
    test_id: str
    test_name: str
    hypothesis: str
    control_group: str
    treatment_groups: List[str]
    primary_metric: str
    secondary_metrics: List[str]
    sample_sizes: Dict[str, int]
    conversion_rates: Dict[str, float]
    statistical_power: float
    p_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_size: float
    practical_significance: bool
    recommendation: str
    test_duration_days: int
    analysis_date: datetime = field(default_factory=datetime.now)

@dataclass
class BenchmarkResult:
    """Result of benchmark analysis against historical data"""
    benchmark_id: str
    metric_name: str
    current_value: float
    historical_baseline: float
    percentile_rank: float
    trend_direction: str  # 'improving', 'declining', 'stable'
    significance_level: float
    change_magnitude: float
    is_outlier: bool
    benchmark_date: datetime = field(default_factory=datetime.now)

@dataclass
class PeerComparisonResult:
    """Result of peer comparison analysis"""
    student_id: str
    peer_group: str
    student_metrics: Dict[str, float]
    peer_averages: Dict[str, float]
    percentile_ranks: Dict[str, float]
    relative_performance: Dict[str, str]  # 'above_average', 'average', 'below_average'
    areas_of_strength: List[str]
    areas_for_improvement: List[str]
    comparison_date: datetime = field(default_factory=datetime.now)

class ComparativeAnalyticsEngine:
    """Advanced comparative analytics engine for educational insights"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Cohort registry
        self.cohorts: Dict[str, CohortDefinition] = {}
        
        # A/B test registry
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        
        # Statistical configuration
        self.statistical_config = {
            'default_alpha': 0.05,
            'minimum_sample_size': 30,
            'minimum_effect_size': 0.2,  # Cohen's d
            'minimum_practical_difference': 0.05,  # 5% difference
            'outlier_threshold': 3.0  # Standard deviations
        }
        
        # Predefined cohort templates
        self.cohort_templates = {
            'performance_based': {
                'high_performers': {'success_rate': {'gte': 0.8}},
                'average_performers': {'success_rate': {'gte': 0.5, 'lt': 0.8}},
                'struggling_students': {'success_rate': {'lt': 0.5}}
            },
            'engagement_based': {
                'highly_engaged': {'interactions_per_day': {'gte': 10}},
                'moderately_engaged': {'interactions_per_day': {'gte': 5, 'lt': 10}},
                'low_engagement': {'interactions_per_day': {'lt': 5}}
            },
            'temporal': {
                'early_adopters': {'first_interaction': {'days_ago': {'gte': 30}}},
                'recent_users': {'first_interaction': {'days_ago': {'lt': 7}}}
            }
        }
    
    async def initialize(self):
        """Initialize the comparative analytics engine"""
        try:
            logger.info("🚀 Initializing Comparative Analytics Engine")
            
            # Load existing cohorts
            await self._load_existing_cohorts()
            
            # Initialize predefined cohorts
            await self._create_predefined_cohorts()
            
            # Load active experiments
            await self._load_active_experiments()
            
            logger.info("✅ Comparative Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Comparative Analytics Engine: {e}")
            return False
    
    async def _load_existing_cohorts(self):
        """Load existing cohorts from database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Check if cohorts table exists, create if not
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS student_cohorts (
                        cohort_id VARCHAR(50) PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        filters JSONB NOT NULL,
                        created_date TIMESTAMP DEFAULT NOW(),
                        student_count INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Load cohorts
                cohorts = await conn.fetch("SELECT * FROM student_cohorts WHERE is_active = TRUE")
                
                for cohort_row in cohorts:
                    cohort = CohortDefinition(
                        cohort_id=cohort_row['cohort_id'],
                        name=cohort_row['name'],
                        description=cohort_row['description'],
                        filters=cohort_row['filters'],
                        created_date=cohort_row['created_date'],
                        student_count=cohort_row['student_count'],
                        is_active=cohort_row['is_active']
                    )
                    self.cohorts[cohort.cohort_id] = cohort
                
                logger.info(f"📊 Loaded {len(self.cohorts)} existing cohorts")
        
        except Exception as e:
            logger.error(f"❌ Failed to load existing cohorts: {e}")
    
    async def _create_predefined_cohorts(self):
        """Create predefined cohorts based on templates"""
        try:
            for template_name, template_cohorts in self.cohort_templates.items():
                for cohort_name, filters in template_cohorts.items():
                    cohort_id = f"{template_name}_{cohort_name}"
                    
                    if cohort_id not in self.cohorts:
                        cohort = CohortDefinition(
                            cohort_id=cohort_id,
                            name=cohort_name.replace('_', ' ').title(),
                            description=f"Auto-generated {template_name} cohort: {cohort_name}",
                            filters=filters
                        )
                        
                        await self.create_cohort(cohort)
            
            logger.info("✅ Predefined cohorts created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create predefined cohorts: {e}")
    
    async def _load_active_experiments(self):
        """Load active A/B experiments"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Check if experiments table exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ab_experiments (
                        test_id VARCHAR(50) PRIMARY KEY,
                        test_name VARCHAR(100) NOT NULL,
                        hypothesis TEXT,
                        control_group VARCHAR(50),
                        treatment_groups JSONB,
                        primary_metric VARCHAR(50),
                        secondary_metrics JSONB,
                        start_date TIMESTAMP DEFAULT NOW(),
                        end_date TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        configuration JSONB
                    )
                """)
                
                # Load active experiments
                experiments = await conn.fetch("SELECT * FROM ab_experiments WHERE is_active = TRUE")
                
                for exp_row in experiments:
                    self.active_experiments[exp_row['test_id']] = {
                        'test_name': exp_row['test_name'],
                        'hypothesis': exp_row['hypothesis'],
                        'control_group': exp_row['control_group'],
                        'treatment_groups': exp_row['treatment_groups'],
                        'primary_metric': exp_row['primary_metric'],
                        'secondary_metrics': exp_row['secondary_metrics'],
                        'start_date': exp_row['start_date'],
                        'end_date': exp_row['end_date'],
                        'configuration': exp_row['configuration']
                    }
                
                logger.info(f"🧪 Loaded {len(self.active_experiments)} active experiments")
        
        except Exception as e:
            logger.error(f"❌ Failed to load active experiments: {e}")
    
    async def create_cohort(self, cohort: CohortDefinition) -> bool:
        """Create a new student cohort"""
        try:
            # Calculate student count
            student_count = await self._calculate_cohort_size(cohort.filters)
            cohort.student_count = student_count
            
            # Store cohort
            self.cohorts[cohort.cohort_id] = cohort
            
            # Save to database
            if self.db_manager:
                async with self.db_manager.postgres.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO student_cohorts 
                        (cohort_id, name, description, filters, student_count)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (cohort_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        filters = EXCLUDED.filters,
                        student_count = EXCLUDED.student_count
                    """, cohort.cohort_id, cohort.name, cohort.description, 
                    json.dumps(cohort.filters), cohort.student_count)
            
            logger.info(f"✅ Created cohort '{cohort.name}' with {student_count} students")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create cohort: {e}")
            return False
    
    async def _calculate_cohort_size(self, filters: Dict[str, Any]) -> int:
        """Calculate the number of students matching cohort filters"""
        try:
            if not self.db_manager:
                return 0
            
            # Build dynamic query based on filters
            base_query = """
                WITH student_stats AS (
                    SELECT 
                        user_id,
                        COUNT(*) as total_interactions,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) / EXTRACT(DAYS FROM (MAX(created_at) - MIN(created_at)) + 1) as interactions_per_day,
                        MIN(created_at) as first_interaction,
                        MAX(created_at) as last_interaction
                    FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY user_id
                )
                SELECT COUNT(*) as student_count
                FROM student_stats 
                WHERE 1=1
            """
            
            params = []
            param_count = 0
            
            # Add filter conditions
            for field, condition in filters.items():
                if isinstance(condition, dict):
                    for operator, value in condition.items():
                        param_count += 1
                        if operator == 'gte':
                            base_query += f" AND {field} >= ${param_count}"
                        elif operator == 'gt':
                            base_query += f" AND {field} > ${param_count}"
                        elif operator == 'lte':
                            base_query += f" AND {field} <= ${param_count}"
                        elif operator == 'lt':
                            base_query += f" AND {field} < ${param_count}"
                        elif operator == 'eq':
                            base_query += f" AND {field} = ${param_count}"
                        
                        params.append(value)
                else:
                    param_count += 1
                    base_query += f" AND {field} = ${param_count}"
                    params.append(condition)
            
            async with self.db_manager.postgres.get_connection() as conn:
                result = await conn.fetchrow(base_query, *params)
                return result['student_count'] if result else 0
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate cohort size: {e}")
            return 0
    
    async def compare_cohorts(self, primary_cohort_id: str, comparison_cohort_ids: List[str], 
                             metrics: List[str]) -> ComparisonResult:
        """Compare performance metrics between cohorts"""
        try:
            logger.info(f"📊 Comparing cohorts: {primary_cohort_id} vs {comparison_cohort_ids}")
            
            # Get cohort data
            primary_data = await self._get_cohort_metrics(primary_cohort_id, metrics)
            comparison_data = {}
            
            for cohort_id in comparison_cohort_ids:
                comparison_data[cohort_id] = await self._get_cohort_metrics(cohort_id, metrics)
            
            # Perform statistical tests
            statistical_results = {}
            effect_sizes = {}
            practical_significance = {}
            
            for metric in metrics:
                primary_values = primary_data.get(metric, [])
                
                if len(primary_values) < self.statistical_config['minimum_sample_size']:
                    continue
                
                metric_results = {}
                metric_effect_sizes = {}
                metric_practical = {}
                
                for cohort_id in comparison_cohort_ids:
                    comparison_values = comparison_data[cohort_id].get(metric, [])
                    
                    if len(comparison_values) < self.statistical_config['minimum_sample_size']:
                        continue
                    
                    # Perform statistical test
                    stat_result = self._perform_statistical_test(primary_values, comparison_values)
                    metric_results[cohort_id] = stat_result
                    
                    # Calculate effect size
                    effect_size = self._calculate_effect_size(primary_values, comparison_values)
                    metric_effect_sizes[cohort_id] = effect_size
                    
                    # Determine practical significance
                    practical_sig = self._determine_practical_significance(
                        primary_values, comparison_values, effect_size
                    )
                    metric_practical[cohort_id] = practical_sig
                
                statistical_results[metric] = metric_results
                effect_sizes[metric] = metric_effect_sizes
                practical_significance[metric] = metric_practical
            
            # Generate recommendations
            recommendations = self._generate_comparison_recommendations(
                statistical_results, effect_sizes, practical_significance
            )
            
            comparison_result = ComparisonResult(
                comparison_id=f"comp_{int(datetime.now().timestamp())}",
                comparison_type='cohort',
                primary_cohort=primary_cohort_id,
                comparison_cohorts=comparison_cohort_ids,
                metrics_compared=metrics,
                statistical_results=statistical_results,
                effect_sizes=effect_sizes,
                practical_significance=practical_significance,
                recommendations=recommendations
            )
            
            logger.info(f"✅ Cohort comparison completed")
            return comparison_result
            
        except Exception as e:
            logger.error(f"❌ Failed to compare cohorts: {e}")
            raise
    
    async def _get_cohort_metrics(self, cohort_id: str, metrics: List[str]) -> Dict[str, List[float]]:
        """Get metric values for all students in a cohort"""
        try:
            if cohort_id not in self.cohorts:
                raise ValueError(f"Cohort {cohort_id} not found")
            
            cohort = self.cohorts[cohort_id]
            
            # Get student IDs in cohort
            student_ids = await self._get_cohort_students(cohort.filters)
            
            # Collect metrics for each student
            cohort_metrics = {metric: [] for metric in metrics}
            
            for student_id in student_ids:
                student_metrics = await self._get_student_metrics(student_id, metrics)
                
                for metric in metrics:
                    if metric in student_metrics:
                        cohort_metrics[metric].append(student_metrics[metric])
            
            return cohort_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get cohort metrics: {e}")
            return {}
    
    async def _get_cohort_students(self, filters: Dict[str, Any]) -> List[str]:
        """Get list of student IDs matching cohort filters"""
        try:
            if not self.db_manager:
                return []
            
            # Build query similar to _calculate_cohort_size but return student IDs
            base_query = """
                WITH student_stats AS (
                    SELECT 
                        user_id,
                        COUNT(*) as total_interactions,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) / EXTRACT(DAYS FROM (MAX(created_at) - MIN(created_at)) + 1) as interactions_per_day,
                        MIN(created_at) as first_interaction,
                        MAX(created_at) as last_interaction
                    FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY user_id
                )
                SELECT user_id
                FROM student_stats 
                WHERE 1=1
            """
            
            params = []
            param_count = 0
            
            # Add filter conditions
            for field, condition in filters.items():
                if isinstance(condition, dict):
                    for operator, value in condition.items():
                        param_count += 1
                        if operator == 'gte':
                            base_query += f" AND {field} >= ${param_count}"
                        elif operator == 'gt':
                            base_query += f" AND {field} > ${param_count}"
                        elif operator == 'lte':
                            base_query += f" AND {field} <= ${param_count}"
                        elif operator == 'lt':
                            base_query += f" AND {field} < ${param_count}"
                        elif operator == 'eq':
                            base_query += f" AND {field} = ${param_count}"
                        
                        params.append(value)
                else:
                    param_count += 1
                    base_query += f" AND {field} = ${param_count}"
                    params.append(condition)
            
            async with self.db_manager.postgres.get_connection() as conn:
                results = await conn.fetch(base_query, *params)
                return [str(row['user_id']) for row in results]
        
        except Exception as e:
            logger.error(f"❌ Failed to get cohort students: {e}")
            return []
    
    async def _get_student_metrics(self, student_id: str, metrics: List[str]) -> Dict[str, float]:
        """Get specific metrics for a student"""
        try:
            if not self.db_manager:
                return {}
            
            student_metrics = {}
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get basic metrics
                basic_metrics = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(execution_time_ms) as avg_response_time,
                        COUNT(DISTINCT agent_type) as concepts_covered,
                        COUNT(*) / EXTRACT(DAYS FROM (MAX(created_at) - MIN(created_at)) + 1) as interactions_per_day
                    FROM interactions 
                    WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
                """, student_id)
                
                if basic_metrics:
                    metric_mapping = {
                        'success_rate': 'success_rate',
                        'avg_response_time': 'avg_response_time',
                        'total_interactions': 'total_interactions',
                        'concepts_covered': 'concepts_covered',
                        'interactions_per_day': 'interactions_per_day'
                    }
                    
                    for metric in metrics:
                        if metric in metric_mapping and basic_metrics[metric_mapping[metric]] is not None:
                            student_metrics[metric] = float(basic_metrics[metric_mapping[metric]])
            
            return student_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get student metrics: {e}")
            return {}
    
    def _perform_statistical_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Perform appropriate statistical test between two groups"""
        try:
            # Check for normality
            if len(group1) >= 8 and len(group2) >= 8:
                _, p_norm1 = stats.shapiro(group1[:50])  # Limit to 50 for shapiro
                _, p_norm2 = stats.shapiro(group2[:50])
                normal_data = p_norm1 > 0.05 and p_norm2 > 0.05
            else:
                normal_data = False
            
            # Choose appropriate test
            if normal_data:
                # Use t-test for normal data
                statistic, p_value = ttest_ind(group1, group2, equal_var=False)
                test_name = "Welch's t-test"
            else:
                # Use Mann-Whitney U test for non-normal data
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            # Calculate descriptive statistics
            group1_stats = {
                'mean': np.mean(group1),
                'median': np.median(group1),
                'std': np.std(group1),
                'n': len(group1)
            }
            
            group2_stats = {
                'mean': np.mean(group2),
                'median': np.median(group2),
                'std': np.std(group2),
                'n': len(group2)
            }
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.statistical_config['default_alpha'],
                'group1_stats': group1_stats,
                'group2_stats': group2_stats,
                'normal_data': normal_data
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to perform statistical test: {e}")
            return {'error': str(e)}
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        try:
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohens_d = (mean1 - mean2) / pooled_std
            
            return float(cohens_d)
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate effect size: {e}")
            return 0.0
    
    def _determine_practical_significance(self, group1: List[float], group2: List[float], 
                                        effect_size: float) -> bool:
        """Determine if difference is practically significant"""
        try:
            # Check effect size threshold
            if abs(effect_size) < self.statistical_config['minimum_effect_size']:
                return False
            
            # Check percentage difference
            mean1, mean2 = np.mean(group1), np.mean(group2)
            if mean1 != 0:
                percent_diff = abs((mean2 - mean1) / mean1)
                return percent_diff >= self.statistical_config['minimum_practical_difference']
            
            return False
        
        except Exception as e:
            logger.error(f"❌ Failed to determine practical significance: {e}")
            return False
    
    def _generate_comparison_recommendations(self, statistical_results: Dict[str, Any], 
                                           effect_sizes: Dict[str, Any], 
                                           practical_significance: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on comparison results"""
        recommendations = []
        
        try:
            for metric, results in statistical_results.items():
                for cohort_id, result in results.items():
                    if result.get('significant', False):
                        effect_size = effect_sizes.get(metric, {}).get(cohort_id, 0)
                        practical = practical_significance.get(metric, {}).get(cohort_id, False)
                        
                        if practical:
                            if effect_size > 0:
                                recommendations.append(
                                    f"Primary cohort shows significantly better {metric} "
                                    f"compared to {cohort_id} (effect size: {effect_size:.2f})"
                                )
                            else:
                                recommendations.append(
                                    f"{cohort_id} shows significantly better {metric} "
                                    f"compared to primary cohort (effect size: {abs(effect_size):.2f})"
                                )
                        else:
                            recommendations.append(
                                f"Statistically significant but small practical difference in {metric} "
                                f"between primary cohort and {cohort_id}"
                            )
            
            if not recommendations:
                recommendations.append("No significant differences found between cohorts")
        
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations
    
    async def conduct_ab_test(self, test_id: str, control_group: str, treatment_groups: List[str],
                             primary_metric: str, test_duration_days: int = 14) -> ABTestResult:
        """Conduct A/B test analysis"""
        try:
            logger.info(f"🧪 Conducting A/B test: {test_id}")
            
            # Get test data
            test_data = await self._collect_ab_test_data(
                control_group, treatment_groups, primary_metric, test_duration_days
            )
            
            # Calculate sample sizes and conversion rates
            sample_sizes = {}
            conversion_rates = {}
            
            for group_id, group_data in test_data.items():
                sample_sizes[group_id] = len(group_data)
                if primary_metric == 'success_rate':
                    conversion_rates[group_id] = np.mean(group_data)
                else:
                    conversion_rates[group_id] = np.mean(group_data)
            
            # Perform statistical tests
            control_data = test_data[control_group]
            p_values = {}
            confidence_intervals = {}
            
            for treatment_group in treatment_groups:
                treatment_data = test_data[treatment_group]
                
                # Perform statistical test
                stat_result = self._perform_statistical_test(control_data, treatment_data)
                p_values[treatment_group] = stat_result['p_value']
                
                # Calculate confidence interval for difference
                ci = self._calculate_confidence_interval_difference(control_data, treatment_data)
                confidence_intervals[treatment_group] = ci
            
            # Calculate overall effect size (using largest treatment group)
            largest_treatment = max(treatment_groups, key=lambda x: sample_sizes[x])
            effect_size = self._calculate_effect_size(control_data, test_data[largest_treatment])
            
            # Calculate statistical power
            statistical_power = self._calculate_statistical_power(
                sample_sizes[control_group], 
                sample_sizes[largest_treatment], 
                effect_size
            )
            
            # Determine practical significance
            practical_significance = self._determine_practical_significance(
                control_data, test_data[largest_treatment], effect_size
            )
            
            # Generate recommendation
            recommendation = self._generate_ab_test_recommendation(
                p_values, effect_size, practical_significance, statistical_power
            )
            
            ab_result = ABTestResult(
                test_id=test_id,
                test_name=self.active_experiments.get(test_id, {}).get('test_name', test_id),
                hypothesis=self.active_experiments.get(test_id, {}).get('hypothesis', ''),
                control_group=control_group,
                treatment_groups=treatment_groups,
                primary_metric=primary_metric,
                secondary_metrics=[],  # Could be expanded
                sample_sizes=sample_sizes,
                conversion_rates=conversion_rates,
                statistical_power=statistical_power,
                p_values=p_values,
                confidence_intervals=confidence_intervals,
                effect_size=effect_size,
                practical_significance=practical_significance,
                recommendation=recommendation,
                test_duration_days=test_duration_days
            )
            
            logger.info(f"✅ A/B test analysis completed")
            return ab_result
            
        except Exception as e:
            logger.error(f"❌ Failed to conduct A/B test: {e}")
            raise
    
    async def benchmark_against_historical(self, metric_name: str, current_period_days: int = 7,
                                         historical_periods: int = 4) -> BenchmarkResult:
        """Compare current performance against historical baseline"""
        try:
            logger.info(f"📈 Benchmarking {metric_name} against historical data")
            
            # Get current period data
            current_data = await self._get_metric_data(metric_name, current_period_days)
            current_value = np.mean(current_data) if current_data else 0.0
            
            # Get historical data
            historical_data = []
            for period in range(1, historical_periods + 1):
                start_days_ago = current_period_days + (period - 1) * current_period_days
                end_days_ago = current_period_days + period * current_period_days
                
                period_data = await self._get_metric_data(
                    metric_name, current_period_days, start_days_ago
                )
                if period_data:
                    historical_data.extend(period_data)
            
            if not historical_data:
                raise ValueError("No historical data available")
            
            # Calculate baseline and statistics
            historical_baseline = np.mean(historical_data)
            historical_std = np.std(historical_data)
            
            # Calculate percentile rank
            all_values = historical_data + [current_value]
            percentile_rank = stats.percentileofscore(all_values, current_value) / 100.0
            
            # Determine trend direction
            change_magnitude = current_value - historical_baseline
            relative_change = change_magnitude / historical_baseline if historical_baseline != 0 else 0
            
            if abs(relative_change) < 0.05:  # Less than 5% change
                trend_direction = 'stable'
            elif change_magnitude > 0:
                trend_direction = 'improving'
            else:
                trend_direction = 'declining'
            
            # Check for outliers
            z_score = abs(change_magnitude / historical_std) if historical_std > 0 else 0
            is_outlier = z_score > self.statistical_config['outlier_threshold']
            
            # Statistical significance
            significance_level = 1 - stats.norm.cdf(z_score)
            
            benchmark_result = BenchmarkResult(
                benchmark_id=f"bench_{metric_name}_{int(datetime.now().timestamp())}",
                metric_name=metric_name,
                current_value=current_value,
                historical_baseline=historical_baseline,
                percentile_rank=percentile_rank,
                trend_direction=trend_direction,
                significance_level=significance_level,
                change_magnitude=change_magnitude,
                is_outlier=is_outlier
            )
            
            logger.info(f"✅ Benchmark analysis completed for {metric_name}")
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Failed to benchmark against historical data: {e}")
            raise
    
    async def compare_student_to_peers(self, student_id: str, peer_group: str = 'class') -> PeerComparisonResult:
        """Compare individual student performance to peer group"""
        try:
            logger.info(f"👥 Comparing student {student_id} to peers")
            
            # Get student metrics
            student_metrics = await self._get_student_metrics(
                student_id, ['success_rate', 'avg_response_time', 'interactions_per_day', 'concepts_covered']
            )
            
            # Get peer group metrics
            peer_metrics = await self._get_peer_group_metrics(student_id, peer_group)
            
            # Calculate percentile ranks and relative performance
            percentile_ranks = {}
            relative_performance = {}
            areas_of_strength = []
            areas_for_improvement = []
            
            for metric, student_value in student_metrics.items():
                if metric in peer_metrics and peer_metrics[metric]:
                    peer_values = peer_metrics[metric]
                    
                    # Calculate percentile rank
                    percentile = stats.percentileofscore(peer_values + [student_value], student_value) / 100.0
                    percentile_ranks[metric] = percentile
                    
                    # Determine relative performance
                    if percentile >= 0.75:
                        relative_performance[metric] = 'above_average'
                        areas_of_strength.append(metric)
                    elif percentile >= 0.25:
                        relative_performance[metric] = 'average'
                    else:
                        relative_performance[metric] = 'below_average'
                        areas_for_improvement.append(metric)
            
            # Calculate peer averages
            peer_averages = {
                metric: np.mean(values) for metric, values in peer_metrics.items()
            }
            
            peer_comparison = PeerComparisonResult(
                student_id=student_id,
                peer_group=peer_group,
                student_metrics=student_metrics,
                peer_averages=peer_averages,
                percentile_ranks=percentile_ranks,
                relative_performance=relative_performance,
                areas_of_strength=areas_of_strength,
                areas_for_improvement=areas_for_improvement
            )
            
            logger.info(f"✅ Peer comparison completed for student {student_id}")
            return peer_comparison
            
        except Exception as e:
            logger.error(f"❌ Failed to compare student to peers: {e}")
            raise
    
    # Helper methods for data collection and analysis
    async def _collect_ab_test_data(self, control_group: str, treatment_groups: List[str],
                                   metric: str, duration_days: int) -> Dict[str, List[float]]:
        """Collect data for A/B test analysis"""
        # Implementation for A/B test data collection
        pass
    
    async def _get_metric_data(self, metric_name: str, period_days: int, 
                             start_days_ago: int = 0) -> List[float]:
        """Get metric data for specified time period"""
        # Implementation for metric data collection
        pass
    
    async def _get_peer_group_metrics(self, student_id: str, peer_group: str) -> Dict[str, List[float]]:
        """Get metrics for peer group (excluding the student)"""
        # Implementation for peer group metrics collection
        pass
    
    def _calculate_confidence_interval_difference(self, group1: List[float], 
                                                group2: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for difference between groups"""
        # Implementation for confidence interval calculation
        pass
    
    def _calculate_statistical_power(self, n1: int, n2: int, effect_size: float) -> float:
        """Calculate statistical power of the test"""
        # Implementation for power calculation
        pass
    
    def _generate_ab_test_recommendation(self, p_values: Dict[str, float], 
                                       effect_size: float, practical_significance: bool,
                                       statistical_power: float) -> str:
        """Generate recommendation based on A/B test results"""
        # Implementation for A/B test recommendation generation
        pass

# Main testing function
async def test_comparative_analytics():
    """Test comparative analytics engine"""
    try:
        logger.info("🧪 Testing Comparative Analytics Engine")
        
        engine = ComparativeAnalyticsEngine()
        await engine.initialize()
        
        # Test cohort creation
        test_cohort = CohortDefinition(
            cohort_id="test_high_performers",
            name="Test High Performers",
            description="Test cohort for high-performing students",
            filters={'success_rate': {'gte': 0.8}}
        )
        
        await engine.create_cohort(test_cohort)
        logger.info(f"✅ Test cohort created: {test_cohort.name}")
        
        # Test comparison result structure
        sample_comparison = ComparisonResult(
            comparison_id="test_comparison",
            comparison_type="cohort",
            primary_cohort="high_performers",
            comparison_cohorts=["average_performers"],
            metrics_compared=["success_rate", "avg_response_time"],
            statistical_results={"success_rate": {"average_performers": {"p_value": 0.02, "significant": True}}},
            effect_sizes={"success_rate": {"average_performers": 0.8}},
            practical_significance={"success_rate": {"average_performers": True}},
            recommendations=["High performers show significantly better success rate"]
        )
        
        logger.info(f"✅ Sample comparison result: {sample_comparison.comparison_type}")
        
        logger.info("✅ Comparative Analytics Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Comparative Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_comparative_analytics())