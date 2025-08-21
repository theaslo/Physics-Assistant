#!/usr/bin/env python3
"""
Automated Insights Generation Engine for Physics Assistant
Implements AI-powered insight generation, natural language summaries, automated alert systems,
actionable recommendation generation, and executive summary reports.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from textstat import flesch_reading_ease
import re
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AutomatedInsight:
    """Automated insight generated from analytics data"""
    insight_id: str
    insight_type: str  # 'trend', 'anomaly', 'performance', 'recommendation'
    title: str
    summary: str
    detailed_explanation: str
    confidence_score: float
    importance_level: str  # 'critical', 'high', 'medium', 'low'
    supporting_data: Dict[str, Any]
    actionable_recommendations: List[str]
    affected_entities: List[str]  # students, concepts, etc.
    time_frame: Tuple[datetime, datetime]
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class NaturalLanguageSummary:
    """Natural language summary of complex analytics data"""
    summary_id: str
    summary_type: str  # 'student_progress', 'class_overview', 'content_analysis'
    target_audience: str  # 'educator', 'administrator', 'student'
    headline: str
    key_points: List[str]
    narrative_text: str
    metrics_mentioned: List[str]
    reading_level: float
    length_category: str  # 'brief', 'standard', 'detailed'
    visualizations_referenced: List[str]

@dataclass
class AutomatedAlert:
    """Automated alert triggered by significant changes or patterns"""
    alert_id: str
    alert_type: str  # 'performance_drop', 'anomaly_detected', 'milestone_achieved'
    severity: str  # 'critical', 'warning', 'info'
    trigger_condition: str
    alert_message: str
    recommended_actions: List[str]
    affected_entities: List[str]
    threshold_values: Dict[str, float]
    current_values: Dict[str, float]
    time_sensitive: bool
    auto_resolve: bool
    escalation_path: List[str]
    triggered_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutiveSummary:
    """Executive summary report with key metrics and insights"""
    summary_id: str
    report_period: Tuple[datetime, datetime]
    executive_headline: str
    key_achievements: List[str]
    areas_of_concern: List[str]
    performance_indicators: Dict[str, Dict[str, Any]]
    trend_analysis: Dict[str, str]
    strategic_recommendations: List[str]
    budget_impact: Optional[Dict[str, float]]
    next_period_projections: Dict[str, float]
    appendix_data: Dict[str, Any]

@dataclass
class InsightPattern:
    """Pattern template for generating insights"""
    pattern_id: str
    pattern_type: str
    trigger_conditions: Dict[str, Any]
    narrative_template: str
    importance_rules: Dict[str, Any]
    recommendation_templates: List[str]

class AutomatedInsightsEngine:
    """AI-powered automated insights generation engine"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Insights storage
        self.generated_insights: Dict[str, AutomatedInsight] = {}
        self.nl_summaries: Dict[str, NaturalLanguageSummary] = {}
        self.active_alerts: Dict[str, AutomatedAlert] = {}
        self.executive_summaries: Dict[str, ExecutiveSummary] = {}
        
        # Insight generation patterns
        self.insight_patterns = self._initialize_insight_patterns()
        
        # Language generation templates
        self.language_templates = self._initialize_language_templates()
        
        # Configuration
        self.config = {
            'insight_generation': {
                'minimum_confidence': 0.6,
                'significance_threshold': 0.05,
                'trend_detection_window': 7,  # days
                'anomaly_sensitivity': 2.0,  # standard deviations
                'importance_weights': {
                    'impact_size': 0.4,
                    'affected_population': 0.3,
                    'trend_strength': 0.2,
                    'actionability': 0.1
                }
            },
            'language_generation': {
                'target_reading_level': 8.0,  # 8th grade
                'max_sentence_length': 20,
                'preferred_voice': 'active',
                'technical_detail_level': 'medium'
            },
            'alert_thresholds': {
                'performance_drop': 0.15,  # 15% drop
                'engagement_decline': 0.20,
                'success_rate_drop': 0.10,
                'response_time_increase': 0.30,
                'completion_rate_drop': 0.12
            }
        }
    
    async def initialize(self):
        """Initialize the automated insights engine"""
        try:
            logger.info("🚀 Initializing Automated Insights Engine")
            
            # Create insights tables
            await self._create_insights_tables()
            
            # Load existing insights
            await self._load_existing_insights()
            
            # Initialize pattern matchers
            self._compile_pattern_matchers()
            
            logger.info("✅ Automated Insights Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Automated Insights Engine: {e}")
            return False
    
    def _initialize_insight_patterns(self) -> Dict[str, InsightPattern]:
        """Initialize insight generation patterns"""
        patterns = {}
        
        # Performance trend patterns
        patterns['performance_improvement'] = InsightPattern(
            pattern_id='performance_improvement',
            pattern_type='trend',
            trigger_conditions={
                'success_rate_increase': {'min_change': 0.1, 'min_duration': 3},
                'statistical_significance': {'p_value': 0.05}
            },
            narrative_template="Student performance has shown significant improvement over the past {duration} days, with success rates increasing by {change_percent}%. This positive trend suggests that recent educational interventions are having the desired effect.",
            importance_rules={
                'high': {'change_percent': 15, 'affected_students': 50},
                'medium': {'change_percent': 10, 'affected_students': 20},
                'low': {'change_percent': 5, 'affected_students': 10}
            },
            recommendation_templates=[
                "Continue current teaching strategies that are proving effective",
                "Consider expanding successful interventions to other student groups",
                "Document best practices for future implementation"
            ]
        )
        
        patterns['engagement_decline'] = InsightPattern(
            pattern_id='engagement_decline',
            pattern_type='trend',
            trigger_conditions={
                'engagement_score_decrease': {'min_change': 0.15, 'min_duration': 5},
                'interaction_frequency_drop': {'min_change': 0.20}
            },
            narrative_template="Student engagement has declined by {change_percent}% over the past {duration} days. Daily interaction rates have dropped from {previous_rate} to {current_rate} per student. This trend requires immediate attention to prevent further disengagement.",
            importance_rules={
                'critical': {'change_percent': 30, 'affected_students': 100},
                'high': {'change_percent': 20, 'affected_students': 50},
                'medium': {'change_percent': 15, 'affected_students': 25}
            },
            recommendation_templates=[
                "Implement gamification elements to re-engage students",
                "Review content difficulty and adjust if necessary",
                "Introduce interactive activities and real-world applications",
                "Consider individual outreach to most affected students"
            ]
        )
        
        patterns['concept_difficulty_spike'] = InsightPattern(
            pattern_id='concept_difficulty_spike',
            pattern_type='anomaly',
            trigger_conditions={
                'success_rate_drop': {'min_change': 0.25, 'concept_specific': True},
                'help_request_increase': {'min_change': 0.40}
            },
            narrative_template="The concept '{concept_name}' is showing unexpected difficulty for students, with success rates dropping to {success_rate}% and help requests increasing by {help_increase}%. This suggests the current approach to teaching this concept may need revision.",
            importance_rules={
                'high': {'success_rate': 50, 'help_increase': 50},
                'medium': {'success_rate': 60, 'help_increase': 40},
                'low': {'success_rate': 70, 'help_increase': 30}
            },
            recommendation_templates=[
                "Review prerequisite concepts and ensure adequate coverage",
                "Provide additional scaffolding and step-by-step guidance",
                "Add more practice problems with varying difficulty levels",
                "Consider alternative explanation approaches or examples"
            ]
        )
        
        patterns['student_achievement'] = InsightPattern(
            pattern_id='student_achievement',
            pattern_type='performance',
            trigger_conditions={
                'mastery_milestone': {'concepts_mastered': 5, 'time_frame': 14},
                'consistent_high_performance': {'success_rate': 0.85, 'duration': 7}
            },
            narrative_template="{student_count} students have achieved significant milestones, mastering {concepts_count} concepts with an average success rate of {success_rate}%. These high-performing students demonstrate excellent progress and may benefit from advanced challenges.",
            importance_rules={
                'high': {'student_count': 10, 'concepts_count': 8},
                'medium': {'student_count': 5, 'concepts_count': 5},
                'low': {'student_count': 2, 'concepts_count': 3}
            },
            recommendation_templates=[
                "Provide enrichment activities for high-achieving students",
                "Consider peer tutoring opportunities",
                "Offer advanced problem sets or real-world applications",
                "Recognize and celebrate student achievements"
            ]
        )
        
        return patterns
    
    def _initialize_language_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize natural language generation templates"""
        templates = {
            'trend_descriptions': {
                'increasing': "showing an upward trend",
                'decreasing': "exhibiting a downward trend", 
                'stable': "remaining relatively stable",
                'volatile': "displaying high variability",
                'improving': "demonstrating improvement",
                'declining': "experiencing decline"
            },
            'magnitude_descriptors': {
                'slight': [0, 5],      # 0-5% change
                'moderate': [5, 15],   # 5-15% change
                'significant': [15, 30], # 15-30% change
                'substantial': [30, 50], # 30-50% change
                'dramatic': [50, 100]    # 50%+ change
            },
            'time_descriptors': {
                'recent': [0, 3],      # 0-3 days
                'this_week': [3, 7],   # 3-7 days
                'past_week': [7, 14],  # 1-2 weeks
                'recent_weeks': [14, 30], # 2-4 weeks
                'past_month': [30, 60]    # 1-2 months
            },
            'confidence_phrases': {
                'high': "with high confidence",
                'medium': "indicates that",
                'low': "suggests that"
            },
            'action_verbs': {
                'improve': ['enhance', 'strengthen', 'boost', 'elevate'],
                'review': ['examine', 'analyze', 'assess', 'evaluate'],
                'implement': ['deploy', 'introduce', 'establish', 'initiate'],
                'monitor': ['track', 'observe', 'watch', 'follow']
            }
        }
        return templates
    
    def _compile_pattern_matchers(self):
        """Compile pattern matching rules for efficient insight detection"""
        try:
            self.compiled_patterns = {}
            
            for pattern_id, pattern in self.insight_patterns.items():
                compiled_conditions = {}
                
                for condition_name, condition_data in pattern.trigger_conditions.items():
                    # Create pattern matching functions
                    if 'min_change' in condition_data:
                        compiled_conditions[condition_name] = {
                            'type': 'threshold',
                            'threshold': condition_data['min_change'],
                            'direction': condition_data.get('direction', 'any')
                        }
                    elif 'p_value' in condition_data:
                        compiled_conditions[condition_name] = {
                            'type': 'statistical',
                            'threshold': condition_data['p_value']
                        }
                
                self.compiled_patterns[pattern_id] = compiled_conditions
            
            logger.info("✅ Pattern matchers compiled successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to compile pattern matchers: {e}")
    
    async def generate_insights(self, analytics_data: Dict[str, Any], 
                              time_window_days: int = 7) -> List[AutomatedInsight]:
        """Generate automated insights from analytics data"""
        try:
            logger.info("🧠 Generating automated insights from analytics data")
            
            insights = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_window_days)
            
            # 1. Trend-based insights
            trend_insights = await self._generate_trend_insights(analytics_data, start_date, end_date)
            insights.extend(trend_insights)
            
            # 2. Anomaly-based insights
            anomaly_insights = await self._generate_anomaly_insights(analytics_data, start_date, end_date)
            insights.extend(anomaly_insights)
            
            # 3. Performance-based insights
            performance_insights = await self._generate_performance_insights(analytics_data, start_date, end_date)
            insights.extend(performance_insights)
            
            # 4. Comparative insights
            comparative_insights = await self._generate_comparative_insights(analytics_data, start_date, end_date)
            insights.extend(comparative_insights)
            
            # 5. Predictive insights
            predictive_insights = await self._generate_predictive_insights(analytics_data, start_date, end_date)
            insights.extend(predictive_insights)
            
            # Filter and rank insights
            filtered_insights = self._filter_and_rank_insights(insights)
            
            # Store insights
            for insight in filtered_insights:
                self.generated_insights[insight.insight_id] = insight
                await self._save_insight(insight)
            
            logger.info(f"✅ Generated {len(filtered_insights)} automated insights")
            return filtered_insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate automated insights: {e}")
            return []
    
    async def _generate_trend_insights(self, analytics_data: Dict[str, Any], 
                                     start_date: datetime, end_date: datetime) -> List[AutomatedInsight]:
        """Generate insights based on trend analysis"""
        insights = []
        
        try:
            # Analyze key metrics trends
            key_metrics = ['success_rate', 'engagement_score', 'completion_rate', 'avg_response_time']
            
            for metric in key_metrics:
                if metric in analytics_data:
                    trend_data = analytics_data[metric]
                    trend_analysis = self._analyze_metric_trend(trend_data, metric)
                    
                    if trend_analysis['significant']:
                        insight = self._create_trend_insight(metric, trend_analysis, start_date, end_date)
                        if insight:
                            insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trend insights: {e}")
            return []
    
    def _analyze_metric_trend(self, trend_data: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
        """Analyze trend in a specific metric"""
        try:
            if len(trend_data) < 3:
                return {'significant': False}
            
            # Extract values and timestamps
            values = [point['value'] for point in trend_data]
            timestamps = [point['timestamp'] for point in trend_data]
            
            # Linear regression for trend analysis
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Calculate percentage change
            if values[0] != 0:
                pct_change = ((values[-1] - values[0]) / abs(values[0])) * 100
            else:
                pct_change = 0
            
            # Determine significance
            significant = (p_value < 0.05) and (abs(pct_change) > 5)
            
            return {
                'significant': significant,
                'slope': slope,
                'pct_change': pct_change,
                'p_value': p_value,
                'r_squared': r_value ** 2,
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'strength': abs(r_value),
                'start_value': values[0],
                'end_value': values[-1],
                'duration_days': len(values)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze metric trend: {e}")
            return {'significant': False}
    
    def _create_trend_insight(self, metric: str, trend_analysis: Dict[str, Any], 
                            start_date: datetime, end_date: datetime) -> Optional[AutomatedInsight]:
        """Create insight from trend analysis"""
        try:
            # Determine insight type and importance
            pct_change = abs(trend_analysis['pct_change'])
            direction = trend_analysis['direction']
            
            # Map to insight patterns
            if metric == 'success_rate' and direction == 'increasing' and pct_change > 10:
                pattern = self.insight_patterns['performance_improvement']
            elif metric == 'engagement_score' and direction == 'decreasing' and pct_change > 15:
                pattern = self.insight_patterns['engagement_decline']
            else:
                # Generic trend insight
                pattern = None
            
            if pattern:
                # Use pattern to generate insight
                narrative = pattern.narrative_template.format(
                    duration=trend_analysis['duration_days'],
                    change_percent=f"{pct_change:.1f}",
                    metric=metric.replace('_', ' ')
                )
            else:
                # Generate generic narrative
                magnitude = self._get_magnitude_descriptor(pct_change)
                time_desc = self._get_time_descriptor(trend_analysis['duration_days'])
                
                narrative = f"The metric '{metric.replace('_', ' ')}' has shown a {magnitude} {direction} trend over the {time_desc}, changing by {pct_change:.1f}%."
            
            # Determine importance level
            importance = self._determine_importance_level(metric, pct_change, trend_analysis['strength'])
            
            # Generate recommendations
            recommendations = self._generate_trend_recommendations(metric, direction, pct_change)
            
            insight_id = f"trend_{metric}_{int(start_date.timestamp())}"
            
            return AutomatedInsight(
                insight_id=insight_id,
                insight_type='trend',
                title=f"{metric.replace('_', ' ').title()} {direction.title()} Trend",
                summary=f"{metric.replace('_', ' ').title()} has {direction} by {pct_change:.1f}% over {trend_analysis['duration_days']} days",
                detailed_explanation=narrative,
                confidence_score=min(trend_analysis['strength'], 0.95),
                importance_level=importance,
                supporting_data={
                    'metric': metric,
                    'pct_change': pct_change,
                    'p_value': trend_analysis['p_value'],
                    'r_squared': trend_analysis['r_squared'],
                    'start_value': trend_analysis['start_value'],
                    'end_value': trend_analysis['end_value']
                },
                actionable_recommendations=recommendations,
                affected_entities=['all_students'],
                time_frame=(start_date, end_date)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create trend insight: {e}")
            return None
    
    def _get_magnitude_descriptor(self, pct_change: float) -> str:
        """Get appropriate magnitude descriptor for percentage change"""
        for magnitude, (min_val, max_val) in self.language_templates['magnitude_descriptors'].items():
            if min_val <= pct_change < max_val:
                return magnitude
        return 'dramatic'
    
    def _get_time_descriptor(self, days: int) -> str:
        """Get appropriate time descriptor for duration"""
        for time_desc, (min_val, max_val) in self.language_templates['time_descriptors'].items():
            if min_val <= days < max_val:
                return time_desc.replace('_', ' ')
        return 'extended period'
    
    def _determine_importance_level(self, metric: str, pct_change: float, strength: float) -> str:
        """Determine importance level of insight"""
        try:
            # Base importance on change magnitude and statistical strength
            combined_score = (pct_change / 100) * strength
            
            # Adjust based on metric type
            metric_weights = {
                'success_rate': 1.2,
                'engagement_score': 1.1,
                'completion_rate': 1.0,
                'avg_response_time': 0.8
            }
            
            weighted_score = combined_score * metric_weights.get(metric, 1.0)
            
            if weighted_score > 0.3:
                return 'critical'
            elif weighted_score > 0.2:
                return 'high'
            elif weighted_score > 0.1:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"❌ Failed to determine importance level: {e}")
            return 'medium'
    
    def _generate_trend_recommendations(self, metric: str, direction: str, pct_change: float) -> List[str]:
        """Generate actionable recommendations based on trend"""
        recommendations = []
        
        try:
            if metric == 'success_rate':
                if direction == 'increasing':
                    recommendations.extend([
                        "Continue current teaching strategies that are proving effective",
                        "Consider documenting successful approaches for broader implementation",
                        "Identify students who may benefit from advanced challenges"
                    ])
                else:
                    recommendations.extend([
                        "Review recent changes in curriculum or teaching methods",
                        "Provide additional support and scaffolding for struggling students",
                        "Consider adjusting difficulty levels or pacing"
                    ])
            
            elif metric == 'engagement_score':
                if direction == 'decreasing':
                    recommendations.extend([
                        "Introduce more interactive and hands-on activities",
                        "Survey students to understand engagement challenges",
                        "Consider gamification elements or competitive activities",
                        "Review content relevance and real-world applications"
                    ])
                else:
                    recommendations.extend([
                        "Maintain current engagement strategies",
                        "Share successful engagement practices with other educators"
                    ])
            
            elif metric == 'completion_rate':
                if direction == 'decreasing':
                    recommendations.extend([
                        "Break down complex tasks into smaller, manageable chunks",
                        "Provide clearer instructions and learning objectives",
                        "Implement progress tracking and milestone celebrations"
                    ])
            
            elif metric == 'avg_response_time':
                if direction == 'increasing':
                    recommendations.extend([
                        "Review content complexity and adjust if necessary",
                        "Provide additional examples and practice opportunities",
                        "Consider technical issues that might be slowing responses"
                    ])
            
            # Add magnitude-specific recommendations
            if pct_change > 20:
                recommendations.append("Given the significant change, consider immediate intervention")
            
            return recommendations[:4]  # Limit to 4 recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to generate trend recommendations: {e}")
            return ["Monitor the situation and gather additional data"]
    
    async def _generate_anomaly_insights(self, analytics_data: Dict[str, Any], 
                                       start_date: datetime, end_date: datetime) -> List[AutomatedInsight]:
        """Generate insights based on anomaly detection"""
        insights = []
        
        try:
            # Check for anomalies in different data types
            if 'anomalies' in analytics_data:
                for anomaly in analytics_data['anomalies']:
                    insight = self._create_anomaly_insight(anomaly, start_date, end_date)
                    if insight:
                        insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate anomaly insights: {e}")
            return []
    
    def _create_anomaly_insight(self, anomaly: Dict[str, Any], 
                               start_date: datetime, end_date: datetime) -> Optional[AutomatedInsight]:
        """Create insight from anomaly data"""
        try:
            anomaly_type = anomaly.get('type', 'unknown')
            severity = anomaly.get('severity', 'medium')
            
            # Generate narrative based on anomaly type
            if anomaly_type == 'statistical_outlier':
                narrative = f"An unusual data point was detected on {anomaly['timestamp']}, with a value of {anomaly['value']:.2f} that deviates significantly from the normal pattern."
            elif anomaly_type == 'concept_difficulty_spike':
                narrative = f"The concept '{anomaly.get('concept', 'Unknown')}' showed unexpected difficulty, with success rates dropping significantly below normal levels."
            else:
                narrative = f"An anomaly of type '{anomaly_type}' was detected that requires attention."
            
            recommendations = [
                "Investigate the underlying cause of this anomaly",
                "Check for external factors that might have influenced the data",
                "Monitor for similar patterns in future data"
            ]
            
            if severity == 'high':
                recommendations.insert(0, "Take immediate action to address this significant deviation")
            
            insight_id = f"anomaly_{anomaly_type}_{int(start_date.timestamp())}"
            
            return AutomatedInsight(
                insight_id=insight_id,
                insight_type='anomaly',
                title=f"Anomaly Detected: {anomaly_type.replace('_', ' ').title()}",
                summary=f"Unusual pattern detected in {anomaly.get('metric', 'data')}",
                detailed_explanation=narrative,
                confidence_score=anomaly.get('confidence', 0.8),
                importance_level=severity,
                supporting_data=anomaly,
                actionable_recommendations=recommendations,
                affected_entities=['data_integrity'],
                time_frame=(start_date, end_date)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to create anomaly insight: {e}")
            return None
    
    # Placeholder methods for additional insight types
    async def _generate_performance_insights(self, analytics_data, start_date, end_date):
        """Generate performance-based insights"""
        return []
    
    async def _generate_comparative_insights(self, analytics_data, start_date, end_date):
        """Generate comparative insights"""
        return []
    
    async def _generate_predictive_insights(self, analytics_data, start_date, end_date):
        """Generate predictive insights"""
        return []
    
    def _filter_and_rank_insights(self, insights: List[AutomatedInsight]) -> List[AutomatedInsight]:
        """Filter and rank insights by importance and confidence"""
        try:
            # Filter by minimum confidence
            min_confidence = self.config['insight_generation']['minimum_confidence']
            filtered = [insight for insight in insights if insight.confidence_score >= min_confidence]
            
            # Remove duplicates
            unique_insights = {}
            for insight in filtered:
                key = f"{insight.insight_type}_{insight.title}"
                if key not in unique_insights or insight.confidence_score > unique_insights[key].confidence_score:
                    unique_insights[key] = insight
            
            # Rank by importance and confidence
            importance_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            
            ranked_insights = sorted(
                unique_insights.values(),
                key=lambda x: (importance_order.get(x.importance_level, 0), x.confidence_score),
                reverse=True
            )
            
            return ranked_insights[:20]  # Limit to top 20 insights
            
        except Exception as e:
            logger.error(f"❌ Failed to filter and rank insights: {e}")
            return insights
    
    async def generate_natural_language_summary(self, analytics_data: Dict[str, Any], 
                                               summary_type: str, target_audience: str = 'educator') -> NaturalLanguageSummary:
        """Generate natural language summary of analytics data"""
        try:
            logger.info(f"📝 Generating natural language summary for {target_audience}")
            
            # Extract key metrics and insights
            key_metrics = self._extract_key_metrics(analytics_data)
            insights = analytics_data.get('insights', [])
            
            # Generate headline
            headline = self._generate_headline(key_metrics, insights, summary_type)
            
            # Generate key points
            key_points = self._generate_key_points(key_metrics, insights, target_audience)
            
            # Generate narrative text
            narrative_text = self._generate_narrative(key_metrics, insights, target_audience)
            
            # Analyze reading level
            reading_level = flesch_reading_ease(narrative_text)
            
            # Determine length category
            word_count = len(narrative_text.split())
            if word_count < 100:
                length_category = 'brief'
            elif word_count < 300:
                length_category = 'standard'
            else:
                length_category = 'detailed'
            
            summary_id = f"summary_{summary_type}_{int(datetime.now().timestamp())}"
            
            nl_summary = NaturalLanguageSummary(
                summary_id=summary_id,
                summary_type=summary_type,
                target_audience=target_audience,
                headline=headline,
                key_points=key_points,
                narrative_text=narrative_text,
                metrics_mentioned=list(key_metrics.keys()),
                reading_level=reading_level,
                length_category=length_category,
                visualizations_referenced=[]
            )
            
            # Store summary
            self.nl_summaries[summary_id] = nl_summary
            await self._save_nl_summary(nl_summary)
            
            logger.info(f"✅ Generated natural language summary: {summary_id}")
            return nl_summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate natural language summary: {e}")
            raise
    
    def _extract_key_metrics(self, analytics_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics from analytics data"""
        key_metrics = {}
        
        # Standard metrics
        metric_mappings = {
            'success_rate': 'Success Rate',
            'engagement_score': 'Engagement Score',
            'completion_rate': 'Completion Rate',
            'avg_response_time': 'Average Response Time',
            'total_interactions': 'Total Interactions',
            'active_users': 'Active Users'
        }
        
        for key, label in metric_mappings.items():
            if key in analytics_data:
                if isinstance(analytics_data[key], (int, float)):
                    key_metrics[label] = analytics_data[key]
                elif isinstance(analytics_data[key], list) and analytics_data[key]:
                    # Take the latest value if it's a time series
                    key_metrics[label] = analytics_data[key][-1]['value']
        
        return key_metrics
    
    def _generate_headline(self, key_metrics: Dict[str, float], insights: List[Dict], summary_type: str) -> str:
        """Generate compelling headline for summary"""
        try:
            if summary_type == 'student_progress':
                if 'Success Rate' in key_metrics:
                    success_rate = key_metrics['Success Rate'] * 100 if key_metrics['Success Rate'] <= 1 else key_metrics['Success Rate']
                    if success_rate >= 80:
                        return f"Strong Academic Performance: Students Achieving {success_rate:.0f}% Success Rate"
                    elif success_rate >= 60:
                        return f"Steady Progress: Students Maintaining {success_rate:.0f}% Success Rate"
                    else:
                        return f"Performance Challenge: Success Rate at {success_rate:.0f}% Requires Attention"
            
            elif summary_type == 'class_overview':
                if 'Active Users' in key_metrics:
                    users = int(key_metrics['Active Users'])
                    return f"Class Activity Report: {users} Students Actively Engaged in Learning"
            
            elif summary_type == 'content_analysis':
                return "Content Effectiveness Analysis: Insights from Student Interactions"
            
            return "Analytics Summary: Key Insights from Student Learning Data"
            
        except Exception as e:
            logger.error(f"❌ Failed to generate headline: {e}")
            return "Analytics Summary Report"
    
    def _generate_key_points(self, key_metrics: Dict[str, float], insights: List[Dict], 
                           target_audience: str) -> List[str]:
        """Generate key bullet points for summary"""
        key_points = []
        
        try:
            # Performance summary
            if 'Success Rate' in key_metrics:
                success_rate = key_metrics['Success Rate'] * 100 if key_metrics['Success Rate'] <= 1 else key_metrics['Success Rate']
                key_points.append(f"Overall success rate: {success_rate:.1f}%")
            
            # Engagement summary
            if 'Engagement Score' in key_metrics:
                engagement = key_metrics['Engagement Score'] * 100 if key_metrics['Engagement Score'] <= 1 else key_metrics['Engagement Score']
                key_points.append(f"Student engagement level: {engagement:.1f}%")
            
            # Activity summary
            if 'Total Interactions' in key_metrics and 'Active Users' in key_metrics:
                total_interactions = int(key_metrics['Total Interactions'])
                active_users = int(key_metrics['Active Users'])
                avg_interactions = total_interactions / active_users if active_users > 0 else 0
                key_points.append(f"Average {avg_interactions:.1f} interactions per active student")
            
            # Top insights
            if insights:
                high_priority_insights = [insight for insight in insights if insight.get('importance_level') in ['critical', 'high']]
                for insight in high_priority_insights[:2]:  # Top 2 high-priority insights
                    key_points.append(insight.get('summary', insight.get('title', 'Important finding identified')))
            
            # Audience-specific points
            if target_audience == 'administrator':
                if 'Active Users' in key_metrics:
                    key_points.append(f"Platform utilization: {int(key_metrics['Active Users'])} active students")
            elif target_audience == 'educator':
                if 'Average Response Time' in key_metrics:
                    response_time = key_metrics['Average Response Time'] / 1000  # Convert to seconds
                    key_points.append(f"Average problem-solving time: {response_time:.1f} seconds")
            
            return key_points[:5]  # Limit to 5 key points
            
        except Exception as e:
            logger.error(f"❌ Failed to generate key points: {e}")
            return ["Analytics data processed successfully"]
    
    def _generate_narrative(self, key_metrics: Dict[str, float], insights: List[Dict], 
                          target_audience: str) -> str:
        """Generate narrative text for summary"""
        try:
            paragraphs = []
            
            # Opening paragraph with overall assessment
            opening = self._generate_opening_paragraph(key_metrics, target_audience)
            paragraphs.append(opening)
            
            # Performance analysis paragraph
            performance_para = self._generate_performance_paragraph(key_metrics, target_audience)
            if performance_para:
                paragraphs.append(performance_para)
            
            # Insights paragraph
            if insights:
                insights_para = self._generate_insights_paragraph(insights, target_audience)
                if insights_para:
                    paragraphs.append(insights_para)
            
            # Recommendations paragraph
            recommendations_para = self._generate_recommendations_paragraph(key_metrics, insights, target_audience)
            if recommendations_para:
                paragraphs.append(recommendations_para)
            
            return ' '.join(paragraphs)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate narrative: {e}")
            return "Analytics data has been processed and is available for review."
    
    def _generate_opening_paragraph(self, key_metrics: Dict[str, float], target_audience: str) -> str:
        """Generate opening paragraph for narrative"""
        try:
            if 'Success Rate' in key_metrics and 'Active Users' in key_metrics:
                success_rate = key_metrics['Success Rate'] * 100 if key_metrics['Success Rate'] <= 1 else key_metrics['Success Rate']
                active_users = int(key_metrics['Active Users'])
                
                if target_audience == 'administrator':
                    return f"The learning platform shows {active_users} active students with an overall success rate of {success_rate:.1f}%. This data provides insights into current academic performance and student engagement levels."
                else:
                    return f"Student performance analysis reveals a {success_rate:.1f}% success rate across {active_users} active learners. The data indicates {'strong' if success_rate >= 75 else 'moderate' if success_rate >= 60 else 'concerning'} academic progress."
            
            return "Recent analytics data provides valuable insights into student learning patterns and academic performance."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate opening paragraph: {e}")
            return "Analytics summary is available for review."
    
    def _generate_performance_paragraph(self, key_metrics: Dict[str, float], target_audience: str) -> str:
        """Generate performance analysis paragraph"""
        try:
            if 'Engagement Score' in key_metrics and 'Completion Rate' in key_metrics:
                engagement = key_metrics['Engagement Score'] * 100 if key_metrics['Engagement Score'] <= 1 else key_metrics['Engagement Score']
                completion = key_metrics['Completion Rate'] * 100 if key_metrics['Completion Rate'] <= 1 else key_metrics['Completion Rate']
                
                engagement_desc = 'high' if engagement >= 75 else 'moderate' if engagement >= 50 else 'low'
                completion_desc = 'strong' if completion >= 80 else 'adequate' if completion >= 60 else 'concerning'
                
                return f"Student engagement levels are {engagement_desc} at {engagement:.1f}%, while task completion rates show {completion_desc} performance at {completion:.1f}%. These metrics suggest {'effective' if engagement >= 60 and completion >= 70 else 'mixed'} learning outcomes."
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance paragraph: {e}")
            return None
    
    def _generate_insights_paragraph(self, insights: List[Dict], target_audience: str) -> str:
        """Generate insights paragraph"""
        try:
            if not insights:
                return None
            
            # Focus on most important insights
            important_insights = [insight for insight in insights if insight.get('importance_level') in ['critical', 'high']]
            
            if important_insights:
                insight_summaries = [insight.get('summary', insight.get('title', '')) for insight in important_insights[:2]]
                insights_text = ' Additionally, '.join(insight_summaries)
                return f"Key findings indicate that {insights_text}. These patterns provide important guidance for instructional decisions."
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to generate insights paragraph: {e}")
            return None
    
    def _generate_recommendations_paragraph(self, key_metrics: Dict[str, float], 
                                          insights: List[Dict], target_audience: str) -> str:
        """Generate recommendations paragraph"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if 'Success Rate' in key_metrics:
                success_rate = key_metrics['Success Rate'] * 100 if key_metrics['Success Rate'] <= 1 else key_metrics['Success Rate']
                if success_rate < 60:
                    recommendations.append("consider additional support interventions")
                elif success_rate > 85:
                    recommendations.append("explore opportunities for advanced challenges")
            
            # Engagement-based recommendations
            if 'Engagement Score' in key_metrics:
                engagement = key_metrics['Engagement Score'] * 100 if key_metrics['Engagement Score'] <= 1 else key_metrics['Engagement Score']
                if engagement < 50:
                    recommendations.append("implement strategies to increase student engagement")
            
            # Insight-based recommendations
            if insights:
                for insight in insights[:2]:
                    if insight.get('actionable_recommendations'):
                        recommendations.extend(insight['actionable_recommendations'][:1])
            
            if recommendations:
                recommendations_text = ', '.join(recommendations[:3])
                return f"Based on these findings, it is recommended to {recommendations_text}. Continued monitoring will help track the effectiveness of any implemented changes."
            
            return "Regular monitoring of these metrics will provide ongoing insights into student progress and learning effectiveness."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendations paragraph: {e}")
            return "Continued observation of learning patterns is recommended."
    
    # Database operations and helper methods
    async def _create_insights_tables(self):
        """Create database tables for insights storage"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Automated insights table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS automated_insights (
                        insight_id VARCHAR(100) PRIMARY KEY,
                        insight_type VARCHAR(50) NOT NULL,
                        title VARCHAR(200) NOT NULL,
                        summary TEXT NOT NULL,
                        detailed_explanation TEXT,
                        confidence_score FLOAT DEFAULT 0.0,
                        importance_level VARCHAR(20) DEFAULT 'medium',
                        supporting_data JSONB DEFAULT '{}',
                        actionable_recommendations JSONB DEFAULT '[]',
                        affected_entities JSONB DEFAULT '[]',
                        start_date TIMESTAMP,
                        end_date TIMESTAMP,
                        generated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Natural language summaries table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS nl_summaries (
                        summary_id VARCHAR(100) PRIMARY KEY,
                        summary_type VARCHAR(50) NOT NULL,
                        target_audience VARCHAR(50) NOT NULL,
                        headline VARCHAR(300) NOT NULL,
                        key_points JSONB DEFAULT '[]',
                        narrative_text TEXT,
                        metrics_mentioned JSONB DEFAULT '[]',
                        reading_level FLOAT DEFAULT 0.0,
                        length_category VARCHAR(20) DEFAULT 'standard',
                        visualizations_referenced JSONB DEFAULT '[]',
                        generated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                logger.info("✅ Insights tables created")
        
        except Exception as e:
            logger.error(f"❌ Failed to create insights tables: {e}")
    
    async def _load_existing_insights(self):
        """Load existing insights from database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Load recent insights
                insights = await conn.fetch("""
                    SELECT * FROM automated_insights 
                    WHERE generated_at >= NOW() - INTERVAL '7 days'
                    ORDER BY generated_at DESC
                """)
                
                for insight_row in insights:
                    insight = AutomatedInsight(
                        insight_id=insight_row['insight_id'],
                        insight_type=insight_row['insight_type'],
                        title=insight_row['title'],
                        summary=insight_row['summary'],
                        detailed_explanation=insight_row['detailed_explanation'],
                        confidence_score=insight_row['confidence_score'],
                        importance_level=insight_row['importance_level'],
                        supporting_data=insight_row['supporting_data'],
                        actionable_recommendations=insight_row['actionable_recommendations'],
                        affected_entities=insight_row['affected_entities'],
                        time_frame=(insight_row['start_date'], insight_row['end_date']),
                        generated_at=insight_row['generated_at']
                    )
                    self.generated_insights[insight.insight_id] = insight
                
                logger.info(f"📊 Loaded {len(self.generated_insights)} recent insights")
        
        except Exception as e:
            logger.error(f"❌ Failed to load existing insights: {e}")
    
    async def _save_insight(self, insight: AutomatedInsight):
        """Save insight to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO automated_insights 
                    (insight_id, insight_type, title, summary, detailed_explanation,
                     confidence_score, importance_level, supporting_data,
                     actionable_recommendations, affected_entities, start_date, end_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (insight_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    detailed_explanation = EXCLUDED.detailed_explanation,
                    confidence_score = EXCLUDED.confidence_score,
                    importance_level = EXCLUDED.importance_level,
                    supporting_data = EXCLUDED.supporting_data,
                    actionable_recommendations = EXCLUDED.actionable_recommendations,
                    affected_entities = EXCLUDED.affected_entities
                """,
                insight.insight_id, insight.insight_type, insight.title,
                insight.summary, insight.detailed_explanation, insight.confidence_score,
                insight.importance_level, json.dumps(insight.supporting_data),
                json.dumps(insight.actionable_recommendations),
                json.dumps(insight.affected_entities),
                insight.time_frame[0], insight.time_frame[1])
        
        except Exception as e:
            logger.error(f"❌ Failed to save insight: {e}")
    
    async def _save_nl_summary(self, summary: NaturalLanguageSummary):
        """Save natural language summary to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO nl_summaries 
                    (summary_id, summary_type, target_audience, headline, key_points,
                     narrative_text, metrics_mentioned, reading_level, length_category,
                     visualizations_referenced)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (summary_id) DO UPDATE SET
                    headline = EXCLUDED.headline,
                    key_points = EXCLUDED.key_points,
                    narrative_text = EXCLUDED.narrative_text,
                    metrics_mentioned = EXCLUDED.metrics_mentioned,
                    reading_level = EXCLUDED.reading_level,
                    length_category = EXCLUDED.length_category,
                    visualizations_referenced = EXCLUDED.visualizations_referenced
                """,
                summary.summary_id, summary.summary_type, summary.target_audience,
                summary.headline, json.dumps(summary.key_points), summary.narrative_text,
                json.dumps(summary.metrics_mentioned), summary.reading_level,
                summary.length_category, json.dumps(summary.visualizations_referenced))
        
        except Exception as e:
            logger.error(f"❌ Failed to save natural language summary: {e}")

# Main testing function
async def test_automated_insights():
    """Test automated insights engine"""
    try:
        logger.info("🧪 Testing Automated Insights Engine")
        
        engine = AutomatedInsightsEngine()
        await engine.initialize()
        
        # Test insight generation with sample data
        sample_analytics_data = {
            'success_rate': [
                {'timestamp': '2024-01-01', 'value': 0.75},
                {'timestamp': '2024-01-02', 'value': 0.78},
                {'timestamp': '2024-01-03', 'value': 0.82},
                {'timestamp': '2024-01-04', 'value': 0.85}
            ],
            'engagement_score': 0.73,
            'total_interactions': 1250,
            'active_users': 45
        }
        
        insights = await engine.generate_insights(sample_analytics_data)
        logger.info(f"✅ Generated {len(insights)} insights")
        
        # Test natural language summary generation
        nl_summary = await engine.generate_natural_language_summary(
            sample_analytics_data, 'student_progress', 'educator'
        )
        logger.info(f"✅ Generated natural language summary: {nl_summary.headline}")
        
        logger.info("✅ Automated Insights Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Automated Insights test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_automated_insights())