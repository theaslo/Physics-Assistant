#!/usr/bin/env python3
"""
Multi-Timeframe Prediction System for Physics Assistant Phase 6.3
Implements predictions across multiple time horizons (short, medium, long-term)
with adaptive forecasting and temporal pattern analysis.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import uuid
from collections import defaultdict, deque
import math
import statistics
from scipy import stats, signal
import warnings

# Import related modules
from .predictive_analytics import PredictiveAnalyticsEngine, PredictionResult
from .time_to_mastery_predictor import TimeToMasteryPredictor, MasteryPrediction

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeHorizon(Enum):
    IMMEDIATE = "immediate"     # Next session (1-4 hours)
    SHORT_TERM = "short_term"   # Next 1-3 days
    MEDIUM_TERM = "medium_term" # Next 1-2 weeks
    LONG_TERM = "long_term"     # Next 1-3 months
    SEMESTER = "semester"       # Full semester/course

class PredictionType(Enum):
    PERFORMANCE = "performance"
    ENGAGEMENT = "engagement"
    MASTERY_PROGRESS = "mastery_progress"
    CONCEPT_COMPLETION = "concept_completion"
    LEARNING_VELOCITY = "learning_velocity"
    DIFFICULTY_ADAPTATION = "difficulty_adaptation"
    HELP_SEEKING_BEHAVIOR = "help_seeking_behavior"
    SESSION_DURATION = "session_duration"
    DROPOUT_RISK = "dropout_risk"
    ACHIEVEMENT_LIKELIHOOD = "achievement_likelihood"

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"     # 90%+ confidence
    HIGH = "high"               # 75-90% confidence
    MEDIUM = "medium"           # 60-75% confidence
    LOW = "low"                 # 40-60% confidence
    VERY_LOW = "very_low"       # <40% confidence

@dataclass
class TimeframePrediction:
    """Prediction for a specific timeframe"""
    prediction_id: str
    student_id: str
    prediction_type: PredictionType
    time_horizon: TimeHorizon
    target_date: datetime
    predicted_value: float
    confidence_level: ConfidenceLevel
    confidence_score: float
    confidence_interval: Tuple[float, float]
    contributing_factors: Dict[str, float]
    uncertainty_sources: List[str]
    trend_analysis: Dict[str, float]
    seasonal_patterns: Dict[str, float]
    risk_factors: List[str]
    opportunities: List[str]
    context_metadata: Dict[str, Any]
    model_version: str
    prediction_date: datetime = field(default_factory=datetime.now)

@dataclass
class MultiTimeframeForecast:
    """Comprehensive multi-timeframe forecast"""
    forecast_id: str
    student_id: str
    prediction_type: PredictionType
    timeframe_predictions: Dict[TimeHorizon, TimeframePrediction]
    trend_trajectory: List[Tuple[datetime, float]]  # Predicted values over time
    inflection_points: List[Tuple[datetime, str]]   # Key transition points
    scenario_analysis: Dict[str, List[float]]       # Best/worst/expected case
    adaptive_recommendations: Dict[TimeHorizon, List[str]]
    monitoring_alerts: List[Dict[str, Any]]
    forecast_accuracy_history: List[float]
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class TemporalPattern:
    """Identified temporal patterns in student behavior"""
    pattern_id: str
    pattern_type: str  # 'weekly', 'daily', 'seasonal', 'trend'
    strength: float    # 0-1, how consistent the pattern is
    frequency: float   # Cycles per unit time
    amplitude: float   # Magnitude of variation
    phase_shift: float # Time offset
    confidence: float
    detected_periods: List[Tuple[datetime, datetime]]
    pattern_description: str

class TemporalPatternAnalyzer:
    """Analyzes temporal patterns in student learning data"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.min_data_points = 10
        self.pattern_types = [
            'daily_cycle',      # Daily activity patterns
            'weekly_cycle',     # Weekly study patterns
            'performance_trend', # Learning progress trends
            'engagement_cycle', # Engagement fluctuations
            'difficulty_adaptation', # Response to difficulty changes
            'fatigue_pattern',  # Learning fatigue cycles
            'motivation_wave',  # Motivation fluctuations
            'session_rhythm'    # Study session patterns
        ]
    
    async def analyze_patterns(self, student_id: str, 
                             interaction_data: pd.DataFrame,
                             lookback_days: int = 60) -> List[TemporalPattern]:
        """Analyze temporal patterns in student data"""
        try:
            patterns = []
            
            if len(interaction_data) < self.min_data_points:
                return patterns
            
            # Prepare time series data
            df = interaction_data.copy()
            df['timestamp'] = pd.to_datetime(df['created_at'])
            df = df.sort_values('timestamp')
            
            # Daily patterns
            daily_patterns = await self._analyze_daily_patterns(df)
            patterns.extend(daily_patterns)
            
            # Weekly patterns
            weekly_patterns = await self._analyze_weekly_patterns(df)
            patterns.extend(weekly_patterns)
            
            # Performance trends
            performance_trends = await self._analyze_performance_trends(df)
            patterns.extend(performance_trends)
            
            # Engagement cycles
            engagement_cycles = await self._analyze_engagement_cycles(df)
            patterns.extend(engagement_cycles)
            
            # Session patterns
            session_patterns = await self._analyze_session_patterns(df)
            patterns.extend(session_patterns)
            
            # Cache results
            self.pattern_cache[student_id] = {
                'patterns': patterns,
                'last_updated': datetime.now()
            }
            
            logger.info(f"🔍 Identified {len(patterns)} temporal patterns for student {student_id}")
            return patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze temporal patterns: {e}")
            return []
    
    async def _analyze_daily_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze daily activity patterns"""
        patterns = []
        
        try:
            # Extract hourly activity
            df['hour'] = df['timestamp'].dt.hour
            hourly_activity = df.groupby('hour').size()
            
            if len(hourly_activity) > 4:  # Need sufficient hours
                # Find peak activity times
                hours = hourly_activity.index.values
                activity = hourly_activity.values
                
                # Use FFT to detect daily cycles
                if len(activity) >= 8:
                    fft = np.fft.fft(activity)
                    frequencies = np.fft.fftfreq(len(activity))
                    
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    dominant_freq = frequencies[dominant_freq_idx]
                    
                    if dominant_freq > 0:
                        cycle_strength = np.abs(fft[dominant_freq_idx]) / np.sum(np.abs(fft))
                        
                        if cycle_strength > 0.3:  # Significant pattern
                            # Find peak hour
                            peak_hour = hours[np.argmax(activity)]
                            
                            pattern = TemporalPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type='daily_cycle',
                                strength=cycle_strength,
                                frequency=24.0,  # 24-hour cycle
                                amplitude=np.std(activity),
                                phase_shift=peak_hour,
                                confidence=min(0.9, cycle_strength * 2),
                                detected_periods=[],
                                pattern_description=f"Peak activity at {peak_hour}:00, cycle strength {cycle_strength:.2f}"
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze daily patterns: {e}")
        
        return patterns
    
    async def _analyze_weekly_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze weekly study patterns"""
        patterns = []
        
        try:
            # Extract day of week activity
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            daily_activity = df.groupby('day_of_week').size()
            
            if len(daily_activity) >= 5:  # Need most days of week
                days = daily_activity.index.values
                activity = daily_activity.values
                
                # Calculate weekly pattern strength
                activity_normalized = activity / np.sum(activity)
                uniform_distribution = np.ones(len(activity)) / len(activity)
                
                # Use KL divergence to measure deviation from uniform
                kl_divergence = stats.entropy(activity_normalized, uniform_distribution)
                pattern_strength = min(1.0, kl_divergence / 2.0)
                
                if pattern_strength > 0.2:
                    # Find peak days
                    peak_day = days[np.argmax(activity)]
                    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    
                    pattern = TemporalPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type='weekly_cycle',
                        strength=pattern_strength,
                        frequency=7.0,  # 7-day cycle
                        amplitude=np.std(activity),
                        phase_shift=peak_day,
                        confidence=min(0.8, pattern_strength * 1.5),
                        detected_periods=[],
                        pattern_description=f"Peak activity on {day_names[peak_day]}, pattern strength {pattern_strength:.2f}"
                    )
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze weekly patterns: {e}")
        
        return patterns
    
    async def _analyze_performance_trends(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze performance trend patterns"""
        patterns = []
        
        try:
            if 'success' in df.columns and len(df) >= 20:
                # Calculate rolling performance
                df = df.sort_values('timestamp')
                window_size = max(5, len(df) // 10)
                df['rolling_performance'] = df['success'].rolling(window=window_size, center=True).mean()
                
                # Remove NaN values
                performance_data = df['rolling_performance'].dropna()
                
                if len(performance_data) >= 10:
                    # Detect trend using linear regression
                    x = np.arange(len(performance_data))
                    y = performance_data.values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Significant trend if p < 0.05 and |r| > 0.3
                    if p_value < 0.05 and abs(r_value) > 0.3:
                        trend_strength = abs(r_value)
                        trend_direction = 'improving' if slope > 0 else 'declining'
                        
                        pattern = TemporalPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type='performance_trend',
                            strength=trend_strength,
                            frequency=0.0,  # Not cyclical
                            amplitude=abs(slope),
                            phase_shift=0.0,
                            confidence=1.0 - p_value,
                            detected_periods=[],
                            pattern_description=f"{trend_direction.title()} performance trend (r={r_value:.3f})"
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze performance trends: {e}")
        
        return patterns
    
    async def _analyze_engagement_cycles(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze engagement fluctuation patterns"""
        patterns = []
        
        try:
            # Create engagement metric from session frequency and duration
            df = df.sort_values('timestamp')
            df['date'] = df['timestamp'].dt.date
            
            daily_engagement = df.groupby('date').agg({
                'timestamp': 'count',  # Number of interactions
                'execution_time_ms': 'mean'  # Average response time
            }).rename(columns={'timestamp': 'interaction_count'})
            
            if len(daily_engagement) >= 14:  # Need at least 2 weeks
                # Create engagement score
                daily_engagement['engagement_score'] = (
                    daily_engagement['interaction_count'] / daily_engagement['interaction_count'].max()
                ) * (
                    1.0 - (daily_engagement['execution_time_ms'] - daily_engagement['execution_time_ms'].min()) / 
                    (daily_engagement['execution_time_ms'].max() - daily_engagement['execution_time_ms'].min() + 1)
                )
                
                engagement_values = daily_engagement['engagement_score'].values
                
                # Detect cycles using autocorrelation
                autocorr = np.correlate(engagement_values, engagement_values, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Find peaks in autocorrelation (excluding lag 0)
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.3, distance=3)
                
                if len(peaks) > 0:
                    # Most significant cycle
                    main_peak = peaks[np.argmax(autocorr[peaks + 1])] + 1
                    cycle_strength = autocorr[main_peak]
                    
                    if cycle_strength > 0.4:
                        pattern = TemporalPattern(
                            pattern_id=str(uuid.uuid4()),
                            pattern_type='engagement_cycle',
                            strength=cycle_strength,
                            frequency=main_peak,  # Period in days
                            amplitude=np.std(engagement_values),
                            phase_shift=0.0,
                            confidence=cycle_strength,
                            detected_periods=[],
                            pattern_description=f"Engagement cycle every {main_peak} days (strength {cycle_strength:.2f})"
                        )
                        patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze engagement cycles: {e}")
        
        return patterns
    
    async def _analyze_session_patterns(self, df: pd.DataFrame) -> List[TemporalPattern]:
        """Analyze study session timing patterns"""
        patterns = []
        
        try:
            if len(df) >= 20:
                # Identify sessions based on time gaps
                df = df.sort_values('timestamp')
                df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60  # Minutes
                
                # New session if gap > 30 minutes
                df['session_id'] = (df['time_diff'] > 30).cumsum()
                
                sessions = df.groupby('session_id').agg({
                    'timestamp': ['min', 'max', 'count'],
                    'success': 'mean'
                })
                
                sessions.columns = ['start_time', 'end_time', 'interaction_count', 'avg_success']
                sessions['duration_minutes'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds() / 60
                
                if len(sessions) >= 10:
                    # Analyze session duration patterns
                    durations = sessions['duration_minutes'].values
                    
                    # Find optimal session length
                    success_by_duration = []
                    duration_bins = np.linspace(durations.min(), durations.max(), 5)
                    
                    for i in range(len(duration_bins) - 1):
                        mask = (durations >= duration_bins[i]) & (durations < duration_bins[i + 1])
                        if np.sum(mask) > 0:
                            avg_success = sessions[mask]['avg_success'].mean()
                            success_by_duration.append(avg_success)
                    
                    if len(success_by_duration) >= 3:
                        optimal_duration_idx = np.argmax(success_by_duration)
                        optimal_duration = (duration_bins[optimal_duration_idx] + duration_bins[optimal_duration_idx + 1]) / 2
                        
                        pattern_strength = (max(success_by_duration) - min(success_by_duration)) / max(success_by_duration)
                        
                        if pattern_strength > 0.1:
                            pattern = TemporalPattern(
                                pattern_id=str(uuid.uuid4()),
                                pattern_type='session_rhythm',
                                strength=pattern_strength,
                                frequency=optimal_duration,
                                amplitude=np.std(success_by_duration),
                                phase_shift=0.0,
                                confidence=min(0.8, pattern_strength * 3),
                                detected_periods=[],
                                pattern_description=f"Optimal session length: {optimal_duration:.1f} minutes"
                            )
                            patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze session patterns: {e}")
        
        return patterns

class MultiTimeframeNeuralNetwork(nn.Module):
    """Neural network for multi-timeframe predictions"""
    
    def __init__(self, input_dim: int, num_timeframes: int = 5, hidden_dims: List[int] = [128, 64, 32]):
        super(MultiTimeframeNeuralNetwork, self).__init__()
        
        self.num_timeframes = num_timeframes
        
        # Shared feature extraction
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Timeframe-specific prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_timeframes)
        ])
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus()
            ) for _ in range(num_timeframes)
        ])
        
        # Temporal consistency layer
        self.temporal_consistency = nn.Sequential(
            nn.Linear(num_timeframes, num_timeframes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Get predictions for each timeframe
        predictions = []
        uncertainties = []
        
        for i in range(self.num_timeframes):
            pred = self.prediction_heads[i](features)
            uncertainty = self.uncertainty_heads[i](features)
            predictions.append(pred)
            uncertainties.append(uncertainty)
        
        predictions = torch.cat(predictions, dim=1)
        uncertainties = torch.cat(uncertainties, dim=1)
        
        # Apply temporal consistency
        consistency_weights = self.temporal_consistency(predictions)
        predictions = predictions * consistency_weights
        
        return predictions, uncertainties

class MultiTimeframePredictor:
    """Advanced multi-timeframe prediction system"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Components
        self.pattern_analyzer = TemporalPatternAnalyzer()
        self.neural_models = {}
        self.scalers = {}
        
        # Time horizons configuration
        self.time_horizons = {
            TimeHorizon.IMMEDIATE: {'hours': 4, 'confidence_decay': 0.95},
            TimeHorizon.SHORT_TERM: {'hours': 72, 'confidence_decay': 0.9},
            TimeHorizon.MEDIUM_TERM: {'hours': 336, 'confidence_decay': 0.8},  # 2 weeks
            TimeHorizon.LONG_TERM: {'hours': 2160, 'confidence_decay': 0.6},   # 3 months
            TimeHorizon.SEMESTER: {'hours': 4320, 'confidence_decay': 0.4}     # 6 months
        }
        
        # Prediction tracking
        self.student_forecasts: Dict[str, Dict[PredictionType, MultiTimeframeForecast]] = defaultdict(dict)
        self.prediction_history: Dict[str, List[TimeframePrediction]] = defaultdict(list)
        
        # Model configuration
        self.config = {
            'min_historical_data_points': 20,
            'pattern_detection_threshold': 0.3,
            'prediction_update_frequency': 3600,  # 1 hour
            'forecast_horizon_days': 180,
            'confidence_threshold': 0.6,
            'temporal_weight_decay': 0.95
        }
    
    async def initialize(self):
        """Initialize the multi-timeframe prediction system"""
        try:
            logger.info("🚀 Initializing Multi-Timeframe Prediction System")
            
            # Initialize neural networks for each prediction type
            input_dim = 40  # Comprehensive feature set including temporal patterns
            
            for pred_type in PredictionType:
                self.neural_models[pred_type] = MultiTimeframeNeuralNetwork(
                    input_dim=input_dim,
                    num_timeframes=len(self.time_horizons)
                )
                
                # Initialize scaler for this prediction type
                self.scalers[pred_type] = StandardScaler()
            
            logger.info("✅ Multi-Timeframe Prediction System initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Multi-Timeframe Prediction System: {e}")
            return False
    
    async def generate_multi_timeframe_forecast(self, student_id: str, 
                                              prediction_type: PredictionType) -> MultiTimeframeForecast:
        """Generate comprehensive multi-timeframe forecast"""
        try:
            logger.info(f"🔮 Generating multi-timeframe forecast for student {student_id}, type {prediction_type.value}")
            
            # Get historical data
            historical_data = await self._get_historical_data(student_id)
            
            if len(historical_data) < self.config['min_historical_data_points']:
                return self._create_minimal_forecast(student_id, prediction_type)
            
            # Analyze temporal patterns
            patterns = await self.pattern_analyzer.analyze_patterns(student_id, historical_data)
            
            # Extract comprehensive features
            features = await self._extract_temporal_features(student_id, historical_data, patterns)
            
            # Generate predictions for each timeframe
            timeframe_predictions = {}
            for horizon in TimeHorizon:
                prediction = await self._predict_for_timeframe(
                    student_id, prediction_type, horizon, features, patterns
                )
                timeframe_predictions[horizon] = prediction
            
            # Generate trajectory
            trajectory = await self._generate_prediction_trajectory(
                student_id, prediction_type, timeframe_predictions, patterns
            )
            
            # Identify inflection points
            inflection_points = await self._identify_inflection_points(trajectory, patterns)
            
            # Scenario analysis
            scenario_analysis = await self._generate_scenario_analysis(
                timeframe_predictions, patterns, features
            )
            
            # Adaptive recommendations
            recommendations = await self._generate_adaptive_recommendations(
                student_id, prediction_type, timeframe_predictions, patterns
            )
            
            # Monitoring alerts
            alerts = await self._generate_monitoring_alerts(
                student_id, prediction_type, timeframe_predictions
            )
            
            # Create forecast
            forecast = MultiTimeframeForecast(
                forecast_id=str(uuid.uuid4()),
                student_id=student_id,
                prediction_type=prediction_type,
                timeframe_predictions=timeframe_predictions,
                trend_trajectory=trajectory,
                inflection_points=inflection_points,
                scenario_analysis=scenario_analysis,
                adaptive_recommendations=recommendations,
                monitoring_alerts=alerts,
                forecast_accuracy_history=[]
            )
            
            # Store forecast
            self.student_forecasts[student_id][prediction_type] = forecast
            
            logger.info(f"✅ Generated forecast with {len(timeframe_predictions)} timeframes")
            return forecast
            
        except Exception as e:
            logger.error(f"❌ Failed to generate multi-timeframe forecast: {e}")
            return self._create_minimal_forecast(student_id, prediction_type)
    
    async def _predict_for_timeframe(self, student_id: str, prediction_type: PredictionType,
                                   horizon: TimeHorizon, features: Dict[str, float],
                                   patterns: List[TemporalPattern]) -> TimeframePrediction:
        """Generate prediction for specific timeframe"""
        try:
            # Calculate target date
            hours_ahead = self.time_horizons[horizon]['hours']
            target_date = datetime.now() + timedelta(hours=hours_ahead)
            
            # Get base prediction
            base_prediction = await self._calculate_base_prediction(
                student_id, prediction_type, features, hours_ahead
            )
            
            # Apply temporal patterns
            pattern_adjusted_prediction = await self._apply_temporal_patterns(
                base_prediction, patterns, hours_ahead, prediction_type
            )
            
            # Calculate confidence
            confidence_score, confidence_level = await self._calculate_timeframe_confidence(
                horizon, features, patterns, prediction_type
            )
            
            # Calculate confidence interval
            confidence_interval = await self._calculate_confidence_interval(
                pattern_adjusted_prediction, confidence_score, horizon
            )
            
            # Identify contributing factors
            contributing_factors = await self._identify_contributing_factors(
                features, patterns, prediction_type, horizon
            )
            
            # Identify uncertainty sources
            uncertainty_sources = await self._identify_uncertainty_sources(
                horizon, patterns, features
            )
            
            # Trend analysis
            trend_analysis = await self._analyze_trends_for_timeframe(
                features, patterns, horizon
            )
            
            # Seasonal patterns
            seasonal_patterns = await self._extract_seasonal_patterns(
                patterns, horizon
            )
            
            # Risk factors and opportunities
            risk_factors, opportunities = await self._identify_risks_and_opportunities(
                student_id, prediction_type, horizon, features, patterns
            )
            
            return TimeframePrediction(
                prediction_id=str(uuid.uuid4()),
                student_id=student_id,
                prediction_type=prediction_type,
                time_horizon=horizon,
                target_date=target_date,
                predicted_value=pattern_adjusted_prediction,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                confidence_interval=confidence_interval,
                contributing_factors=contributing_factors,
                uncertainty_sources=uncertainty_sources,
                trend_analysis=trend_analysis,
                seasonal_patterns=seasonal_patterns,
                risk_factors=risk_factors,
                opportunities=opportunities,
                context_metadata={
                    'patterns_used': len(patterns),
                    'feature_count': len(features),
                    'hours_ahead': hours_ahead
                },
                model_version='1.0'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to predict for timeframe {horizon.value}: {e}")
            return self._create_default_timeframe_prediction(student_id, prediction_type, horizon)
    
    async def _calculate_base_prediction(self, student_id: str, prediction_type: PredictionType,
                                       features: Dict[str, float], hours_ahead: float) -> float:
        """Calculate base prediction without temporal adjustments"""
        try:
            # Simple heuristic-based prediction (would use trained models in production)
            current_value = features.get('current_performance', 0.5)
            trend = features.get('performance_trend', 0.0)
            volatility = features.get('performance_volatility', 0.1)
            
            # Apply trend over time
            time_factor = hours_ahead / 168.0  # Normalize to weeks
            trend_effect = trend * time_factor
            
            # Add some decay for longer predictions
            decay_factor = math.exp(-time_factor * 0.1)
            
            base_prediction = current_value + trend_effect * decay_factor
            
            # Add prediction type specific adjustments
            if prediction_type == PredictionType.PERFORMANCE:
                # Performance tends to be more stable
                base_prediction = base_prediction * 0.9 + current_value * 0.1
            elif prediction_type == PredictionType.ENGAGEMENT:
                # Engagement can be more volatile
                volatility_effect = np.random.normal(0, volatility * 0.1)
                base_prediction += volatility_effect
            elif prediction_type == PredictionType.DROPOUT_RISK:
                # Dropout risk increases over time if performance is poor
                if current_value < 0.5:
                    base_prediction += time_factor * 0.1
            
            return max(0.0, min(1.0, base_prediction))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate base prediction: {e}")
            return 0.5
    
    async def _apply_temporal_patterns(self, base_prediction: float, patterns: List[TemporalPattern],
                                     hours_ahead: float, prediction_type: PredictionType) -> float:
        """Apply identified temporal patterns to base prediction"""
        try:
            adjusted_prediction = base_prediction
            
            for pattern in patterns:
                if pattern.confidence < 0.5:
                    continue
                
                # Apply pattern based on its type and the prediction timeframe
                if pattern.pattern_type == 'daily_cycle' and hours_ahead <= 72:
                    # Daily patterns matter for short-term predictions
                    target_hour = (datetime.now() + timedelta(hours=hours_ahead)).hour
                    phase_difference = abs(target_hour - pattern.phase_shift) / 12.0  # Normalize
                    cycle_effect = pattern.amplitude * math.cos(2 * math.pi * phase_difference) * pattern.strength
                    adjusted_prediction += cycle_effect * 0.1
                
                elif pattern.pattern_type == 'weekly_cycle' and hours_ahead <= 336:
                    # Weekly patterns matter for medium-term predictions
                    target_day = (datetime.now() + timedelta(hours=hours_ahead)).weekday()
                    phase_difference = abs(target_day - pattern.phase_shift) / 3.5  # Normalize
                    cycle_effect = pattern.amplitude * math.cos(2 * math.pi * phase_difference) * pattern.strength
                    adjusted_prediction += cycle_effect * 0.15
                
                elif pattern.pattern_type == 'performance_trend':
                    # Trends apply across all timeframes with decay
                    time_weeks = hours_ahead / 168.0
                    trend_decay = math.exp(-time_weeks * 0.2)
                    trend_effect = pattern.amplitude * pattern.strength * trend_decay
                    if 'declining' in pattern.pattern_description:
                        adjusted_prediction -= trend_effect
                    else:
                        adjusted_prediction += trend_effect
                
                elif pattern.pattern_type == 'engagement_cycle' and prediction_type == PredictionType.ENGAGEMENT:
                    # Engagement cycles affect engagement predictions
                    days_ahead = hours_ahead / 24.0
                    cycle_position = (days_ahead % pattern.frequency) / pattern.frequency
                    cycle_effect = pattern.amplitude * math.sin(2 * math.pi * cycle_position) * pattern.strength
                    adjusted_prediction += cycle_effect * 0.2
            
            return max(0.0, min(1.0, adjusted_prediction))
            
        except Exception as e:
            logger.error(f"❌ Failed to apply temporal patterns: {e}")
            return base_prediction
    
    async def _calculate_timeframe_confidence(self, horizon: TimeHorizon, features: Dict[str, float],
                                            patterns: List[TemporalPattern], 
                                            prediction_type: PredictionType) -> Tuple[float, ConfidenceLevel]:
        """Calculate confidence for specific timeframe"""
        try:
            # Base confidence from horizon
            base_confidence = self.time_horizons[horizon]['confidence_decay']
            
            # Adjust based on data quality
            data_quality_factors = [
                min(1.0, features.get('historical_data_points', 10) / 50.0),
                features.get('pattern_consistency', 0.5),
                features.get('recent_activity_level', 0.5)
            ]
            data_quality_score = np.mean(data_quality_factors)
            
            # Adjust based on pattern strength
            pattern_confidence = 0.5
            if patterns:
                strong_patterns = [p for p in patterns if p.confidence > 0.6]
                if strong_patterns:
                    pattern_confidence = np.mean([p.confidence for p in strong_patterns])
            
            # Combine factors
            final_confidence = base_confidence * data_quality_score * pattern_confidence
            
            # Determine confidence level
            if final_confidence >= 0.9:
                level = ConfidenceLevel.VERY_HIGH
            elif final_confidence >= 0.75:
                level = ConfidenceLevel.HIGH
            elif final_confidence >= 0.6:
                level = ConfidenceLevel.MEDIUM
            elif final_confidence >= 0.4:
                level = ConfidenceLevel.LOW
            else:
                level = ConfidenceLevel.VERY_LOW
            
            return final_confidence, level
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate timeframe confidence: {e}")
            return 0.5, ConfidenceLevel.MEDIUM
    
    async def _calculate_confidence_interval(self, prediction: float, confidence: float,
                                           horizon: TimeHorizon) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        try:
            # Base margin increases with time horizon
            horizon_factors = {
                TimeHorizon.IMMEDIATE: 0.05,
                TimeHorizon.SHORT_TERM: 0.1,
                TimeHorizon.MEDIUM_TERM: 0.15,
                TimeHorizon.LONG_TERM: 0.25,
                TimeHorizon.SEMESTER: 0.35
            }
            
            base_margin = horizon_factors.get(horizon, 0.2)
            
            # Adjust margin based on confidence
            margin = base_margin * (1.0 - confidence) + base_margin * 0.5
            
            lower_bound = max(0.0, prediction - margin)
            upper_bound = min(1.0, prediction + margin)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence interval: {e}")
            return (prediction * 0.8, prediction * 1.2)
    
    async def _generate_prediction_trajectory(self, student_id: str, prediction_type: PredictionType,
                                            timeframe_predictions: Dict[TimeHorizon, TimeframePrediction],
                                            patterns: List[TemporalPattern]) -> List[Tuple[datetime, float]]:
        """Generate detailed prediction trajectory over time"""
        try:
            trajectory = []
            
            # Start with current time and current value
            current_time = datetime.now()
            current_value = timeframe_predictions[TimeHorizon.IMMEDIATE].predicted_value
            trajectory.append((current_time, current_value))
            
            # Add intermediate points based on timeframe predictions
            for horizon in [TimeHorizon.SHORT_TERM, TimeHorizon.MEDIUM_TERM, TimeHorizon.LONG_TERM, TimeHorizon.SEMESTER]:
                if horizon in timeframe_predictions:
                    pred = timeframe_predictions[horizon]
                    trajectory.append((pred.target_date, pred.predicted_value))
            
            # Interpolate additional points for smooth trajectory
            interpolated_trajectory = []
            for i in range(len(trajectory) - 1):
                start_time, start_value = trajectory[i]
                end_time, end_value = trajectory[i + 1]
                
                # Add intermediate points
                time_diff = end_time - start_time
                num_points = max(2, int(time_diff.total_seconds() / 86400))  # One point per day
                
                for j in range(num_points + 1):
                    interp_time = start_time + timedelta(seconds=time_diff.total_seconds() * j / num_points)
                    interp_value = start_value + (end_value - start_value) * j / num_points
                    
                    # Apply pattern effects for more realistic trajectory
                    for pattern in patterns:
                        if pattern.pattern_type in ['daily_cycle', 'weekly_cycle', 'engagement_cycle']:
                            hours_from_now = (interp_time - current_time).total_seconds() / 3600
                            pattern_effect = self._calculate_pattern_effect(pattern, hours_from_now)
                            interp_value += pattern_effect * 0.05  # Small effect for smoothness
                    
                    interpolated_trajectory.append((interp_time, max(0.0, min(1.0, interp_value))))
            
            return interpolated_trajectory
            
        except Exception as e:
            logger.error(f"❌ Failed to generate prediction trajectory: {e}")
            return [(datetime.now(), 0.5)]
    
    def _calculate_pattern_effect(self, pattern: TemporalPattern, hours_ahead: float) -> float:
        """Calculate pattern effect at specific time point"""
        try:
            if pattern.pattern_type == 'daily_cycle':
                cycle_position = (hours_ahead % 24) / 24
                return pattern.amplitude * math.sin(2 * math.pi * cycle_position) * pattern.strength
            
            elif pattern.pattern_type == 'weekly_cycle':
                cycle_position = (hours_ahead % 168) / 168
                return pattern.amplitude * math.sin(2 * math.pi * cycle_position) * pattern.strength
            
            elif pattern.pattern_type == 'engagement_cycle':
                cycle_position = (hours_ahead / 24) % pattern.frequency / pattern.frequency
                return pattern.amplitude * math.sin(2 * math.pi * cycle_position) * pattern.strength
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    async def _identify_inflection_points(self, trajectory: List[Tuple[datetime, float]],
                                        patterns: List[TemporalPattern]) -> List[Tuple[datetime, str]]:
        """Identify key inflection points in the trajectory"""
        inflection_points = []
        
        try:
            if len(trajectory) < 3:
                return inflection_points
            
            # Find points where trend changes significantly
            values = [point[1] for point in trajectory]
            times = [point[0] for point in trajectory]
            
            # Calculate first and second derivatives
            first_deriv = np.gradient(values)
            second_deriv = np.gradient(first_deriv)
            
            # Find inflection points where second derivative changes sign
            for i in range(1, len(second_deriv) - 1):
                if second_deriv[i-1] * second_deriv[i+1] < 0:  # Sign change
                    if abs(second_deriv[i]) > 0.01:  # Significant change
                        if first_deriv[i] > 0:
                            inflection_type = "acceleration_point"
                        else:
                            inflection_type = "deceleration_point"
                        
                        inflection_points.append((times[i], inflection_type))
            
            # Add pattern-based inflection points
            for pattern in patterns:
                if pattern.pattern_type == 'performance_trend' and pattern.strength > 0.6:
                    # Trend reversal points
                    if 'declining' in pattern.pattern_description:
                        # Predict when decline might stop
                        reversal_time = datetime.now() + timedelta(days=30)
                        inflection_points.append((reversal_time, "potential_trend_reversal"))
            
            return inflection_points
            
        except Exception as e:
            logger.error(f"❌ Failed to identify inflection points: {e}")
            return []
    
    async def _generate_scenario_analysis(self, timeframe_predictions: Dict[TimeHorizon, TimeframePrediction],
                                        patterns: List[TemporalPattern],
                                        features: Dict[str, float]) -> Dict[str, List[float]]:
        """Generate best/worst/expected case scenarios"""
        try:
            scenarios = {
                'best_case': [],
                'expected_case': [],
                'worst_case': []
            }
            
            for horizon in TimeHorizon:
                if horizon in timeframe_predictions:
                    pred = timeframe_predictions[horizon]
                    expected = pred.predicted_value
                    
                    # Calculate best and worst case based on confidence interval
                    confidence_range = pred.confidence_interval[1] - pred.confidence_interval[0]
                    
                    # Best case: upper confidence bound + optimistic factors
                    optimistic_boost = 0.1 * pred.confidence_score  # Higher confidence = more potential
                    best_case = min(1.0, pred.confidence_interval[1] + optimistic_boost)
                    
                    # Worst case: lower confidence bound - pessimistic factors
                    pessimistic_penalty = 0.1 * (1.0 - pred.confidence_score)
                    worst_case = max(0.0, pred.confidence_interval[0] - pessimistic_penalty)
                    
                    scenarios['expected_case'].append(expected)
                    scenarios['best_case'].append(best_case)
                    scenarios['worst_case'].append(worst_case)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"❌ Failed to generate scenario analysis: {e}")
            return {'expected_case': [0.5], 'best_case': [0.7], 'worst_case': [0.3]}
    
    # Additional helper methods would be implemented here...
    
    async def _get_historical_data(self, student_id: str) -> pd.DataFrame:
        """Get historical interaction data for student"""
        try:
            if not self.db_manager:
                # Return empty DataFrame for testing
                return pd.DataFrame()
            
            async with self.db_manager.postgres.get_connection() as conn:
                interactions = await conn.fetch("""
                    SELECT * FROM interactions 
                    WHERE user_id = $1 
                    AND created_at >= $2
                    ORDER BY created_at ASC
                """, student_id, datetime.now() - timedelta(days=90))
                
                return pd.DataFrame([dict(row) for row in interactions])
                
        except Exception as e:
            logger.error(f"❌ Failed to get historical data: {e}")
            return pd.DataFrame()
    
    async def _extract_temporal_features(self, student_id: str, historical_data: pd.DataFrame,
                                       patterns: List[TemporalPattern]) -> Dict[str, float]:
        """Extract comprehensive temporal features"""
        features = {
            'historical_data_points': len(historical_data),
            'pattern_consistency': np.mean([p.confidence for p in patterns]) if patterns else 0.5,
            'recent_activity_level': 0.5,
            'current_performance': 0.5,
            'performance_trend': 0.0,
            'performance_volatility': 0.1
        }
        
        try:
            if len(historical_data) > 0:
                # Calculate recent activity
                recent_data = historical_data[historical_data['created_at'] >= datetime.now() - timedelta(days=7)]
                features['recent_activity_level'] = min(1.0, len(recent_data) / 20.0)
                
                # Calculate performance metrics
                if 'success' in historical_data.columns:
                    success_values = historical_data['success'].astype(int)
                    features['current_performance'] = success_values.tail(10).mean() if len(success_values) >= 10 else success_values.mean()
                    
                    # Performance trend
                    if len(success_values) >= 20:
                        recent = success_values.tail(10).mean()
                        earlier = success_values.head(10).mean()
                        features['performance_trend'] = recent - earlier
                    
                    # Performance volatility
                    if len(success_values) >= 10:
                        rolling_performance = success_values.rolling(window=5).mean()
                        features['performance_volatility'] = rolling_performance.std()
            
            # Add pattern-derived features
            for pattern in patterns:
                feature_name = f"pattern_{pattern.pattern_type}_strength"
                features[feature_name] = pattern.strength
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Failed to extract temporal features: {e}")
            return features
    
    def _create_minimal_forecast(self, student_id: str, prediction_type: PredictionType) -> MultiTimeframeForecast:
        """Create minimal forecast when insufficient data"""
        timeframe_predictions = {}
        
        for horizon in TimeHorizon:
            pred = self._create_default_timeframe_prediction(student_id, prediction_type, horizon)
            timeframe_predictions[horizon] = pred
        
        return MultiTimeframeForecast(
            forecast_id=str(uuid.uuid4()),
            student_id=student_id,
            prediction_type=prediction_type,
            timeframe_predictions=timeframe_predictions,
            trend_trajectory=[(datetime.now(), 0.5)],
            inflection_points=[],
            scenario_analysis={'expected_case': [0.5], 'best_case': [0.7], 'worst_case': [0.3]},
            adaptive_recommendations={horizon: ["Insufficient data for detailed recommendations"] for horizon in TimeHorizon},
            monitoring_alerts=[],
            forecast_accuracy_history=[]
        )
    
    def _create_default_timeframe_prediction(self, student_id: str, prediction_type: PredictionType,
                                           horizon: TimeHorizon) -> TimeframePrediction:
        """Create default prediction for timeframe"""
        hours_ahead = self.time_horizons[horizon]['hours']
        target_date = datetime.now() + timedelta(hours=hours_ahead)
        
        return TimeframePrediction(
            prediction_id=str(uuid.uuid4()),
            student_id=student_id,
            prediction_type=prediction_type,
            time_horizon=horizon,
            target_date=target_date,
            predicted_value=0.5,
            confidence_level=ConfidenceLevel.LOW,
            confidence_score=0.3,
            confidence_interval=(0.3, 0.7),
            contributing_factors={'insufficient_data': 1.0},
            uncertainty_sources=['limited_historical_data', 'no_temporal_patterns'],
            trend_analysis={'trend_direction': 'stable', 'trend_strength': 0.0},
            seasonal_patterns={},
            risk_factors=['prediction_uncertainty'],
            opportunities=[],
            context_metadata={'data_points': 0, 'patterns_detected': 0},
            model_version='1.0'
        )
    
    # Placeholder methods for remaining functionality
    async def _identify_contributing_factors(self, features: Dict[str, float], patterns: List[TemporalPattern],
                                           prediction_type: PredictionType, horizon: TimeHorizon) -> Dict[str, float]:
        """Identify factors contributing to prediction"""
        return {'historical_performance': 0.4, 'recent_trend': 0.3, 'temporal_patterns': 0.3}
    
    async def _identify_uncertainty_sources(self, horizon: TimeHorizon, patterns: List[TemporalPattern],
                                          features: Dict[str, float]) -> List[str]:
        """Identify sources of prediction uncertainty"""
        sources = []
        if len(patterns) < 2:
            sources.append('limited_pattern_detection')
        if features.get('historical_data_points', 0) < 50:
            sources.append('insufficient_historical_data')
        return sources
    
    async def _analyze_trends_for_timeframe(self, features: Dict[str, float], patterns: List[TemporalPattern],
                                          horizon: TimeHorizon) -> Dict[str, float]:
        """Analyze trends relevant to timeframe"""
        return {
            'trend_direction': features.get('performance_trend', 0.0),
            'trend_strength': abs(features.get('performance_trend', 0.0)),
            'trend_consistency': 0.7
        }
    
    async def _extract_seasonal_patterns(self, patterns: List[TemporalPattern], horizon: TimeHorizon) -> Dict[str, float]:
        """Extract seasonal patterns relevant to timeframe"""
        seasonal = {}
        for pattern in patterns:
            if pattern.pattern_type in ['weekly_cycle', 'daily_cycle']:
                seasonal[pattern.pattern_type] = pattern.strength
        return seasonal
    
    async def _identify_risks_and_opportunities(self, student_id: str, prediction_type: PredictionType,
                                              horizon: TimeHorizon, features: Dict[str, float],
                                              patterns: List[TemporalPattern]) -> Tuple[List[str], List[str]]:
        """Identify risks and opportunities"""
        risks = []
        opportunities = []
        
        if features.get('performance_trend', 0.0) < -0.1:
            risks.append('declining_performance_trend')
        
        if features.get('current_performance', 0.5) > 0.8:
            opportunities.append('high_performance_momentum')
        
        return risks, opportunities
    
    async def _generate_adaptive_recommendations(self, student_id: str, prediction_type: PredictionType,
                                               timeframe_predictions: Dict[TimeHorizon, TimeframePrediction],
                                               patterns: List[TemporalPattern]) -> Dict[TimeHorizon, List[str]]:
        """Generate timeframe-specific recommendations"""
        recommendations = {}
        
        for horizon, pred in timeframe_predictions.items():
            recs = []
            
            if pred.confidence_score < 0.6:
                recs.append("Increase learning activity to improve prediction accuracy")
            
            if pred.predicted_value < 0.5:
                recs.append(f"Focus on improvement strategies for {horizon.value} timeframe")
            
            recommendations[horizon] = recs
        
        return recommendations
    
    async def _generate_monitoring_alerts(self, student_id: str, prediction_type: PredictionType,
                                        timeframe_predictions: Dict[TimeHorizon, TimeframePrediction]) -> List[Dict[str, Any]]:
        """Generate monitoring alerts"""
        alerts = []
        
        for horizon, pred in timeframe_predictions.items():
            if pred.predicted_value < 0.4 and pred.confidence_score > 0.7:
                alerts.append({
                    'type': 'performance_alert',
                    'severity': 'high',
                    'timeframe': horizon.value,
                    'message': f"Low predicted {prediction_type.value} for {horizon.value} period",
                    'confidence': pred.confidence_score
                })
        
        return alerts
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'active_forecasts': sum(len(forecasts) for forecasts in self.student_forecasts.values()),
            'tracked_students': len(self.student_forecasts),
            'prediction_types': len(PredictionType),
            'time_horizons': len(TimeHorizon),
            'neural_models': len(self.neural_models),
            'last_updated': datetime.now().isoformat()
        }

# Testing function
async def test_multi_timeframe_predictor():
    """Test the multi-timeframe prediction system"""
    try:
        logger.info("🧪 Testing Multi-Timeframe Prediction System")
        
        predictor = MultiTimeframePredictor()
        await predictor.initialize()
        
        # Test forecast generation
        forecast = await predictor.generate_multi_timeframe_forecast("test_student", PredictionType.PERFORMANCE)
        logger.info(f"✅ Generated forecast with {len(forecast.timeframe_predictions)} timeframes")
        
        # Test system status
        status = await predictor.get_system_status()
        logger.info(f"✅ System status: {status['neural_models']} models loaded")
        
        logger.info("✅ Multi-Timeframe Prediction System test completed")
        
    except Exception as e:
        logger.error(f"❌ Multi-Timeframe Prediction System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_multi_timeframe_predictor())