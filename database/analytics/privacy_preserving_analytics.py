#!/usr/bin/env python3
"""
Privacy-Preserving Analytics with Differential Privacy - Phase 6
Implements differential privacy mechanisms to protect student data while
enabling educational analytics and machine learning on sensitive educational data.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import hashlib
import hmac
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import warnings
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyMechanism(Enum):
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    RANDOMIZED_RESPONSE = "randomized_response"
    GEOMETRIC = "geometric"

class DataUtility(Enum):
    HIGH = "high"      # ε > 1.0
    MEDIUM = "medium"  # 0.1 < ε <= 1.0
    LOW = "low"        # ε <= 0.1

class PrivacyLevel(Enum):
    STRICT = "strict"      # ε ≤ 0.1
    MODERATE = "moderate"  # 0.1 < ε ≤ 1.0
    RELAXED = "relaxed"    # ε > 1.0

@dataclass
class PrivacyBudget:
    """Differential privacy budget management"""
    total_epsilon: float
    total_delta: float
    used_epsilon: float = 0.0
    used_delta: float = 0.0
    allocations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PrivacyParameters:
    """Privacy parameters for a specific query"""
    epsilon: float
    delta: float
    mechanism: PrivacyMechanism
    sensitivity: float
    query_id: str
    description: str = ""

@dataclass
class PrivateQuery:
    """A differentially private query"""
    query_id: str
    query_name: str
    query_function: Callable
    privacy_params: PrivacyParameters
    result_transformer: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivacyAuditLog:
    """Privacy audit log entry"""
    timestamp: datetime
    query_id: str
    epsilon_used: float
    delta_used: float
    mechanism_used: str
    data_size: int
    user_id: Optional[str] = None
    purpose: str = ""

class DifferentialPrivacy:
    """Core differential privacy mechanisms"""
    
    def __init__(self):
        self.mechanisms = {
            PrivacyMechanism.LAPLACE: self._laplace_mechanism,
            PrivacyMechanism.GAUSSIAN: self._gaussian_mechanism,
            PrivacyMechanism.EXPONENTIAL: self._exponential_mechanism,
            PrivacyMechanism.RANDOMIZED_RESPONSE: self._randomized_response,
            PrivacyMechanism.GEOMETRIC: self._geometric_mechanism
        }
    
    def _laplace_mechanism(self, true_value: float, sensitivity: float, 
                          epsilon: float) -> float:
        """Laplace mechanism for differential privacy"""
        try:
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale)
            return true_value + noise
        except Exception as e:
            logger.error(f"❌ Laplace mechanism failed: {e}")
            return true_value
    
    def _gaussian_mechanism(self, true_value: float, sensitivity: float,
                           epsilon: float, delta: float) -> float:
        """Gaussian mechanism for differential privacy"""
        try:
            if delta <= 0:
                raise ValueError("Delta must be positive for Gaussian mechanism")
            
            # Calculate standard deviation
            c = np.sqrt(2 * np.log(1.25 / delta))
            sigma = c * sensitivity / epsilon
            
            noise = np.random.normal(0, sigma)
            return true_value + noise
        except Exception as e:
            logger.error(f"❌ Gaussian mechanism failed: {e}")
            return true_value
    
    def _exponential_mechanism(self, candidates: List[Any], 
                              utility_function: Callable,
                              sensitivity: float, epsilon: float) -> Any:
        """Exponential mechanism for selecting from discrete options"""
        try:
            if not candidates:
                raise ValueError("No candidates provided")
            
            # Calculate utilities
            utilities = [utility_function(candidate) for candidate in candidates]
            
            # Calculate probabilities
            scaled_utilities = [epsilon * u / (2 * sensitivity) for u in utilities]
            max_utility = max(scaled_utilities)
            
            # Normalize for numerical stability
            exp_utilities = [np.exp(u - max_utility) for u in scaled_utilities]
            total_exp = sum(exp_utilities)
            
            probabilities = [exp_u / total_exp for exp_u in exp_utilities]
            
            # Sample according to probabilities
            return np.random.choice(candidates, p=probabilities)
        except Exception as e:
            logger.error(f"❌ Exponential mechanism failed: {e}")
            return candidates[0] if candidates else None
    
    def _randomized_response(self, true_value: bool, epsilon: float) -> bool:
        """Randomized response for binary data"""
        try:
            p = np.exp(epsilon) / (np.exp(epsilon) + 1)
            
            if true_value:
                return np.random.random() < p
            else:
                return np.random.random() >= p
        except Exception as e:
            logger.error(f"❌ Randomized response failed: {e}")
            return true_value
    
    def _geometric_mechanism(self, true_value: int, sensitivity: float,
                           epsilon: float) -> int:
        """Geometric mechanism for integer-valued queries"""
        try:
            alpha = np.exp(-epsilon / sensitivity)
            
            # Sample from geometric distribution
            if np.random.random() < 0.5:
                noise = np.random.geometric(1 - alpha) - 1
            else:
                noise = -(np.random.geometric(1 - alpha) - 1)
            
            return int(true_value + noise)
        except Exception as e:
            logger.error(f"❌ Geometric mechanism failed: {e}")
            return true_value
    
    def add_noise(self, true_value: Union[float, int, bool], 
                 privacy_params: PrivacyParameters) -> Union[float, int, bool]:
        """Add differential privacy noise to a value"""
        try:
            mechanism_func = self.mechanisms.get(privacy_params.mechanism)
            if not mechanism_func:
                raise ValueError(f"Unknown mechanism: {privacy_params.mechanism}")
            
            if privacy_params.mechanism == PrivacyMechanism.LAPLACE:
                return mechanism_func(float(true_value), privacy_params.sensitivity, 
                                    privacy_params.epsilon)
            elif privacy_params.mechanism == PrivacyMechanism.GAUSSIAN:
                return mechanism_func(float(true_value), privacy_params.sensitivity,
                                    privacy_params.epsilon, privacy_params.delta)
            elif privacy_params.mechanism == PrivacyMechanism.RANDOMIZED_RESPONSE:
                return mechanism_func(bool(true_value), privacy_params.epsilon)
            elif privacy_params.mechanism == PrivacyMechanism.GEOMETRIC:
                return mechanism_func(int(true_value), privacy_params.sensitivity,
                                    privacy_params.epsilon)
            else:
                return true_value
                
        except Exception as e:
            logger.error(f"❌ Failed to add privacy noise: {e}")
            return true_value

class DataAnonymizer:
    """Data anonymization and pseudonymization utilities"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        if secret_key is None:
            secret_key = Fernet.generate_key()
        self.cipher_suite = Fernet(secret_key)
        self.salt = secrets.token_bytes(32)
    
    def pseudonymize_id(self, original_id: str, context: str = "") -> str:
        """Create pseudonymous identifier"""
        try:
            # Create deterministic pseudonym using HMAC
            key = self.salt + context.encode('utf-8')
            pseudonym = hmac.new(key, original_id.encode('utf-8'), hashlib.sha256)
            return base64.urlsafe_b64encode(pseudonym.digest()).decode('utf-8')[:16]
        except Exception as e:
            logger.error(f"❌ Pseudonymization failed: {e}")
            return original_id
    
    def anonymize_demographics(self, age: int, location: str) -> Tuple[str, str]:
        """Anonymize demographic data"""
        try:
            # Age to age group
            if age < 18:
                age_group = "under_18"
            elif age < 25:
                age_group = "18_24"
            elif age < 35:
                age_group = "25_34"
            elif age < 50:
                age_group = "35_49"
            else:
                age_group = "50_plus"
            
            # Location to region (simplified)
            location_parts = location.split(',')
            if len(location_parts) >= 2:
                region = location_parts[-1].strip()  # Country/state
            else:
                region = "unknown"
            
            return age_group, region
        except Exception as e:
            logger.error(f"❌ Demographic anonymization failed: {e}")
            return "unknown", "unknown"
    
    def k_anonymize_quasi_identifiers(self, data: pd.DataFrame, 
                                    quasi_identifiers: List[str], k: int = 5) -> pd.DataFrame:
        """Apply k-anonymity to quasi-identifiers"""
        try:
            if data.empty or not quasi_identifiers:
                return data
            
            anonymized_data = data.copy()
            
            # Group records and generalize if group size < k
            for qi in quasi_identifiers:
                if qi in anonymized_data.columns:
                    # Simple generalization: group rare values
                    value_counts = anonymized_data[qi].value_counts()
                    rare_values = value_counts[value_counts < k].index
                    
                    if len(rare_values) > 0:
                        anonymized_data.loc[anonymized_data[qi].isin(rare_values), qi] = 'other'
            
            return anonymized_data
        except Exception as e:
            logger.error(f"❌ K-anonymization failed: {e}")
            return data

class PrivacyBudgetManager:
    """Manage differential privacy budget allocation"""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.budget = PrivacyBudget(total_epsilon, total_delta)
        self.query_allocations = {}
        self.audit_log = []
    
    def allocate_budget(self, query_id: str, epsilon: float, 
                       delta: float = 0.0) -> bool:
        """Allocate privacy budget for a query"""
        try:
            # Check if sufficient budget available
            if (self.budget.used_epsilon + epsilon > self.budget.total_epsilon or
                self.budget.used_delta + delta > self.budget.total_delta):
                logger.warning(f"⚠️ Insufficient privacy budget for query {query_id}")
                return False
            
            # Allocate budget
            self.budget.used_epsilon += epsilon
            self.budget.used_delta += delta
            
            self.budget.allocations[query_id] = {
                'epsilon': epsilon,
                'delta': delta,
                'allocated_at': datetime.now()
            }
            
            logger.info(f"💰 Allocated ε={epsilon}, δ={delta} to query {query_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Budget allocation failed: {e}")
            return False
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (
            self.budget.total_epsilon - self.budget.used_epsilon,
            self.budget.total_delta - self.budget.used_delta
        )
    
    def reset_budget(self):
        """Reset privacy budget (use with caution)"""
        self.budget.used_epsilon = 0.0
        self.budget.used_delta = 0.0
        self.budget.allocations = {}
        logger.info("🔄 Privacy budget reset")
    
    def log_query(self, query_id: str, epsilon_used: float, delta_used: float,
                 mechanism: str, data_size: int, purpose: str = ""):
        """Log privacy budget usage"""
        log_entry = PrivacyAuditLog(
            timestamp=datetime.now(),
            query_id=query_id,
            epsilon_used=epsilon_used,
            delta_used=delta_used,
            mechanism_used=mechanism,
            data_size=data_size,
            purpose=purpose
        )
        self.audit_log.append(log_entry)

class EducationalPrivacyAnalytics:
    """Privacy-preserving analytics for educational data"""
    
    def __init__(self, total_epsilon: float = 10.0, total_delta: float = 1e-5):
        self.dp_engine = DifferentialPrivacy()
        self.budget_manager = PrivacyBudgetManager(total_epsilon, total_delta)
        self.anonymizer = DataAnonymizer()
        
        # Predefined queries with privacy parameters
        self.registered_queries = {}
        
        # Cache for repeated queries
        self.query_cache = {}
        
        # Educational data sensitivities
        self.sensitivity_map = {
            'count': 1.0,
            'sum': 1.0,
            'mean': 1.0,  # Assuming normalized scores 0-1
            'success_rate': 1.0,
            'time_spent': 100.0,  # Maximum realistic session time in minutes
            'help_requests': 1.0,
            'difficulty_level': 1.0
        }
    
    async def initialize(self):
        """Initialize privacy-preserving analytics"""
        try:
            logger.info("🔒 Initializing Privacy-Preserving Analytics")
            
            # Register common educational queries
            await self._register_common_queries()
            
            logger.info("✅ Privacy-Preserving Analytics initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Privacy-Preserving Analytics: {e}")
            return False
    
    async def _register_common_queries(self):
        """Register commonly used educational analytics queries"""
        try:
            # Student performance queries
            self.register_query(
                "student_success_rate",
                lambda data: data['success'].mean() if 'success' in data.columns else 0.0,
                PrivacyParameters(
                    epsilon=0.1,
                    delta=1e-6,
                    mechanism=PrivacyMechanism.LAPLACE,
                    sensitivity=1.0 / len(pd.DataFrame()),  # Will be updated with actual data size
                    query_id="student_success_rate",
                    description="Average success rate across students"
                )
            )
            
            # Engagement metrics
            self.register_query(
                "avg_session_time",
                lambda data: data['session_duration'].mean() if 'session_duration' in data.columns else 0.0,
                PrivacyParameters(
                    epsilon=0.2,
                    delta=1e-6,
                    mechanism=PrivacyMechanism.LAPLACE,
                    sensitivity=100.0,  # Max session time sensitivity
                    query_id="avg_session_time",
                    description="Average session duration"
                )
            )
            
            # Help-seeking behavior
            self.register_query(
                "help_seeking_rate",
                lambda data: (data['help_requests'] > 0).mean() if 'help_requests' in data.columns else 0.0,
                PrivacyParameters(
                    epsilon=0.1,
                    delta=1e-6,
                    mechanism=PrivacyMechanism.RANDOMIZED_RESPONSE,
                    sensitivity=1.0,
                    query_id="help_seeking_rate",
                    description="Proportion of students seeking help"
                )
            )
            
            # Concept difficulty analysis
            self.register_query(
                "concept_difficulty_ranking",
                lambda data: data.groupby('concept')['success'].mean().sort_values(),
                PrivacyParameters(
                    epsilon=0.5,
                    delta=1e-5,
                    mechanism=PrivacyMechanism.EXPONENTIAL,
                    sensitivity=1.0,
                    query_id="concept_difficulty_ranking",
                    description="Ranking of concepts by difficulty"
                )
            )
            
            logger.info("📋 Registered common educational queries")
            
        except Exception as e:
            logger.error(f"❌ Failed to register queries: {e}")
    
    def register_query(self, query_name: str, query_function: Callable,
                      privacy_params: PrivacyParameters,
                      result_transformer: Optional[Callable] = None):
        """Register a private query"""
        try:
            query = PrivateQuery(
                query_id=privacy_params.query_id,
                query_name=query_name,
                query_function=query_function,
                privacy_params=privacy_params,
                result_transformer=result_transformer
            )
            
            self.registered_queries[query_name] = query
            logger.info(f"📝 Registered query: {query_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register query: {e}")
    
    async def execute_private_query(self, query_name: str, data: pd.DataFrame,
                                  additional_params: Dict[str, Any] = None) -> Any:
        """Execute a registered private query"""
        try:
            if query_name not in self.registered_queries:
                raise ValueError(f"Query {query_name} not registered")
            
            query = self.registered_queries[query_name]
            
            # Check cache first
            cache_key = self._generate_cache_key(query_name, data, additional_params)
            if cache_key in self.query_cache:
                logger.info(f"📋 Using cached result for {query_name}")
                return self.query_cache[cache_key]
            
            # Update sensitivity based on data size
            updated_params = self._update_sensitivity_for_data(query.privacy_params, data)
            
            # Check and allocate privacy budget
            if not self.budget_manager.allocate_budget(
                query.query_id, updated_params.epsilon, updated_params.delta
            ):
                raise ValueError("Insufficient privacy budget")
            
            # Anonymize data first
            anonymized_data = await self._anonymize_data(data)
            
            # Execute query on anonymized data
            true_result = query.query_function(anonymized_data)
            
            # Add differential privacy noise
            if isinstance(true_result, (int, float, bool)):
                private_result = self.dp_engine.add_noise(true_result, updated_params)
            elif isinstance(true_result, pd.Series):
                # For series results (like rankings), apply noise element-wise
                private_result = true_result.copy()
                for idx in private_result.index:
                    private_result[idx] = self.dp_engine.add_noise(
                        private_result[idx], updated_params
                    )
            else:
                # For complex results, apply transformation if available
                if query.result_transformer:
                    private_result = query.result_transformer(true_result, updated_params)
                else:
                    private_result = true_result
            
            # Log the query
            self.budget_manager.log_query(
                query.query_id, updated_params.epsilon, updated_params.delta,
                updated_params.mechanism.value, len(data),
                f"Educational analytics: {query_name}"
            )
            
            # Cache result
            self.query_cache[cache_key] = private_result
            
            logger.info(f"🔒 Executed private query: {query_name}")
            return private_result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute private query {query_name}: {e}")
            raise
    
    async def _anonymize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Anonymize data before processing"""
        try:
            anonymized_data = data.copy()
            
            # Pseudonymize student IDs
            if 'student_id' in anonymized_data.columns:
                anonymized_data['student_id'] = anonymized_data['student_id'].apply(
                    lambda x: self.anonymizer.pseudonymize_id(str(x), "student")
                )
            
            if 'user_id' in anonymized_data.columns:
                anonymized_data['user_id'] = anonymized_data['user_id'].apply(
                    lambda x: self.anonymizer.pseudonymize_id(str(x), "user")
                )
            
            # Anonymize demographic data if present
            if 'age' in anonymized_data.columns and 'location' in anonymized_data.columns:
                demo_data = anonymized_data[['age', 'location']].apply(
                    lambda row: self.anonymizer.anonymize_demographics(row['age'], row['location']),
                    axis=1, result_type='expand'
                )
                anonymized_data['age_group'] = demo_data[0]
                anonymized_data['region'] = demo_data[1]
                anonymized_data = anonymized_data.drop(['age', 'location'], axis=1)
            
            # Apply k-anonymity to quasi-identifiers
            quasi_identifiers = ['age_group', 'region', 'grade_level']
            available_qi = [qi for qi in quasi_identifiers if qi in anonymized_data.columns]
            if available_qi:
                anonymized_data = self.anonymizer.k_anonymize_quasi_identifiers(
                    anonymized_data, available_qi, k=5
                )
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"❌ Data anonymization failed: {e}")
            return data
    
    def _update_sensitivity_for_data(self, params: PrivacyParameters, 
                                   data: pd.DataFrame) -> PrivacyParameters:
        """Update sensitivity based on actual data characteristics"""
        try:
            updated_params = PrivacyParameters(
                epsilon=params.epsilon,
                delta=params.delta,
                mechanism=params.mechanism,
                sensitivity=params.sensitivity,
                query_id=params.query_id,
                description=params.description
            )
            
            # Adjust sensitivity for count queries
            if 'count' in params.query_id.lower():
                updated_params.sensitivity = 1.0
            
            # Adjust sensitivity for mean queries based on data range
            elif 'mean' in params.query_id.lower() or 'avg' in params.query_id.lower():
                # For normalized data (0-1 range)
                if len(data) > 0:
                    updated_params.sensitivity = 1.0 / len(data)
                
            # Adjust sensitivity for sum queries
            elif 'sum' in params.query_id.lower():
                updated_params.sensitivity = 1.0
            
            return updated_params
            
        except Exception as e:
            logger.error(f"❌ Sensitivity update failed: {e}")
            return params
    
    def _generate_cache_key(self, query_name: str, data: pd.DataFrame,
                          params: Dict[str, Any] = None) -> str:
        """Generate cache key for query result"""
        try:
            # Create hash based on query name, data shape, and parameters
            key_components = [
                query_name,
                str(data.shape),
                str(sorted(data.columns.tolist())),
                str(params) if params else ""
            ]
            
            key_string = "|".join(key_components)
            return hashlib.md5(key_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"❌ Cache key generation failed: {e}")
            return query_name
    
    async def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        try:
            remaining_epsilon, remaining_delta = self.budget_manager.get_remaining_budget()
            
            report = {
                'privacy_budget': {
                    'total_epsilon': self.budget_manager.budget.total_epsilon,
                    'total_delta': self.budget_manager.budget.total_delta,
                    'used_epsilon': self.budget_manager.budget.used_epsilon,
                    'used_delta': self.budget_manager.budget.used_delta,
                    'remaining_epsilon': remaining_epsilon,
                    'remaining_delta': remaining_delta,
                    'budget_utilization': self.budget_manager.budget.used_epsilon / self.budget_manager.budget.total_epsilon
                },
                'query_statistics': {
                    'total_queries': len(self.budget_manager.audit_log),
                    'unique_query_types': len(set(log.query_id for log in self.budget_manager.audit_log)),
                    'registered_queries': len(self.registered_queries)
                },
                'privacy_mechanisms_used': Counter(log.mechanism_used for log in self.budget_manager.audit_log),
                'data_processed': sum(log.data_size for log in self.budget_manager.audit_log),
                'compliance_status': self._assess_compliance_status(),
                'recommendations': self._generate_privacy_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Privacy report generation failed: {e}")
            return {}
    
    def _assess_compliance_status(self) -> Dict[str, Any]:
        """Assess privacy compliance status"""
        try:
            remaining_epsilon, remaining_delta = self.budget_manager.get_remaining_budget()
            
            # Check budget exhaustion
            budget_status = "healthy"
            if remaining_epsilon < 0.1:
                budget_status = "critical"
            elif remaining_epsilon < 1.0:
                budget_status = "warning"
            
            # Check for proper mechanism usage
            mechanism_diversity = len(set(log.mechanism_used for log in self.budget_manager.audit_log))
            
            compliance = {
                'budget_status': budget_status,
                'budget_exhaustion_risk': remaining_epsilon < 1.0,
                'mechanism_diversity': mechanism_diversity,
                'audit_trail_complete': len(self.budget_manager.audit_log) > 0,
                'privacy_level': self._determine_privacy_level()
            }
            
            return compliance
            
        except Exception as e:
            logger.error(f"❌ Compliance assessment failed: {e}")
            return {}
    
    def _determine_privacy_level(self) -> str:
        """Determine overall privacy level"""
        try:
            if self.budget_manager.budget.total_epsilon <= 0.1:
                return PrivacyLevel.STRICT.value
            elif self.budget_manager.budget.total_epsilon <= 1.0:
                return PrivacyLevel.MODERATE.value
            else:
                return PrivacyLevel.RELAXED.value
                
        except Exception as e:
            logger.error(f"❌ Privacy level determination failed: {e}")
            return "unknown"
    
    def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy improvement recommendations"""
        try:
            recommendations = []
            remaining_epsilon, _ = self.budget_manager.get_remaining_budget()
            
            if remaining_epsilon < 0.5:
                recommendations.append("Consider resetting privacy budget for next analysis period")
            
            if len(self.registered_queries) < 5:
                recommendations.append("Register more common queries to improve efficiency")
            
            mechanism_counts = Counter(log.mechanism_used for log in self.budget_manager.audit_log)
            if len(mechanism_counts) == 1:
                recommendations.append("Consider using diverse privacy mechanisms for different query types")
            
            if not recommendations:
                recommendations.append("Privacy configuration appears optimal")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return ["Unable to generate recommendations"]
    
    async def simulate_query_impact(self, query_name: str, epsilon: float,
                                  delta: float = 0.0) -> Dict[str, Any]:
        """Simulate the impact of running a query on privacy budget"""
        try:
            remaining_epsilon, remaining_delta = self.budget_manager.get_remaining_budget()
            
            impact = {
                'can_execute': epsilon <= remaining_epsilon and delta <= remaining_delta,
                'budget_after_execution': {
                    'epsilon': remaining_epsilon - epsilon,
                    'delta': remaining_delta - delta
                },
                'budget_utilization_after': (self.budget_manager.budget.used_epsilon + epsilon) / self.budget_manager.budget.total_epsilon,
                'privacy_level_after': self._get_privacy_level_for_epsilon(remaining_epsilon - epsilon),
                'risk_assessment': self._assess_query_risk(epsilon, delta)
            }
            
            return impact
            
        except Exception as e:
            logger.error(f"❌ Query impact simulation failed: {e}")
            return {}
    
    def _get_privacy_level_for_epsilon(self, epsilon: float) -> str:
        """Get privacy level for given epsilon"""
        if epsilon <= 0.1:
            return PrivacyLevel.STRICT.value
        elif epsilon <= 1.0:
            return PrivacyLevel.MODERATE.value
        else:
            return PrivacyLevel.RELAXED.value
    
    def _assess_query_risk(self, epsilon: float, delta: float) -> str:
        """Assess risk level of a query"""
        try:
            if epsilon > 1.0:
                return "high"
            elif epsilon > 0.5:
                return "medium"
            else:
                return "low"
        except:
            return "unknown"

# Testing function
async def test_privacy_preserving_analytics():
    """Test privacy-preserving analytics system"""
    try:
        logger.info("🧪 Testing Privacy-Preserving Analytics")
        
        # Initialize system
        privacy_analytics = EducationalPrivacyAnalytics(total_epsilon=5.0, total_delta=1e-5)
        await privacy_analytics.initialize()
        
        # Create sample educational data
        np.random.seed(42)
        n_students = 1000
        
        sample_data = pd.DataFrame({
            'student_id': [f"student_{i}" for i in range(n_students)],
            'success': np.random.binomial(1, 0.7, n_students),
            'session_duration': np.random.exponential(30, n_students),
            'help_requests': np.random.poisson(2, n_students),
            'concept': np.random.choice(['mechanics', 'energy', 'waves', 'electricity'], n_students),
            'age': np.random.randint(18, 25, n_students),
            'location': np.random.choice(['US,CA', 'US,NY', 'UK,London', 'CA,Toronto'], n_students)
        })
        
        logger.info(f"📊 Created sample data with {len(sample_data)} students")
        
        # Test registered queries
        queries_to_test = ['student_success_rate', 'avg_session_time', 'help_seeking_rate']
        
        for query_name in queries_to_test:
            try:
                result = await privacy_analytics.execute_private_query(query_name, sample_data)
                logger.info(f"✅ {query_name}: {result}")
            except Exception as e:
                logger.error(f"❌ Query {query_name} failed: {e}")
        
        # Test custom query
        def custom_concept_success(data):
            return data.groupby('concept')['success'].mean()
        
        privacy_analytics.register_query(
            "concept_success_rates",
            custom_concept_success,
            PrivacyParameters(
                epsilon=0.5,
                delta=1e-6,
                mechanism=PrivacyMechanism.LAPLACE,
                sensitivity=1.0,
                query_id="concept_success_rates",
                description="Success rates by concept"
            )
        )
        
        concept_results = await privacy_analytics.execute_private_query("concept_success_rates", sample_data)
        logger.info(f"📚 Concept success rates: {concept_results}")
        
        # Generate privacy report
        privacy_report = await privacy_analytics.generate_privacy_report()
        logger.info(f"🔒 Privacy Budget Used: {privacy_report['privacy_budget']['budget_utilization']:.1%}")
        logger.info(f"🔒 Privacy Level: {privacy_report['compliance_status']['privacy_level']}")
        
        # Test query impact simulation
        impact = await privacy_analytics.simulate_query_impact("test_query", 1.0, 1e-6)
        logger.info(f"📊 Query impact simulation: Can execute = {impact['can_execute']}")
        
        logger.info("✅ Privacy-Preserving Analytics test completed")
        
    except Exception as e:
        logger.error(f"❌ Privacy-Preserving Analytics test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_privacy_preserving_analytics())