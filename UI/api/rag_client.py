"""
RAG Client for Physics Agent Integration  
Provides interface for agents to interact with the Graph RAG system
"""

import asyncio
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


class RAGClientError(Exception):
    """Base exception for RAG client errors"""
    pass


class RAGClient:
    """
    Client for interacting with the Physics Assistant RAG system
    
    Provides methods for:
    - Semantic and graph-based content retrieval
    - Student profile management
    - Learning path generation
    - Context augmentation for physics agents
    """
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8001",
                 timeout: int = 30,
                 enable_cache: bool = True,
                 max_cache_size: int = 1000,
                 enable_fallback: bool = True):
        
        self.api_base_url = api_base_url.rstrip('/')
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.enable_fallback = enable_fallback
        
        # HTTP session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Simple in-memory cache
        self._cache = {}
        self._cache_times = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "errors": 0,
            "avg_response_time": 0.0
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if hasattr(self, 'session'):
            self.session.close()
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        return f"{endpoint}:{hash(json.dumps(params, sort_keys=True))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid"""
        if not self.enable_cache or cache_key not in self._cache_times:
            return False
        return time.time() - self._cache_times[cache_key] < self._cache_ttl
    
    def _cache_response(self, cache_key: str, response: Any):
        """Cache response with TTL management"""
        if not self.enable_cache:
            return
            
        # Simple LRU eviction
        if len(self._cache) >= self.max_cache_size:
            oldest_key = min(self._cache_times.keys(), key=self._cache_times.get)
            del self._cache[oldest_key]
            del self._cache_times[oldest_key]
        
        self._cache[cache_key] = response
        self._cache_times[cache_key] = time.time()
    
    async def _make_request(self, endpoint: str, method: str = "POST", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to RAG API with error handling"""
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(endpoint, data or {})
            if self._is_cache_valid(cache_key):
                self.metrics["cache_hits"] += 1
                return self._cache[cache_key]
            
            url = f"{self.api_base_url}/{endpoint.lstrip('/')}"
            
            # Make synchronous request (will be wrapped in async context)
            if method.upper() == "GET":
                response = self.session.get(url, params=data, timeout=self.timeout)
            else:
                response = self.session.post(url, json=data, timeout=self.timeout)
            
            # Parse response
            result = response.json()
            
            # Cache successful responses
            if response.status_code == 200:
                self._cache_response(cache_key, result)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1) + response_time) /
                self.metrics["total_queries"]
            )
            
            return result
            
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"RAG API request failed: {e}")
            
            if self.enable_fallback:
                return self._get_fallback_response(endpoint, data)
            else:
                raise RAGClientError(f"RAG API request failed: {e}")
    
    def _get_fallback_response(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback response when RAG system is unavailable"""
        logger.warning(f"Using fallback for endpoint: {endpoint}")
        
        fallback_responses = {
            "rag/query": {
                "status": "success",
                "results": {
                    "concepts": [],
                    "formulas": [],
                    "examples": [],
                    "learning_paths": [],
                    "context": "RAG system unavailable - using basic physics knowledge"
                },
                "metadata": {"fallback": True, "source": "basic_physics"}
            },
            "rag/semantic-search": {
                "status": "success",
                "results": [],
                "metadata": {"fallback": True, "total_results": 0}
            },
            "rag/student-profile": {
                "profile": {
                    "user_id": data.get("user_id", "unknown"),
                    "level": "intermediate",
                    "mastered_concepts": [],
                    "struggling_concepts": [],
                    "learning_preferences": {"visual": 0.5, "algebraic": 0.5}
                },
                "metadata": {"fallback": True}
            }
        }
        
        for pattern, response in fallback_responses.items():
            if pattern in endpoint:
                return response
        
        return {"status": "error", "message": "RAG system unavailable", "fallback": True}
    
    async def augment_agent_context(self, 
                                   problem: str, 
                                   agent_type: str,
                                   user_id: str = "anonymous",
                                   difficulty_level: str = "intermediate") -> Dict[str, Any]:
        """
        Main method for agents to get augmented context for problem solving
        
        Args:
            problem: Physics problem description
            agent_type: Type of physics agent (forces, kinematics, etc.)
            user_id: Student identifier for personalization
            difficulty_level: Problem difficulty (beginner, intermediate, advanced)
            
        Returns:
            Dict containing relevant concepts, formulas, examples, and context
        """
        try:
            # Main RAG query for comprehensive context
            rag_request = {
                "text": problem,
                "agent_type": agent_type,
                "user_id": user_id,
                "search_type": "comprehensive",
                "include_examples": True,
                "include_formulas": True,
                "include_learning_paths": True,
                "difficulty_level": difficulty_level,
                "limit": 10
            }
            
            context = await self._make_request("rag/query", "POST", rag_request)
            
            # Get student profile for personalization
            profile = await self.get_student_profile(user_id)
            
            # Format context for agent consumption
            formatted_context = self._format_agent_context(context, profile, agent_type)
            
            return formatted_context
            
        except Exception as e:
            logger.error(f"Failed to augment agent context: {e}")
            return self._get_fallback_context(problem, agent_type)
    
    def _format_agent_context(self, 
                             rag_context: Dict[str, Any], 
                             student_profile: Dict[str, Any],
                             agent_type: str) -> Dict[str, Any]:
        """Format RAG results for agent consumption"""
        
        results = rag_context.get("results", {})
        profile_data = student_profile.get("profile", {})
        
        return {
            "relevant_concepts": results.get("concepts", []),
            "applicable_formulas": results.get("formulas", []),
            "example_problems": results.get("examples", []),
            "learning_paths": results.get("learning_paths", []),
            "student_context": {
                "level": profile_data.get("level", "intermediate"),
                "mastered_concepts": profile_data.get("mastered_concepts", []),
                "struggling_concepts": profile_data.get("struggling_concepts", []),
                "learning_preferences": profile_data.get("learning_preferences", {})
            },
            "agent_context": {
                "agent_type": agent_type,
                "suggested_approach": self._get_approach_suggestion(agent_type, profile_data),
                "difficulty_adjustment": self._get_difficulty_adjustment(profile_data)
            },
            "educational_context": results.get("context", ""),
            "metadata": {
                "rag_available": not rag_context.get("fallback", False),
                "timestamp": datetime.now().isoformat(),
                "cache_hit": False  # Updated by caching logic
            }
        }
    
    def _get_approach_suggestion(self, agent_type: str, profile: Dict[str, Any]) -> str:
        """Suggest teaching approach based on agent type and student profile"""
        preferences = profile.get("learning_preferences", {})
        level = profile.get("level", "intermediate")
        
        approaches = {
            "forces": {
                "visual": "Start with free body diagrams and vector analysis",
                "algebraic": "Begin with component analysis and equilibrium equations",
                "beginner": "Focus on single-force scenarios before multiple forces",
                "advanced": "Include non-inertial reference frames and complex systems"
            },
            "kinematics": {
                "visual": "Use motion graphs and trajectory visualizations",
                "algebraic": "Focus on kinematic equations and mathematical relationships",
                "beginner": "Start with 1D motion before 2D projectile motion",
                "advanced": "Include relative motion and calculus-based derivations"
            },
            "energy": {
                "visual": "Use energy bar charts and conservation diagrams",
                "algebraic": "Emphasize work-energy theorem calculations",
                "beginner": "Focus on mechanical energy conservation",
                "advanced": "Include non-conservative forces and energy dissipation"
            }
        }
        
        agent_approaches = approaches.get(agent_type, {})
        
        # Prioritize learning preference, then level
        if preferences.get("visual", 0) > preferences.get("algebraic", 0):
            return agent_approaches.get("visual", "Use visual problem-solving methods")
        elif preferences.get("algebraic", 0) > preferences.get("visual", 0):
            return agent_approaches.get("algebraic", "Use algebraic problem-solving methods")
        else:
            return agent_approaches.get(level, "Use balanced visual and algebraic approaches")
    
    def _get_difficulty_adjustment(self, profile: Dict[str, Any]) -> str:
        """Suggest difficulty adjustments based on student profile"""
        struggling = profile.get("struggling_concepts", [])
        mastered = profile.get("mastered_concepts", [])
        
        if len(struggling) > len(mastered):
            return "provide_extra_scaffolding"
        elif len(mastered) > len(struggling) * 2:
            return "increase_challenge"
        else:
            return "maintain_current_level"
    
    def _get_fallback_context(self, problem: str, agent_type: str) -> Dict[str, Any]:
        """Provide basic context when RAG system is unavailable"""
        return {
            "relevant_concepts": [],
            "applicable_formulas": [],
            "example_problems": [],
            "learning_paths": [],
            "student_context": {
                "level": "intermediate",
                "mastered_concepts": [],
                "struggling_concepts": [],
                "learning_preferences": {"visual": 0.5, "algebraic": 0.5}
            },
            "agent_context": {
                "agent_type": agent_type,
                "suggested_approach": "Use standard problem-solving methods",
                "difficulty_adjustment": "maintain_current_level"
            },
            "educational_context": "RAG system unavailable - using basic physics knowledge",
            "metadata": {
                "rag_available": False,
                "timestamp": datetime.now().isoformat(),
                "fallback": True
            }
        }
    
    async def get_student_profile(self, user_id: str) -> Dict[str, Any]:
        """Get student learning profile"""
        return await self._make_request(f"rag/student-profile/{user_id}", "GET")
    
    async def update_student_progress(self, 
                                    user_id: str, 
                                    interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update student profile based on problem-solving interaction"""
        request_data = {
            "user_id": user_id,
            "interaction_data": interaction_data
        }
        return await self._make_request("rag/update-profile", "POST", request_data)
    
    async def semantic_search(self, 
                             query: str, 
                             content_types: List[str] = None,
                             limit: int = 10) -> Dict[str, Any]:
        """Perform semantic search on physics content"""
        request_data = {
            "text": query,
            "content_types": content_types or ["concepts", "formulas", "examples"],
            "limit": limit,
            "min_similarity": 0.3
        }
        return await self._make_request("rag/semantic-search", "POST", request_data)
    
    async def graph_search(self, 
                          query: str,
                          traversal_strategy: str = "breadth_first",
                          include_learning_paths: bool = True) -> Dict[str, Any]:
        """Perform graph-enhanced search"""
        request_data = {
            "text": query,
            "traversal_strategy": traversal_strategy,
            "include_learning_paths": include_learning_paths,
            "student_level": "intermediate"
        }
        return await self._make_request("rag/graph-search", "POST", request_data)
    
    async def generate_learning_path(self, 
                                   start_concept: str,
                                   end_concept: str,
                                   student_level: str = "intermediate") -> Dict[str, Any]:
        """Generate learning path between concepts"""
        request_data = {
            "start_concept": start_concept,
            "end_concept": end_concept,
            "student_level": student_level,
            "max_paths": 3
        }
        return await self._make_request("rag/learning-path", "POST", request_data)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system health status"""
        return await self._make_request("rag/system-status", "GET")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client performance metrics"""
        cache_hit_rate = (
            self.metrics["cache_hits"] / self.metrics["total_queries"] 
            if self.metrics["total_queries"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Clear client-side cache"""
        self._cache.clear()
        self._cache_times.clear()
        logger.info("RAG client cache cleared")


# Singleton instance for global use
_rag_client_instance = None

async def get_rag_client(api_base_url: str = "http://localhost:8001") -> RAGClient:
    """Get or create RAG client instance"""
    global _rag_client_instance
    
    if _rag_client_instance is None:
        _rag_client_instance = RAGClient(api_base_url=api_base_url)
    
    return _rag_client_instance