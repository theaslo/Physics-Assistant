"""
API Client for Physics Assistant FastAPI server
Handles communication with the FastAPI backend hosting CombinedPhysicsAgent
Enhanced with database logging via database API integration
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any
import streamlit as st
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhysicsAPIClient:
    """Client for communicating with the Physics Assistant FastAPI server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Initialize database client for logging
        self.db_client = None
        if Config.DATABASE_LOGGING_ENABLED:
            try:
                from .database_client import DatabaseAPIClient
                self.db_client = DatabaseAPIClient(Config.DATABASE_API_URL)
                logger.info("✅ Database logging enabled for API client")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize database logging: {e}")
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to API server"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                st.session_state['api_connected'] = True
            else:
                st.session_state['api_connected'] = False
        except requests.exceptions.RequestException:
            st.session_state['api_connected'] = False
    
    def is_connected(self) -> bool:
        """Check if API is connected"""
        return st.session_state.get('api_connected', False)
    
    def _log_api_interaction(self, 
                           operation: str,
                           agent_id: str,
                           request_data: Dict[str, Any],
                           response_data: Dict[str, Any],
                           execution_time_ms: int,
                           success: bool) -> None:
        """Log API interaction to database if logging is enabled"""
        if not self.db_client or not self.db_client.is_connected:
            return
            
        try:
            # Get user info for logging
            user_info = st.session_state.get('user_info', {})
            user_id = user_info.get('id', 'anonymous')
            
            # Prepare metadata
            metadata = {
                'operation': operation,
                'api_endpoint': f"{self.base_url}/{operation}",
                'request_data': request_data,
                'response_status': 'success' if success else 'error',
                'execution_time_ms': execution_time_ms,
                'api_client_version': '1.0.0',
                'streamlit_session_id': st.session_state.get('session_id'),
                'timestamp': time.time()
            }
            
            # Format message for database logging
            message = f"API {operation}: {request_data.get('problem', request_data.get('message', 'N/A'))}"
            response = response_data.get('solution', response_data.get('content', str(response_data)))
            
            # Log to database
            self.db_client.log_interaction(
                user_id=user_id,
                agent_type=f"api_{agent_id}",
                message=message,
                response=response,
                session_id=st.session_state.get('db_session_id'),
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to log API interaction: {e}")
    
    def create_agent(self, agent_id: str, use_direct_tools: bool = True) -> Dict[str, Any]:
        """
        Create and initialize a physics agent
        
        Args:
            agent_id: Agent type (forces_agent, kinematics_agent, math_agent, momentum_agent, energy_agent, angular_motion_agent)
            use_direct_tools: Whether to use direct MCP tools
            
        Returns:
            Dict containing success status and agent information
        """
        try:
            payload = {
                "agent_id": agent_id,
                "use_direct_tools": use_direct_tools
            }
            
            response = self.session.post(
                f"{self.base_url}/agent/create",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state[f'agent_{agent_id}_ready'] = True
                return result
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {error_detail}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Connection error: {str(e)}"
            }
    
    def solve_problem(self, 
                     agent_id: str, 
                     problem: str, 
                     context: Optional[Dict[str, Any]] = None,
                     use_direct_tools: bool = True) -> Dict[str, Any]:
        """
        Solve a physics problem using the specified agent
        
        Args:
            agent_id: Agent type (forces_agent, kinematics_agent, math_agent, momentum_agent, energy_agent, angular_motion_agent)
            problem: Problem description
            context: Optional context for the problem
            use_direct_tools: Whether to use direct MCP tools
            
        Returns:
            Dict containing solution and metadata
        """
        start_time = time.time()
        request_data = {
            "problem": problem,
            "context": context,
            "use_direct_tools": use_direct_tools
        }
        
        try:
            payload = {
                "problem": problem,
                "context": context
            }
            
            # Add use_direct_tools as query parameter
            params = {"use_direct_tools": use_direct_tools}
            
            response = self.session.post(
                f"{self.base_url}/agent/{agent_id}/solve",
                json=payload,
                params=params,
                timeout=60  # Longer timeout for problem solving
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                response_data = response.json()
                success = response_data.get('success', True)
                
                # Log interaction to database
                self._log_api_interaction(
                    operation="solve_problem",
                    agent_id=agent_id,
                    request_data=request_data,
                    response_data=response_data,
                    execution_time_ms=execution_time_ms,
                    success=success
                )
                
                return response_data
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                error_response = {
                    'success': False,
                    'agent_id': agent_id,
                    'problem': problem,
                    'error': f"HTTP {response.status_code}: {error_detail}"
                }
                
                # Log failed interaction
                self._log_api_interaction(
                    operation="solve_problem",
                    agent_id=agent_id,
                    request_data=request_data,
                    response_data=error_response,
                    execution_time_ms=execution_time_ms,
                    success=False
                )
                
                return error_response
                
        except requests.exceptions.RequestException as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_response = {
                'success': False,
                'agent_id': agent_id,
                'problem': problem,
                'error': f"Connection error: {str(e)}"
            }
            
            # Log failed interaction
            self._log_api_interaction(
                operation="solve_problem",
                agent_id=agent_id,
                request_data=request_data,
                response_data=error_response,
                execution_time_ms=execution_time_ms,
                success=False
            )
            
            return error_response
    
    def check_agent_health(self, agent_id: str, use_direct_tools: bool = True) -> Dict[str, Any]:
        """
        Check the health status of a physics agent
        
        Args:
            agent_id: Agent type to check
            use_direct_tools: Whether agent uses direct MCP tools
            
        Returns:
            Dict containing health status
        """
        try:
            params = {"use_direct_tools": use_direct_tools}
            
            response = self.session.get(
                f"{self.base_url}/agent/{agent_id}/health",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'agent_id': agent_id,
                    'status': 'error',
                    'ready': False,
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'agent_id': agent_id,
                'status': 'error',
                'ready': False,
                'error': str(e)
            }
    
    def get_agent_capabilities(self, agent_id: str, use_direct_tools: bool = True) -> Dict[str, Any]:
        """
        Get the capabilities of a physics agent
        
        Args:
            agent_id: Agent type
            use_direct_tools: Whether agent uses direct MCP tools
            
        Returns:
            Dict containing agent capabilities
        """
        try:
            params = {"use_direct_tools": use_direct_tools}
            
            response = self.session.get(
                f"{self.base_url}/agent/{agent_id}/capabilities",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'error': f"HTTP {response.status_code}",
                    'capabilities': {}
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'capabilities': {}
            }
    
    def list_available_agents(self) -> Dict[str, Any]:
        """
        List all available physics agents
        
        Returns:
            Dict containing available agents and their information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/agents/list",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                # Return fallback data
                return {
                    'available_agents': [
                        {
                            'agent_id': 'forces_agent',
                            'name': 'Forces Agent',
                            'description': 'Handles force analysis, free body diagrams, and Newton\'s laws'
                        },
                        {
                            'agent_id': 'kinematics_agent',
                            'name': 'Kinematics Agent', 
                            'description': 'Handles motion analysis, projectile motion, and kinematics equations'
                        },
                        {
                            'agent_id': 'math_agent',
                            'name': 'Math Agent',
                            'description': 'Handles mathematical calculations, algebra, and computational problems'
                        },
                        {
                            'agent_id': 'momentum_agent',
                            'name': 'Momentum Agent',
                            'description': 'Handles momentum, impulse, and collision problems'
                        },
                        {
                            'agent_id': 'energy_agent',
                            'name': 'Energy Agent',
                            'description': 'Handles work, energy, power, and conservation of energy problems'
                        },
                        {
                            'agent_id': 'angular_motion_agent',
                            'name': 'Angular Motion Agent',
                            'description': 'Handles rotational motion, angular momentum, and torque problems'
                        }
                    ],
                    'active_agents': []
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'available_agents': [],
                'active_agents': [],
                'error': str(e)
            }
    
    def remove_agent(self, agent_id: str, use_direct_tools: bool = True) -> Dict[str, Any]:
        """
        Remove an agent from the server
        
        Args:
            agent_id: Agent type to remove
            use_direct_tools: Whether agent uses direct MCP tools
            
        Returns:
            Dict containing operation result
        """
        try:
            params = {"use_direct_tools": use_direct_tools}
            
            response = self.session.delete(
                f"{self.base_url}/agent/{agent_id}",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                st.session_state[f'agent_{agent_id}_ready'] = False
                return response.json()
            else:
                return {
                    'error': f"HTTP {response.status_code}",
                    'message': 'Failed to remove agent'
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'error': str(e),
                'message': 'Connection error'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API server health
        
        Returns:
            Dict containing server health status
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state['api_connected'] = True
                return result
            else:
                st.session_state['api_connected'] = False
                return {
                    'status': 'unhealthy',
                    'error': f"HTTP {response.status_code}"
                }
                
        except requests.exceptions.RequestException as e:
            st.session_state['api_connected'] = False
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def send_message(self, agent_id: str, message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Send a message to an agent and get response
        This is a convenience method that wraps solve_problem
        
        Args:
            agent_id: Agent type
            message: User message/problem
            user_id: User identifier (for logging/tracking)
            
        Returns:
            Dict containing agent response
        """
        # Get conversation context
        context = self._get_conversation_context(agent_id)
        
        # Solve the problem using the agent
        result = self.solve_problem(agent_id, message, context)
        
        # Return the API response directly, just add status for compatibility
        if result.get('success'):
            # Return the full API response with additional status field
            result['status'] = 'success'
            return result
        else:
            return {
                'success': False,
                'status': 'error',
                'content': f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}",
                'agent_id': agent_id,
                'error': result.get('error')
            }
    
    def _get_conversation_context(self, agent_id: str = None) -> Optional[Dict[str, Any]]:
        """Get recent conversation context for the agent"""
        # Use agent-specific chat history if agent_id provided
        if agent_id:
            chat_history_key = f'chat_history_{agent_id}'
            chat_history = st.session_state.get(chat_history_key, [])
        else:
            chat_history = st.session_state.get('chat_history', [])
        
        if not chat_history:
            return None
        
        # Return recent context
        recent_messages = chat_history[-5:]  # Last 5 messages
        
        return {
            'recent_conversation': [
                {
                    'role': msg.get('role'),
                    'content': msg.get('content'),
                    'timestamp': msg.get('timestamp')
                }
                for msg in recent_messages
            ],
            'total_messages': len(chat_history)
        }
    
    def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """
        Get formatted agent information for UI display
        
        Args:
            agent_id: Agent type
            
        Returns:
            Dict containing agent display information
        """
        agent_info_map = {
            'forces_agent': {
                'name': 'Forces Agent',
                'icon': '⚖️',
                'description': 'Specialized in force analysis, free body diagrams, and Newton\'s laws of motion',
                'capabilities': [
                    'Force vector addition and resolution',
                    'Free body diagram creation',
                    'Newton\'s laws applications',
                    'Equilibrium analysis',
                    'Spring force calculations'
                ]
            },
            'kinematics_agent': {
                'name': 'Kinematics Agent',
                'icon': '🚀',
                'description': 'Expert in motion analysis, projectile motion, and kinematics equations',
                'capabilities': [
                    'Projectile motion analysis',
                    'Free fall calculations',
                    'Constant acceleration problems',
                    'Uniform motion analysis',
                    'Relative motion problems'
                ]
            },
            'math_agent': {
                'name': 'Math Agent',
                'icon': '🔢',
                'description': 'Expert in mathematical calculations, algebra, and computational problems',
                'capabilities': [
                    'Algebraic equation solving',
                    'Calculus operations',
                    'Numerical computations',
                    'Mathematical modeling',
                    'Statistical analysis'
                ]
            },
            'momentum_agent': {
                'name': 'Momentum Agent',
                'icon': '💥',
                'description': 'Specialized in momentum, impulse, and collision analysis',
                'capabilities': [
                    'Momentum conservation problems',
                    'Impulse-momentum theorem',
                    'Elastic and inelastic collisions',
                    'Center of mass calculations',
                    'Explosion and recoil problems'
                ]
            },
            'energy_agent': {
                'name': 'Energy Agent',
                'icon': '⚡',
                'description': 'Expert in work, energy, power, and conservation principles',
                'capabilities': [
                    'Work-energy theorem applications',
                    'Kinetic and potential energy',
                    'Conservation of energy problems',
                    'Power calculations',
                    'Energy transformation analysis'
                ]
            },
            'angular_motion_agent': {
                'name': 'Angular Motion Agent',
                'icon': '🌀',
                'description': 'Specialized in rotational motion, angular momentum, and torque',
                'capabilities': [
                    'Rotational kinematics',
                    'Torque and angular acceleration',
                    'Angular momentum conservation',
                    'Moment of inertia calculations',
                    'Rolling motion problems'
                ]
            }
        }
        
        return agent_info_map.get(agent_id, {
            'name': 'Physics Agent',
            'icon': '🤖',
            'description': 'General physics problem solver',
            'capabilities': ['Physics problem solving']
        })