import asyncio
import json
import requests
import websockets
import time
import logging
from typing import Dict, Optional, Any, List
import streamlit as st
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Client for communicating with MCP (Model Context Protocol) server"""
    
    def __init__(self):
        self.base_url = Config.MCP_SERVER_URL
        self.websocket_url = Config.MCP_WEBSOCKET_URL
        self.api_key = Config.MCP_API_KEY
        self.session = requests.Session()
        
        # Set up headers
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
        
        # Initialize database client for logging
        self.db_client = None
        if Config.DATABASE_LOGGING_ENABLED:
            try:
                from .database_client import DatabaseAPIClient
                self.db_client = DatabaseAPIClient(Config.DATABASE_API_URL)
                logger.info("✅ Database logging enabled for MCP client")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize database logging: {e}")
    
    def _log_mcp_interaction(self, 
                           operation: str,
                           agent_id: str,
                           request_data: Dict[str, Any],
                           response_data: Dict[str, Any],
                           execution_time_ms: int,
                           success: bool) -> None:
        """Log MCP interaction to database if logging is enabled"""
        if not self.db_client or not self.db_client.is_connected:
            return
            
        try:
            # Get user info for logging
            user_info = st.session_state.get('user_info', {})
            user_id = user_info.get('id', 'anonymous')
            
            # Prepare metadata
            metadata = {
                'operation': operation,
                'mcp_server_url': self.base_url,
                'websocket_url': self.websocket_url,
                'request_data': request_data,
                'response_status': 'success' if success else 'error',
                'execution_time_ms': execution_time_ms,
                'mcp_client_version': '1.0.0',
                'streamlit_session_id': st.session_state.get('session_id'),
                'timestamp': time.time(),
                'has_api_key': bool(self.api_key)
            }
            
            # Format message for database logging
            message = f"MCP {operation}: {request_data.get('message', request_data.get('content', 'N/A'))}"
            response = response_data.get('content', str(response_data))
            
            # Log to database
            self.db_client.log_interaction(
                user_id=user_id,
                agent_type=f"mcp_{agent_id}",
                message=message,
                response=response,
                session_id=st.session_state.get('db_session_id'),
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to log MCP interaction: {e}")
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents from MCP server"""
        try:
            response = self.session.get(f"{self.base_url}/api/agents/available")
            response.raise_for_status()
            return response.json().get('agents', [])
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch agents: {str(e)}")
            # Return default agents from config as fallback
            return [
                {
                    'id': agent_id,
                    'name': agent_info['name'],
                    'description': agent_info['description'],
                    'status': 'offline'
                }
                for agent_id, agent_info in Config.PHYSICS_AGENTS.items()
            ]
    
    def select_agent(self, agent_id: str, user_id: str) -> Dict[str, Any]:
        """Select and initialize an agent for the user"""
        try:
            payload = {
                'agent_id': agent_id,
                'user_id': user_id,
                'session_id': st.session_state.get('session_id', 'default')
            }
            
            response = self.session.post(f"{self.base_url}/api/agents/select", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to select agent: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user with MCP server"""
        try:
            payload = {
                'username': username,
                'password': password
            }
            
            response = self.session.post(f"{self.base_url}/api/auth/login", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': str(e)}
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify authentication token"""
        try:
            headers = {'Authorization': f'Bearer {token}'}
            response = self.session.get(f"{self.base_url}/api/auth/verify", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {'status': 'error', 'message': str(e)}
    
    async def send_message(self, agent_id: str, message: str, user_id: str) -> Dict[str, Any]:
        """Send message to agent via WebSocket"""
        message_data = {
            'agent': agent_id,
            'content': message,
            'user_id': user_id,
            'session_id': st.session_state.get('session_id', 'default'),
            'timestamp': str(time.time())
        }
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                # Send message
                await websocket.send(json.dumps(message_data))
                
                # Wait for response
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"WebSocket error: {str(e)}",
                'content': f"I apologize, but I'm having trouble connecting to the physics tutoring service. This is a demo response showing that your message '{message}' was received by the interface."
            }
    
    def send_message_sync(self, agent_id: str, message: str, user_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for sending messages"""
        start_time = time.time()
        request_data = {
            'agent_id': agent_id,
            'message': message,
            'user_id': user_id
        }
        
        try:
            # For Streamlit compatibility, we'll use HTTP API instead of WebSocket
            # In production, you might want to use WebSocket with proper async handling
            response_data = self._send_http_message(agent_id, message, user_id)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            success = response_data.get('status') == 'success'
            
            # Log interaction to database
            self._log_mcp_interaction(
                operation="send_message",
                agent_id=agent_id,
                request_data=request_data,
                response_data=response_data,
                execution_time_ms=execution_time_ms,
                success=success
            )
            
            return response_data
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_response = {
                'status': 'error',
                'message': str(e),
                'content': f"I apologize, but I'm having trouble connecting to the physics tutoring service. This is a demo response for your message: '{message}'"
            }
            
            # Log failed interaction
            self._log_mcp_interaction(
                operation="send_message",
                agent_id=agent_id,
                request_data=request_data,
                response_data=error_response,
                execution_time_ms=execution_time_ms,
                success=False
            )
            
            return error_response
    
    def _send_http_message(self, agent_id: str, message: str, user_id: str) -> Dict[str, Any]:
        """Send message via HTTP API (fallback)"""
        try:
            payload = {
                'agent_id': agent_id,
                'message': message,
                'user_id': user_id,
                'session_id': st.session_state.get('session_id', 'default'),
                'context': self._get_conversation_context()
            }
            
            response = self.session.post(f"{self.base_url}/api/chat/message", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Return demo response for development/testing
            return self._generate_demo_response(agent_id, message)
    
    def _get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get recent conversation context for the agent"""
        chat_history = st.session_state.get('chat_history', [])
        
        # Return last 10 messages for context
        context = []
        for msg in chat_history[-10:]:
            context.append({
                'role': msg.get('role'),
                'content': msg.get('content'),
                'timestamp': msg.get('timestamp')
            })
        
        return context
    
    def _generate_demo_response(self, agent_id: str, message: str) -> Dict[str, Any]:
        """Generate demo response for development/testing"""
        agent_info = Config.PHYSICS_AGENTS.get(agent_id, {})
        agent_name = agent_info.get('name', 'Physics Agent')
        
        # Simple keyword-based demo responses
        demo_responses = {
            'kinematics': {
                'keywords': ['velocity', 'acceleration', 'motion', 'speed', 'distance'],
                'response': f"As the {agent_name}, I can help with motion problems! For velocity and acceleration questions, I'd analyze the given data and apply kinematic equations like v = u + at or s = ut + ½at²."
            },
            'forces': {
                'keywords': ['force', 'newton', 'friction', 'tension', 'weight'],
                'response': f"Great question about forces! As the {agent_name}, I'd help you draw free body diagrams and apply Newton's laws. Remember F = ma is fundamental to solving force problems."
            },
            'energy': {
                'keywords': ['energy', 'work', 'power', 'kinetic', 'potential'],
                'response': f"Energy problems are my specialty! As the {agent_name}, I'd help you understand work-energy theorem (W = ΔKE) and conservation of mechanical energy."
            },
            'momentum': {
                'keywords': ['momentum', 'collision', 'impulse', 'conservation'],
                'response': f"Momentum questions are interesting! As the {agent_name}, I'd apply conservation of momentum (p = mv) and impulse-momentum theorem to solve collision problems."
            },
            'rotation': {
                'keywords': ['rotation', 'torque', 'angular', 'moment', 'inertia'],
                'response': f"Rotational motion can be tricky! As the {agent_name}, I'd help with torque calculations (τ = rF sin θ) and angular kinematics."
            },
            'math_helper': {
                'keywords': ['vector', 'trigonometry', 'algebra', 'unit', 'conversion'],
                'response': f"Math support coming right up! As the {agent_name}, I'd help with vector operations, trigonometric calculations, and unit conversions needed for physics."
            }
        }
        
        agent_demo = demo_responses.get(agent_id, {})
        keywords = agent_demo.get('keywords', [])
        
        # Check if message contains relevant keywords
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in keywords):
            response_content = agent_demo.get('response', '')
        else:
            response_content = f"Thank you for your question! As the {agent_name}, I'm ready to help with {agent_info.get('description', 'physics problems')}. Could you provide more specific details about what you'd like to learn?"
        
        response_content += f"\n\n*Note: This is a demo response. In the full implementation, I would connect to the MCP server running Ollama models to provide detailed, personalized physics tutoring.*"
        
        return {
            'status': 'success',
            'content': response_content,
            'agent_id': agent_id,
            'agent_name': agent_name
        }
    
    def upload_image(self, agent_id: str, image_file, user_id: str) -> Dict[str, Any]:
        """Upload and analyze physics problem image"""
        try:
            files = {'image': image_file}
            data = {
                'agent_id': agent_id,
                'user_id': user_id,
                'session_id': st.session_state.get('session_id', 'default')
            }
            
            response = self.session.post(
                f"{self.base_url}/api/vision/analyze", 
                files=files, 
                data=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': str(e),
                'content': "I can see your image, but I'm currently unable to analyze it. In the full implementation, I would use computer vision to read and solve physics problems from uploaded images."
            }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent"""
        try:
            response = self.session.get(f"{self.base_url}/api/agents/{agent_id}/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'status': 'unknown',
                'message': str(e),
                'online': False
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if MCP server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return {'status': 'healthy', 'server': 'online'}
        except requests.exceptions.RequestException as e:
            return {'status': 'unhealthy', 'server': 'offline', 'error': str(e)}