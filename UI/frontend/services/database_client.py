"""
Database API Client for Physics Assistant UI
Integrates Streamlit frontend with database API for comprehensive logging
"""
import requests
import json
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseAPIClient:
    """Client for interacting with Physics Assistant Database API"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.api_base_url = api_base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 10  # 10 second timeout
        
        # Test connection on initialization
        self.is_connected = self._test_connection()
        
    def _test_connection(self) -> bool:
        """Test connection to database API"""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Database API connected: {health_data.get('healthy_databases', 'unknown')}")
                return True
            else:
                logger.warning(f"⚠️ Database API unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Database API not available: {e}")
            return False
    
    def log_interaction(self, 
                       user_id: str,
                       agent_type: str,
                       message: str,
                       response: str,
                       session_id: Optional[str] = None,
                       execution_time_ms: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Log user interaction to database via API"""
        if not self.is_connected:
            logger.warning("Database API not connected, skipping interaction logging")
            return None
            
        try:
            payload = {
                "user_id": user_id,
                "agent_type": agent_type,
                "message": message,
                "response": response,
                "session_id": session_id,
                "execution_time_ms": execution_time_ms,
                "metadata": metadata or {}
            }
            
            response = self.session.post(
                f"{self.api_base_url}/interactions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                interaction_id = result.get('interaction_id')
                logger.info(f"✅ Logged interaction: {interaction_id}")
                return interaction_id
            else:
                logger.error(f"❌ Failed to log interaction: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error logging interaction: {e}")
            return None
    
    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new user session in database"""
        if not self.is_connected:
            return None
            
        try:
            payload = {
                "user_id": user_id,
                "metadata": metadata or {}
            }
            
            response = self.session.post(
                f"{self.api_base_url}/sessions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                session_id = result.get('session_id')
                logger.info(f"✅ Created session: {session_id}")
                return session_id
            else:
                logger.error(f"❌ Failed to create session: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating session: {e}")
            return None
    
    def get_physics_concepts(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get physics concepts from knowledge graph"""
        if not self.is_connected:
            return []
            
        try:
            url = f"{self.api_base_url}/physics/concepts"
            params = {"category": category} if category else {}
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                result = response.json()
                concepts = result.get('concepts', [])
                logger.info(f"✅ Retrieved {len(concepts)} physics concepts")
                return concepts
            else:
                logger.error(f"❌ Failed to get concepts: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error getting concepts: {e}")
            return []
    
    def get_analytics_summary(self, days: int = 7, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics summary from database"""
        if not self.is_connected:
            return {}
            
        try:
            params = {"days": days}
            if user_id:
                params["user_id"] = user_id
                
            response = self.session.get(
                f"{self.api_base_url}/analytics/summary",
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Retrieved analytics for {days} days")
                return result
            else:
                logger.error(f"❌ Failed to get analytics: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting analytics: {e}")
            return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get database health status"""
        try:
            response = self.session.get(f"{self.api_base_url}/health")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

class EnhancedSessionDataManager:
    """Enhanced session data manager with database integration"""
    
    def __init__(self, api_base_url: str = "http://localhost:8001"):
        self.session_id = self._get_or_create_session_id()
        self.db_client = DatabaseAPIClient(api_base_url)
        
        # Initialize user session in database if API is available
        if self.db_client.is_connected:
            self._sync_with_database()
    
    def _get_or_create_session_id(self) -> str:
        """Get or create a unique session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = str(uuid.uuid4())
        return st.session_state['session_id']
    
    def _sync_with_database(self):
        """Sync session with database"""
        try:
            # Get or create user ID for database operations
            user_info = st.session_state.get('user_info', {})
            if user_info and user_info.get('id'):
                user_id = user_info['id']
                
                # Create database session if needed
                db_session_id = self.db_client.create_session(
                    user_id=user_id,
                    metadata={
                        'streamlit_session_id': self.session_id,
                        'user_agent': st.context.headers.get('user-agent', 'unknown') if hasattr(st, 'context') else 'unknown',
                        'created_via': 'streamlit_ui'
                    }
                )
                
                if db_session_id:
                    st.session_state['db_session_id'] = db_session_id
                    logger.info(f"✅ Database session synced: {db_session_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync with database: {e}")
    
    def save_chat_message(self, message: Dict[str, Any]):
        """Save chat message to session and database"""
        # Save to Streamlit session state (existing functionality)
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        message_with_metadata = {
            **message,
            'session_id': self.session_id,
            'message_id': f"msg_{len(st.session_state['chat_history'])}_{int(time.time())}",
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        st.session_state['chat_history'].append(message_with_metadata)
        
        # Log interaction to database if this is a complete user-assistant exchange
        if (message.get('role') == 'assistant' and 
            message.get('agent_id') and 
            self.db_client.is_connected):
            
            self._log_interaction_to_db(message_with_metadata)
        
        # Limit chat history size
        from config import Config
        if len(st.session_state['chat_history']) > Config.MAX_CHAT_HISTORY:
            st.session_state['chat_history'] = st.session_state['chat_history'][-Config.MAX_CHAT_HISTORY:]
    
    def _log_interaction_to_db(self, assistant_message: Dict[str, Any]):
        """Log complete interaction to database"""
        try:
            # Find the corresponding user message
            chat_history = st.session_state.get('chat_history', [])
            user_message = None
            
            # Look for the most recent user message
            for msg in reversed(chat_history[:-1]):  # Exclude the current assistant message
                if msg.get('role') == 'user':
                    user_message = msg
                    break
            
            if not user_message:
                logger.warning("⚠️ No user message found for interaction logging")
                return
            
            # Get user info for database logging
            user_info = st.session_state.get('user_info', {})
            if not user_info.get('id'):
                logger.warning("⚠️ No user ID available for interaction logging")
                return
            
            # Calculate response time if available
            execution_time_ms = None
            if ('response_time' in assistant_message and 
                isinstance(assistant_message['response_time'], (int, float))):
                execution_time_ms = int(assistant_message['response_time'] * 1000)
            
            # Prepare metadata
            metadata = {
                'streamlit_session_id': self.session_id,
                'user_message_id': user_message.get('message_id'),
                'assistant_message_id': assistant_message.get('message_id'),
                'timestamp_user': user_message.get('timestamp'),
                'timestamp_assistant': assistant_message.get('timestamp'),
                'agent_icon': assistant_message.get('agent_icon'),
                **(assistant_message.get('metadata', {}))
            }
            
            # Log to database
            interaction_id = self.db_client.log_interaction(
                user_id=user_info['id'],
                agent_type=assistant_message.get('agent_id', 'unknown'),
                message=user_message.get('content', ''),
                response=assistant_message.get('content', ''),
                session_id=st.session_state.get('db_session_id'),
                execution_time_ms=execution_time_ms,
                metadata=metadata
            )
            
            if interaction_id:
                # Store interaction ID in the message for future reference
                assistant_message['db_interaction_id'] = interaction_id
                
        except Exception as e:
            logger.error(f"❌ Failed to log interaction to database: {e}")
    
    def get_chat_history(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get chat history, optionally filtered by agent"""
        history = st.session_state.get('chat_history', [])
        
        if agent_id:
            return [msg for msg in history if msg.get('agent_id') == agent_id]
        
        return history
    
    def get_physics_concepts(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get physics concepts from knowledge graph via database API"""
        return self.db_client.get_physics_concepts(category)
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for current session"""
        user_info = st.session_state.get('user_info', {})
        if user_info.get('id') and self.db_client.is_connected:
            return self.db_client.get_analytics_summary(user_id=user_info['id'])
        return {}
    
    def get_database_health(self) -> Dict[str, Any]:
        """Get database health status"""
        return self.db_client.get_health_status()
    
    # Maintain compatibility with existing methods
    def clear_chat_history(self):
        """Clear all chat history"""
        st.session_state['chat_history'] = []
    
    def save_user_progress(self, topic: str, difficulty: int, success: bool):
        """Save user learning progress (existing functionality)"""
        if 'user_progress' not in st.session_state:
            st.session_state['user_progress'] = {}
        
        if topic not in st.session_state['user_progress']:
            st.session_state['user_progress'][topic] = {
                'attempts': 0,
                'successes': 0,
                'avg_difficulty': 0,
                'last_attempt': None,
                'topics_covered': []
            }
        
        progress = st.session_state['user_progress'][topic]
        progress['attempts'] += 1
        if success:
            progress['successes'] += 1
        
        progress['avg_difficulty'] = (
            (progress['avg_difficulty'] * (progress['attempts'] - 1) + difficulty) / 
            progress['attempts']
        )
        progress['last_attempt'] = datetime.now().isoformat()
    
    def get_user_progress(self) -> Dict[str, Any]:
        """Get user learning progress"""
        return st.session_state.get('user_progress', {})
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data including database connectivity info"""
        base_data = {
            'session_id': self.session_id,
            'db_session_id': st.session_state.get('db_session_id'),
            'user_info': st.session_state.get('user_info', {}),
            'chat_history': st.session_state.get('chat_history', []),
            'user_progress': st.session_state.get('user_progress', {}),
            'export_timestamp': datetime.now().isoformat(),
            'database_connected': self.db_client.is_connected
        }
        
        # Add database health info if connected
        if self.db_client.is_connected:
            base_data['database_health'] = self.get_database_health()
        
        return base_data

# Factory function for easy migration from existing code
def get_data_manager(api_base_url: str = "http://localhost:8001") -> EnhancedSessionDataManager:
    """Get enhanced data manager with database integration"""
    return EnhancedSessionDataManager(api_base_url)