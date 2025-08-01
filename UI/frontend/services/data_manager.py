import streamlit as st
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import Config

class SessionDataManager:
    """Manages session data, user progress, and conversation history"""
    
    def __init__(self):
        self.session_id = self._get_or_create_session_id()
    
    def _get_or_create_session_id(self) -> str:
        """Get or create a unique session ID"""
        if 'session_id' not in st.session_state:
            st.session_state['session_id'] = f"session_{int(time.time())}_{hash(str(time.time()))}"
        return st.session_state['session_id']
    
    def save_chat_message(self, message: Dict[str, Any]):
        """Save a chat message to session history"""
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Add metadata to message
        message_with_metadata = {
            **message,
            'session_id': self.session_id,
            'message_id': f"msg_{len(st.session_state['chat_history'])}_{int(time.time())}",
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        st.session_state['chat_history'].append(message_with_metadata)
        
        # Limit chat history size
        if len(st.session_state['chat_history']) > Config.MAX_CHAT_HISTORY:
            st.session_state['chat_history'] = st.session_state['chat_history'][-Config.MAX_CHAT_HISTORY:]
    
    def get_chat_history(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get chat history, optionally filtered by agent"""
        history = st.session_state.get('chat_history', [])
        
        if agent_id:
            # Filter messages for specific agent
            return [msg for msg in history if msg.get('agent_id') == agent_id]
        
        return history
    
    def clear_chat_history(self):
        """Clear all chat history"""
        st.session_state['chat_history'] = []
    
    def save_user_progress(self, topic: str, difficulty: int, success: bool):
        """Save user learning progress"""
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
        
        # Update average difficulty
        progress['avg_difficulty'] = (
            (progress['avg_difficulty'] * (progress['attempts'] - 1) + difficulty) / 
            progress['attempts']
        )
        progress['last_attempt'] = datetime.now().isoformat()
    
    def get_user_progress(self) -> Dict[str, Any]:
        """Get user learning progress"""
        return st.session_state.get('user_progress', {})
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics"""
        chat_history = st.session_state.get('chat_history', [])
        user_progress = st.session_state.get('user_progress', {})
        
        # Calculate statistics
        total_messages = len(chat_history)
        user_messages = len([msg for msg in chat_history if msg.get('role') == 'user'])
        agent_messages = len([msg for msg in chat_history if msg.get('role') == 'assistant'])
        
        # Get agents used
        agents_used = set()
        for msg in chat_history:
            if msg.get('agent_id'):
                agents_used.add(msg['agent_id'])
        
        # Calculate session duration
        if chat_history:
            start_time = min(msg.get('timestamp', time.time()) for msg in chat_history)
            end_time = max(msg.get('timestamp', time.time()) for msg in chat_history)
            duration_minutes = (end_time - start_time) / 60
        else:
            duration_minutes = 0
        
        return {
            'session_id': self.session_id,
            'total_messages': total_messages,
            'user_messages': user_messages,
            'agent_messages': agent_messages,
            'agents_used': list(agents_used),
            'duration_minutes': round(duration_minutes, 1),
            'topics_covered': len(user_progress),
            'total_attempts': sum(p.get('attempts', 0) for p in user_progress.values()),
            'total_successes': sum(p.get('successes', 0) for p in user_progress.values()),
            'success_rate': self._calculate_success_rate(user_progress)
        }
    
    def _calculate_success_rate(self, user_progress: Dict) -> float:
        """Calculate overall success rate"""
        total_attempts = sum(p.get('attempts', 0) for p in user_progress.values())
        total_successes = sum(p.get('successes', 0) for p in user_progress.values())
        
        if total_attempts == 0:
            return 0.0
        
        return round((total_successes / total_attempts) * 100, 1)
    
    def save_session_data(self, key: str, data: Any):
        """Save arbitrary session data"""
        if 'session_data' not in st.session_state:
            st.session_state['session_data'] = {}
        
        st.session_state['session_data'][key] = {
            'data': data,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
    
    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Get saved session data"""
        session_data = st.session_state.get('session_data', {})
        
        if key in session_data:
            return session_data[key]['data']
        
        return default
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export all session data for analysis or backup"""
        return {
            'session_id': self.session_id,
            'user_info': st.session_state.get('user_info', {}),
            'chat_history': st.session_state.get('chat_history', []),
            'user_progress': st.session_state.get('user_progress', {}),
            'session_data': st.session_state.get('session_data', {}),
            'statistics': self.get_session_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
    
    def clear_session(self):
        """Clear all session data"""
        keys_to_clear = [
            'chat_history',
            'user_progress', 
            'session_data',
            'selected_agent',
            'session_id'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def get_recent_topics(self, limit: int = 5) -> List[str]:
        """Get recently discussed physics topics"""
        chat_history = st.session_state.get('chat_history', [])
        
        # Extract topics from recent messages (simple keyword matching)
        physics_topics = [
            'kinematics', 'velocity', 'acceleration', 'motion',
            'force', 'newton', 'friction', 'tension',
            'energy', 'work', 'power', 'conservation',
            'momentum', 'collision', 'impulse',
            'rotation', 'torque', 'angular', 'inertia',
            'wave', 'frequency', 'amplitude',
            'electricity', 'circuit', 'current', 'voltage',
            'magnetism', 'field', 'flux'
        ]
        
        recent_topics = []
        for msg in reversed(chat_history[-20:]):  # Look at last 20 messages
            content = msg.get('content', '').lower()
            for topic in physics_topics:
                if topic in content and topic not in recent_topics:
                    recent_topics.append(topic)
                    if len(recent_topics) >= limit:
                        return recent_topics
        
        return recent_topics