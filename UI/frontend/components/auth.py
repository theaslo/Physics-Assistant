import streamlit as st
import yaml
import bcrypt
from typing import Dict, Optional
from config import Config

class AuthManager:
    """Handles user authentication and session management"""
    
    def __init__(self):
        self.config = self._load_auth_config()
    
    def _load_auth_config(self) -> Dict:
        """Load authentication configuration"""
        # For demo purposes, using hardcoded users
        # In production, this would come from a database or external auth service
        return {
            'usernames': {
                'student1': {
                    'name': 'Physics Student 1',
                    'password': self._hash_password('password123'),
                    'email': 'student1@university.edu',
                    'role': 'student',
                    'course': 'PHYS101'
                },
                'student2': {
                    'name': 'Physics Student 2',
                    'password': self._hash_password('password123'),
                    'email': 'student2@university.edu',
                    'role': 'student',
                    'course': 'PHYS101'
                },
                'instructor': {
                    'name': 'Dr. Physics',
                    'password': self._hash_password('instructor123'),
                    'email': 'instructor@university.edu',
                    'role': 'instructor',
                    'course': 'PHYS101'
                }
            }
        }
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def authenticate(self, username: str, password: str) -> tuple[bool, Optional[Dict]]:
        """
        Authenticate a user
        Returns: (success, user_info)
        """
        if username in self.config['usernames']:
            user_data = self.config['usernames'][username]
            if self._verify_password(password, user_data['password']):
                # Remove password from returned user info
                user_info = {k: v for k, v in user_data.items() if k != 'password'}
                user_info['username'] = username
                return True, user_info
        return False, None
    
    def render_login(self):
        """Render the login interface"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🔬 Physics Assistant</h1>
            <p style='font-size: 1.2em; color: #666;'>Interactive Physics Tutoring System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create centered login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Student Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your student ID")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                remember_me = st.checkbox("Remember me")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                if submit_button:
                    if username and password:
                        success, user_info = self.authenticate(username, password)
                        
                        if success:
                            st.session_state['authentication_status'] = True
                            st.session_state['username'] = username
                            st.session_state['user_info'] = user_info
                            st.success(f"Welcome, {user_info['name']}!")
                            st.rerun()
                        else:
                            st.session_state['authentication_status'] = False
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
            
            # Demo credentials info
            st.markdown("---")
            st.markdown("**Demo Credentials:**")
            st.markdown("""
            - **Student**: username: `student1`, password: `password123`
            - **Student**: username: `student2`, password: `password123`
            - **Instructor**: username: `instructor`, password: `instructor123`
            """)
            
            # Help and support links
            st.markdown("---")
            col_help1, col_help2 = st.columns(2)
            with col_help1:
                if st.button("🔑 Forgot Password?", use_container_width=True):
                    st.info("Please contact your instructor or IT support.")
            with col_help2:
                if st.button("❓ Need Help?", use_container_width=True):
                    st.info("Contact physics-help@university.edu")
    
    def logout(self):
        """Clear authentication state"""
        st.session_state['authentication_status'] = None
        st.session_state['username'] = None
        st.session_state['user_info'] = {}
        st.session_state['selected_agent'] = None
        st.session_state['chat_history'] = []
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return st.session_state.get('authentication_status') == True
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current user information"""
        if self.is_authenticated():
            return st.session_state.get('user_info')
        return None
    
    def has_role(self, role: str) -> bool:
        """Check if current user has specific role"""
        user_info = self.get_current_user()
        if user_info:
            return user_info.get('role') == role
        return False