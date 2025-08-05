#DO NOT DELETE THIS LINE streamlit run app.py --server.port 8501 --server.address 127.0.0.1
import streamlit as st
from config import Config, PHYSICS_CONSTANTS
from components.auth import AuthManager
from components.chat import ChatInterface
from components.agents import AgentManager
from services.data_manager import SessionDataManager

def initialize_session_state():
    """Initialize session state variables"""
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'user_info' not in st.session_state:
        st.session_state['user_info'] = {}
    if 'selected_agent' not in st.session_state:
        st.session_state['selected_agent'] = None
    # Agent-specific chat histories are initialized as needed
    if 'session_data' not in st.session_state:
        st.session_state['session_data'] = {}

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Initialize managers
    auth_manager = AuthManager()
    session_manager = SessionDataManager()
    
    # Check authentication status
    if st.session_state['authentication_status'] is None:
        # Show login page
        auth_manager.render_login()
        return
    elif st.session_state['authentication_status'] is False:
        st.error('Username/password is incorrect')
        auth_manager.render_login()
        return
    elif st.session_state['authentication_status']:
        # User is authenticated, show main interface
        render_main_interface(session_manager)

def render_main_interface(session_manager):
    """Render the main physics assistant interface"""
    st.title("🔬 Physics Assistant")
    st.markdown(f"Welcome, **{st.session_state['username']}**!")
    
    # Sidebar for agent selection and tools
    with st.sidebar:
        st.header("Physics Agents")
        
        # Agent selection
        agent_manager = AgentManager()
        selected_agent = agent_manager.render_agent_selector()
        
        if selected_agent != st.session_state['selected_agent']:
            st.session_state['selected_agent'] = selected_agent
        
        # Physics constants reference
        with st.expander("📋 Physics Constants"):
            for constant, value in PHYSICS_CONSTANTS.items():
                st.write(f"**{constant}**: {value}")
        
        # Session statistics
        with st.expander("📊 Session Stats"):
            # Count messages for current agent
            selected_agent = st.session_state.get('selected_agent')
            if selected_agent:
                chat_history_key = f'chat_history_{selected_agent}'
                message_count = len(st.session_state.get(chat_history_key, []))
            else:
                message_count = 0
            st.metric("Messages", message_count)
            st.metric("Agent", selected_agent or "None")
        
        # Logout button
        if st.button("🚪 Logout"):
            session_manager.clear_session()
            st.session_state['authentication_status'] = None
            st.rerun()
    
    # Main chat interface  
    current_agent = st.session_state.get('selected_agent')
    
    if current_agent:
        try:
            chat_interface = ChatInterface(current_agent)
            chat_interface.render()  
        except Exception as e:
            st.error(f"❌ ERROR creating ChatInterface: {str(e)}")
            st.exception(e)
    else:
        st.info("👈 Please select a physics agent from the sidebar to start chatting!")
        
        # Display available agents
        st.subheader("Available Physics Agents")
        cols = st.columns(2)
        
        # Get agents from AgentManager instead of Config
        available_agents = agent_manager.get_all_agents()
        
        for i, (agent_id, agent_info) in enumerate(available_agents.items()):
            with cols[i % 2]:
                st.markdown(f"""
                **{agent_info['icon']} {agent_info['name']}**  
                {agent_info['description']}
                """)

if __name__ == "__main__":
    main()
