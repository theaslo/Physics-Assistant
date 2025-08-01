import streamlit as st
from components.auth import AuthManager
from services.data_manager import SessionDataManager
from config import Config

def main():
    """Settings and preferences page"""
    st.set_page_config(
        page_title=f"{Config.PAGE_TITLE} - Settings",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Check authentication
    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        st.warning("Please log in to access settings.")
        if st.button("Go to Login"):
            st.switch_page("pages/login.py")
        return
    
    session_manager = SessionDataManager()
    user_info = st.session_state.get('user_info', {})
    
    st.title("⚙️ Settings & Preferences")
    st.markdown(f"**User:** {user_info.get('name', 'Unknown User')}")
    
    # User Profile Section
    render_user_profile(user_info)
    
    # UI Preferences
    render_ui_preferences()
    
    # Learning Preferences
    render_learning_preferences()
    
    # Data Management
    render_data_management(session_manager)
    
    # System Information
    render_system_info()

def render_user_profile(user_info):
    """Render user profile settings"""
    st.subheader("👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Full Name", value=user_info.get('name', ''), disabled=True)
        st.text_input("Email", value=user_info.get('email', ''), disabled=True)
    
    with col2:
        st.text_input("Username", value=user_info.get('username', ''), disabled=True)
        st.text_input("Course", value=user_info.get('course', ''), disabled=True)
        st.text_input("Role", value=user_info.get('role', '').title(), disabled=True)
    
    st.info("👆 Profile information is managed by your institution and cannot be changed here.")

def render_ui_preferences():
    """Render UI customization options"""
    st.subheader("🎨 Interface Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme preference (placeholder - Streamlit doesn't directly support custom themes)
        theme_option = st.selectbox(
            "Color Theme",
            ["Auto (System)", "Light", "Dark"],
            help="Theme preference (requires app restart)"
        )
        
        # Language preference
        language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "German"],
            help="Interface language (feature coming soon)"
        )
        
        # Chat display options
        show_timestamps = st.checkbox(
            "Show message timestamps",
            value=True,
            help="Display timestamps for chat messages"
        )
    
    with col2:
        # Math rendering options
        latex_style = st.selectbox(
            "Math Rendering Style",
            ["Standard LaTeX", "Large Text", "Inline Style"],
            help="How mathematical equations are displayed"
        )
        
        # Agent avatar display
        show_agent_avatars = st.checkbox(
            "Show agent avatars",
            value=True,
            help="Display icons for physics agents"
        )
        
        # Sidebar preferences
        default_sidebar_state = st.selectbox(
            "Default Sidebar State",
            ["Expanded", "Collapsed"],
            help="Default state of the sidebar when app loads"
        )
    
    # Save preferences
    if st.button("💾 Save UI Preferences"):
        preferences = {
            'theme': theme_option,
            'language': language,
            'show_timestamps': show_timestamps,
            'latex_style': latex_style,
            'show_agent_avatars': show_agent_avatars,
            'default_sidebar_state': default_sidebar_state
        }
        
        # In a full implementation, these would be saved to user profile
        st.session_state['ui_preferences'] = preferences
        st.success("✅ UI preferences saved!")

def render_learning_preferences():
    """Render learning-specific preferences"""
    st.subheader("🎓 Learning Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Difficulty level preference
        difficulty_level = st.slider(
            "Preferred Difficulty Level",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Beginner, 5 = Advanced"
        )
        
        # Units preference
        unit_system = st.selectbox(
            "Preferred Unit System",
            ["SI (Metric)", "Imperial", "Mixed"],
            help="Default unit system for physics problems"
        )
        
        # Explanation detail level
        explanation_detail = st.selectbox(
            "Explanation Detail Level",
            ["Brief", "Standard", "Detailed", "Step-by-step"],
            index=1,
            help="How detailed you want physics explanations"
        )
    
    with col2:
        # Learning goals
        st.markdown("**Current Learning Goals:**")
        learning_goals = st.multiselect(
            "Select topics you want to focus on:",
            ["Kinematics", "Forces & Newton's Laws", "Energy & Work", 
             "Momentum", "Rotational Motion", "Waves & Oscillations",
             "Electricity & Magnetism", "Thermodynamics"],
            default=["Kinematics", "Forces & Newton's Laws"]
        )
        
        # Study reminders
        enable_reminders = st.checkbox(
            "Enable study reminders",
            help="Get reminded to practice physics problems"
        )
        
        if enable_reminders:
            reminder_frequency = st.selectbox(
                "Reminder frequency",
                ["Daily", "Every 2 days", "Weekly"],
                index=2
            )
    
    # Save learning preferences
    if st.button("💾 Save Learning Preferences"):
        learning_prefs = {
            'difficulty_level': difficulty_level,
            'unit_system': unit_system,
            'explanation_detail': explanation_detail,
            'learning_goals': learning_goals,
            'enable_reminders': enable_reminders,
            'reminder_frequency': reminder_frequency if enable_reminders else None
        }
        
        st.session_state['learning_preferences'] = learning_prefs
        st.success("✅ Learning preferences saved!")

def render_data_management(session_manager):
    """Render data management options"""
    st.subheader("🗄️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Session Data:**")
        stats = session_manager.get_session_statistics()
        
        st.metric("Messages in History", stats['total_messages'])
        st.metric("Session Duration", f"{stats['duration_minutes']} min")
        
        # Clear session data
        if st.button("🗑️ Clear Session Data", type="secondary"):
            if st.session_state.get('confirm_clear_session'):
                session_manager.clear_session()
                st.success("✅ Session data cleared!")
                st.session_state['confirm_clear_session'] = False
                st.rerun()
            else:
                st.session_state['confirm_clear_session'] = True
                st.warning("⚠️ Click again to confirm clearing all session data.")
    
    with col2:
        st.markdown("**Privacy Settings:**")
        
        # Data collection preferences
        allow_analytics = st.checkbox(
            "Allow learning analytics collection",
            value=True,
            help="Help improve the physics assistant by sharing anonymized usage data"
        )
        
        save_chat_history = st.checkbox(
            "Save chat history",
            value=True,
            help="Keep your conversation history for progress tracking"
        )
        
        # Data export options
        st.markdown("**Export Options:**")
        if st.button("📤 Export All Data"):
            session_data = session_manager.export_session_data()
            st.download_button(
                label="📥 Download Data Package",
                data=str(session_data),
                file_name="physics_assistant_data.json",
                mime="application/json"
            )
    
    # Save data preferences
    if st.button("💾 Save Data Preferences"):
        data_prefs = {
            'allow_analytics': allow_analytics,
            'save_chat_history': save_chat_history
        }
        
        st.session_state['data_preferences'] = data_prefs
        st.success("✅ Data preferences saved!")

def render_system_info():
    """Render system and application information"""
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Application Info:**")
        st.write(f"**Version:** {Config.APP_VERSION}")
        st.write(f"**App Name:** {Config.APP_NAME}")
        st.write("**Build:** Development")
        
        # Server status (placeholder)
        server_status = "🟢 Online"  # This would check actual MCP server status
        st.write(f"**MCP Server:** {server_status}")
    
    with col2:
        st.markdown("**Physics Constants:**")
        constants_to_show = ["g", "c", "e", "h"]
        
        for constant in constants_to_show:
            value = Config.PHYSICS_CONSTANTS.get(constant, "N/A")
            st.write(f"**{constant}:** {value}")
        
        if st.button("📋 View All Constants"):
            with st.expander("All Physics Constants", expanded=True):
                for const, val in Config.PHYSICS_CONSTANTS.items():
                    st.write(f"**{const}:** {val}")
    
    # Help and support
    st.markdown("---")
    st.subheader("❓ Help & Support")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📖 User Guide", use_container_width=True):
            st.info("User guide feature coming soon!")
    
    with col2:
        if st.button("🐛 Report Bug", use_container_width=True):
            st.info("Bug reporting feature coming soon!")
    
    with col3:
        if st.button("💬 Contact Support", use_container_width=True):
            st.info("For support, contact: physics-help@university.edu")

if __name__ == "__main__":
    main()