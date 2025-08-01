import streamlit as st
from components.auth import AuthManager
from config import Config

def main():
    """Login page for Physics Assistant"""
    st.set_page_config(
        page_title=f"{Config.PAGE_TITLE} - Login",
        page_icon=Config.PAGE_ICON,
        layout="centered"
    )
    
    # Initialize auth manager
    auth_manager = AuthManager()
    
    # Check if already authenticated
    if st.session_state.get('authentication_status'):
        st.success("You are already logged in!")
        if st.button("Go to Physics Assistant"):
            st.switch_page("app.py")
        if st.button("Logout"):
            auth_manager.logout()
            st.rerun()
        return
    
    # Render login interface
    auth_manager.render_login()

if __name__ == "__main__":
    main()