import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from components.auth import AuthManager
from services.data_manager import SessionDataManager
from config import Config

def main():
    """Progress analytics dashboard"""
    st.set_page_config(
        page_title=f"{Config.PAGE_TITLE} - Progress",
        page_icon="📊",
        layout="wide"
    )
    
    # Check authentication
    auth_manager = AuthManager()
    if not auth_manager.is_authenticated():
        st.warning("Please log in to view your progress.")
        if st.button("Go to Login"):
            st.switch_page("pages/login.py")
        return
    
    # Initialize data manager
    session_manager = SessionDataManager()
    
    st.title("📊 Learning Progress Dashboard")
    st.markdown(f"**Student:** {st.session_state.get('username', 'Unknown')}")
    
    # Get session statistics
    stats = session_manager.get_session_statistics()
    user_progress = session_manager.get_user_progress()
    
    # Overview metrics
    render_overview_metrics(stats)
    
    # Progress charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_session_activity(session_manager)
    
    with col2:
        render_agent_usage(stats)
    
    # Detailed progress analysis
    render_topic_progress(user_progress)
    
    # Chat history analysis
    render_chat_analysis(session_manager)
    
    # Export functionality
    render_export_section(session_manager)

def render_overview_metrics(stats):
    """Render overview metrics cards"""
    st.subheader("📈 Session Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Messages",
            stats['total_messages'],
            help="Total number of messages in this session"
        )
    
    with col2:
        st.metric(
            "Session Duration",
            f"{stats['duration_minutes']} min",
            help="Time spent in this session"
        )
    
    with col3:
        success_rate = stats.get('success_rate', 0)
        st.metric(
            "Success Rate",
            f"{success_rate}%",
            help="Percentage of successful problem attempts"
        )
    
    with col4:
        st.metric(
            "Topics Covered",
            stats['topics_covered'],
            help="Number of different physics topics explored"
        )

def render_session_activity(session_manager):
    """Render session activity timeline"""
    st.subheader("⏱️ Activity Timeline")
    
    chat_history = session_manager.get_chat_history()
    
    if not chat_history:
        st.info("No activity data available yet.")
        return
    
    # Create activity timeline data
    activity_data = []
    for msg in chat_history:
        timestamp = msg.get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp)
        activity_data.append({
            'time': dt,
            'hour': dt.hour,
            'minute_group': (dt.minute // 10) * 10,  # Group by 10-minute intervals
            'role': msg.get('role', 'unknown')
        })
    
    if activity_data:
        df = pd.DataFrame(activity_data)
        
        # Create hourly activity chart
        hourly_activity = df.groupby('hour').size().reset_index()
        hourly_activity.columns = ['Hour', 'Messages']
        
        fig = px.bar(
            hourly_activity,
            x='Hour',
            y='Messages',
            title="Messages by Hour",
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Messages",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No activity data to display.")

def render_agent_usage(stats):
    """Render agent usage statistics"""
    st.subheader("🤖 Agent Usage")
    
    agents_used = stats.get('agents_used', [])
    
    if not agents_used:
        st.info("No agents have been used yet.")
        return
    
    # Get agent names and create usage data
    agent_data = []
    for agent_id in agents_used:
        agent_info = Config.PHYSICS_AGENTS.get(agent_id, {})
        agent_data.append({
            'Agent': agent_info.get('name', agent_id),
            'Usage': 1  # In a full implementation, track actual usage counts
        })
    
    if agent_data:
        df = pd.DataFrame(agent_data)
        
        fig = px.pie(
            df,
            values='Usage',
            names='Agent',
            title="Agent Usage Distribution"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

def render_topic_progress(user_progress):
    """Render detailed topic progress"""
    st.subheader("🎯 Topic Mastery Progress")
    
    if not user_progress:
        st.info("No topic progress data available yet. Start chatting with agents to track your learning progress!")
        return
    
    # Create progress DataFrame
    progress_data = []
    for topic, data in user_progress.items():
        success_rate = (data['successes'] / data['attempts'] * 100) if data['attempts'] > 0 else 0
        progress_data.append({
            'Topic': topic.title(),
            'Attempts': data['attempts'],
            'Successes': data['successes'],
            'Success Rate (%)': round(success_rate, 1),
            'Avg Difficulty': round(data['avg_difficulty'], 1),
            'Last Attempt': data['last_attempt']
        })
    
    df = pd.DataFrame(progress_data)
    
    # Display progress table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Success Rate (%)": st.column_config.ProgressColumn(
                "Success Rate (%)",
                help="Percentage of successful attempts",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Avg Difficulty": st.column_config.NumberColumn(
                "Avg Difficulty",
                help="Average difficulty level (1-5)",
                format="%.1f",
                min_value=1,
                max_value=5,
            )
        }
    )
    
    # Progress visualization
    if len(progress_data) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate by topic
            fig_success = px.bar(
                df,
                x='Topic',
                y='Success Rate (%)',
                title="Success Rate by Topic",
                color='Success Rate (%)',
                color_continuous_scale='RdYlGn'
            )
            fig_success.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            # Attempts vs difficulty
            fig_difficulty = px.scatter(
                df,
                x='Avg Difficulty',
                y='Attempts',
                size='Successes',
                color='Success Rate (%)',
                hover_name='Topic',
                title="Difficulty vs Attempts",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_difficulty, use_container_width=True)

def render_chat_analysis(session_manager):
    """Render chat history analysis"""
    st.subheader("💬 Conversation Analysis")
    
    chat_history = session_manager.get_chat_history()
    
    if not chat_history:
        st.info("No conversation data available yet.")
        return
    
    # Analyze message patterns
    user_messages = [msg for msg in chat_history if msg.get('role') == 'user']
    assistant_messages = [msg for msg in chat_history if msg.get('role') == 'assistant']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Your Messages", len(user_messages))
    
    with col2:
        st.metric("Agent Responses", len(assistant_messages))
    
    with col3:
        avg_response_time = "< 1 sec"  # Placeholder
        st.metric("Avg Response Time", avg_response_time)
    
    # Recent topics
    recent_topics = session_manager.get_recent_topics()
    if recent_topics:
        st.markdown("**Recent Topics Discussed:**")
        topic_cols = st.columns(min(len(recent_topics), 5))
        for i, topic in enumerate(recent_topics[:5]):
            with topic_cols[i]:
                st.info(f"📚 {topic.title()}")

def render_export_section(session_manager):
    """Render data export functionality"""
    st.subheader("📤 Export Your Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Progress Report", use_container_width=True):
            session_data = session_manager.export_session_data()
            
            # Create downloadable JSON
            st.download_button(
                label="Download Progress Data (JSON)",
                data=pd.Series(session_data).to_json(indent=2),
                file_name=f"physics_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("💬 Export Chat History", use_container_width=True):
            chat_history = session_manager.get_chat_history()
            
            if chat_history:
                # Create CSV format for chat history
                chat_df = pd.DataFrame([
                    {
                        'Timestamp': datetime.fromtimestamp(msg.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                        'Role': msg.get('role', ''),
                        'Content': msg.get('content', ''),
                        'Agent': msg.get('agent_id', '')
                    }
                    for msg in chat_history
                ])
                
                st.download_button(
                    label="Download Chat History (CSV)",
                    data=chat_df.to_csv(index=False),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No chat history to export.")

if __name__ == "__main__":
    main()