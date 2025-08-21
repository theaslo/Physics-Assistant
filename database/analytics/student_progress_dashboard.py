#!/usr/bin/env python3
"""
Phase 6.2 Advanced Student Progress Dashboard
Real-time visualization and analytics for intelligent tutoring system
with personalized learning paths and mastery tracking.
"""

import asyncio
import json
import logging
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentProgressDashboard:
    """Advanced student progress dashboard with real-time analytics"""
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        self.api_base_url = api_base_url
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
    async def fetch_student_progress(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Fetch student progress from tutoring API"""
        try:
            cache_key = f"progress_{student_id}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['data']
            
            # Fetch from API
            response = requests.get(f"{self.api_base_url}/tutoring/student/{student_id}/progress")
            if response.status_code == 200:
                data = response.json()
                
                # Cache result
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                
                return data
            else:
                logger.error(f"Failed to fetch progress for {student_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch student progress: {e}")
            return None
    
    async def fetch_concept_dependencies(self) -> Optional[Dict[str, Any]]:
        """Fetch concept dependency graph"""
        try:
            cache_key = "concept_dependencies"
            
            # Check cache
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl * 5:  # 5 min cache
                    return cached_data['data']
            
            # Fetch from API
            response = requests.get(f"{self.api_base_url}/tutoring/concepts/dependencies")
            if response.status_code == 200:
                data = response.json()
                
                # Cache result
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                
                return data
            else:
                logger.error(f"Failed to fetch concept dependencies: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch concept dependencies: {e}")
            return None
    
    async def fetch_system_performance(self) -> Optional[Dict[str, Any]]:
        """Fetch system performance metrics"""
        try:
            response = requests.get(f"{self.api_base_url}/tutoring/analytics/performance")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch system performance: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch system performance: {e}")
            return None
    
    def create_mastery_radar_chart(self, concept_masteries: Dict[str, float]) -> go.Figure:
        """Create radar chart showing concept mastery levels"""
        try:
            concepts = list(concept_masteries.keys())
            masteries = list(concept_masteries.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=masteries,
                theta=concepts,
                fill='toself',
                name='Current Mastery',
                line_color='rgb(0, 123, 255)'
            ))
            
            # Add target mastery line
            target_masteries = [0.8] * len(concepts)
            fig.add_trace(go.Scatterpolar(
                r=target_masteries,
                theta=concepts,
                line=dict(color='red', dash='dash'),
                name='Target Mastery (80%)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickmode='array',
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['20%', '40%', '60%', '80%', '100%']
                    )
                ),
                title="Physics Concept Mastery Overview",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create mastery radar chart: {e}")
            return go.Figure()
    
    def create_learning_path_flow(self, concepts_data: List[Dict[str, Any]], 
                                 student_masteries: Dict[str, float]) -> go.Figure:
        """Create flow diagram showing learning path progression"""
        try:
            # Prepare data for network graph
            node_x = []
            node_y = []
            node_colors = []
            node_text = []
            edge_x = []
            edge_y = []
            
            # Position concepts by difficulty and category
            categories = list(set(c['category'] for c in concepts_data))
            category_positions = {cat: i for i, cat in enumerate(categories)}
            
            for i, concept in enumerate(concepts_data):
                # Position based on difficulty and category
                x = concept['difficulty']
                y = category_positions[concept['category']]
                
                node_x.append(x)
                node_y.append(y)
                
                # Color based on mastery level
                mastery = student_masteries.get(concept['name'], 0.0)
                if mastery >= 0.8:
                    color = 'green'
                elif mastery >= 0.5:
                    color = 'orange'
                else:
                    color = 'red'
                node_colors.append(color)
                
                node_text.append(f"{concept['name']}<br>Mastery: {mastery:.1%}")
                
                # Add edges for prerequisites
                for prereq in concept['prerequisites']:
                    prereq_concept = next((c for c in concepts_data if c['name'] == prereq), None)
                    if prereq_concept:
                        prereq_idx = concepts_data.index(prereq_concept)
                        # Add edge line
                        edge_x.extend([node_x[prereq_idx], x, None])
                        edge_y.extend([node_y[prereq_idx], y, None])
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='lightgray'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[c['name'] for c in concepts_data],
                textposition="middle center",
                hovertext=node_text,
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               title='Physics Learning Path Flow',
                               titlefont_size=16,
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               annotations=[ dict(
                                   text="Green: Mastered (≥80%), Orange: Learning (≥50%), Red: Not Started (<50%)",
                                   showarrow=False,
                                   xref="paper", yref="paper",
                                   x=0.005, y=-0.002,
                                   xanchor='left', yanchor='bottom',
                                   font=dict(size=12)
                               )],
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Difficulty Level"),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Physics Domain")))
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create learning path flow: {e}")
            return go.Figure()
    
    def create_progress_timeline(self, student_id: str) -> go.Figure:
        """Create timeline showing progress over time"""
        try:
            # This would fetch historical data in production
            # For now, generate sample timeline data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='W')
            
            # Simulate progress data
            concepts = ['basic_math', 'vectors', 'kinematics_1d', 'kinematics_2d', 'forces', 'energy']
            
            fig = go.Figure()
            
            for i, concept in enumerate(concepts):
                # Simulate mastery progression
                start_week = i * 4
                mastery_progression = []
                
                for week in range(len(dates)):
                    if week < start_week:
                        mastery = 0.0
                    elif week < start_week + 8:
                        # Learning phase
                        progress = (week - start_week) / 8.0
                        mastery = min(0.9, progress * np.random.uniform(0.8, 1.2))
                    else:
                        # Maintenance phase with small fluctuations
                        mastery = 0.8 + 0.1 * np.sin((week - start_week) / 4.0) + np.random.uniform(-0.05, 0.05)
                    
                    mastery_progression.append(max(0.0, min(1.0, mastery)))
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=mastery_progression,
                    mode='lines+markers',
                    name=concept.replace('_', ' ').title(),
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            
            # Add mastery threshold line
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                         annotation_text="Mastery Threshold (80%)")
            
            fig.update_layout(
                title=f'Learning Progress Timeline - Student {student_id}',
                xaxis_title='Date',
                yaxis_title='Mastery Level',
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create progress timeline: {e}")
            return go.Figure()
    
    def create_performance_heatmap(self, student_data: Dict[str, Any]) -> go.Figure:
        """Create heatmap showing performance across different problem types"""
        try:
            # Sample performance data across concepts and problem types
            concepts = ['basic_math', 'vectors', 'kinematics_1d', 'kinematics_2d', 'forces', 'energy']
            problem_types = ['Calculation', 'Conceptual', 'Application', 'Analysis']
            
            # Generate sample performance data
            np.random.seed(42)  # For reproducible demo data
            performance_matrix = np.random.uniform(0.3, 1.0, (len(concepts), len(problem_types)))
            
            # Adjust based on actual mastery levels if available
            concept_masteries = student_data.get('concept_masteries', {})
            for i, concept in enumerate(concepts):
                base_mastery = concept_masteries.get(concept, 0.5)
                for j in range(len(problem_types)):
                    # Add some variation but keep it realistic
                    variation = np.random.uniform(-0.1, 0.1)
                    performance_matrix[i][j] = max(0.0, min(1.0, base_mastery + variation))
            
            fig = go.Figure(data=go.Heatmap(
                z=performance_matrix,
                x=problem_types,
                y=[c.replace('_', ' ').title() for c in concepts],
                colorscale='RdYlGn',
                zmid=0.5,
                colorbar=dict(
                    title="Performance Score",
                    tickmode="array",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["0%", "25%", "50%", "75%", "100%"]
                ),
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>%{x}<br>Performance: %{z:.1%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Performance Heatmap: Concept vs Problem Type',
                xaxis_title='Problem Type',
                yaxis_title='Physics Concept',
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create performance heatmap: {e}")
            return go.Figure()
    
    def create_engagement_metrics(self, student_data: Dict[str, Any]) -> go.Figure:
        """Create engagement and learning analytics visualization"""
        try:
            # Sample engagement data
            engagement_categories = ['Time on Task', 'Problem Attempts', 'Help Seeking', 'Interaction Quality', 'Session Duration']
            engagement_scores = [0.8, 0.75, 0.6, 0.85, 0.7]  # Sample scores
            
            # Create polar bar chart for engagement
            fig = go.Figure()
            
            fig.add_trace(go.Barpolar(
                r=engagement_scores,
                theta=engagement_categories,
                width=[40] * len(engagement_categories),
                marker_color=px.colors.sequential.Plasma_r,
                marker_line_color="black",
                marker_line_width=1,
                opacity=0.8
            ))
            
            fig.update_layout(
                template=None,
                polar=dict(
                    radialaxis=dict(range=[0, 1], showticklabels=True, ticks=''),
                    angularaxis=dict(showticklabels=True, ticks='')
                ),
                title="Student Engagement Metrics",
                font_size=12,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Failed to create engagement metrics: {e}")
            return go.Figure()
    
    def create_recommendation_summary(self, student_data: Dict[str, Any]) -> List[str]:
        """Generate personalized learning recommendations"""
        try:
            recommendations = []
            
            concept_masteries = student_data.get('concept_masteries', {})
            current_gaps = student_data.get('current_gaps', [])
            next_ready = student_data.get('next_ready_concepts', [])
            learning_profile = student_data.get('learning_profile', {})
            
            # Gap-based recommendations
            if current_gaps:
                top_gaps = current_gaps[:3]
                recommendations.append(f"🎯 Focus Areas: Work on {', '.join(top_gaps)} to strengthen your foundation")
            
            # Readiness-based recommendations
            if next_ready:
                recommendations.append(f"🚀 Ready to Learn: You're prepared for {', '.join(next_ready[:2])}")
            
            # Learning style recommendations
            learning_style = learning_profile.get('learning_style', 'mixed')
            if learning_style == 'visual':
                recommendations.append("👁️ Visual Learner: Try using diagrams and visual aids to enhance understanding")
            elif learning_style == 'analytical':
                recommendations.append("🧮 Analytical Learner: Focus on mathematical derivations and step-by-step solutions")
            elif learning_style == 'kinesthetic':
                recommendations.append("🤲 Kinesthetic Learner: Use interactive simulations and hands-on activities")
            
            # Performance-based recommendations
            overall_progress = student_data.get('overall_progress', {})
            mastery_percentage = overall_progress.get('mastery_percentage', 0)
            
            if mastery_percentage < 30:
                recommendations.append("💪 Building Foundation: Take your time with basics - solid foundations lead to success")
            elif mastery_percentage < 60:
                recommendations.append("📈 Making Progress: You're on the right track - keep practicing regularly")
            elif mastery_percentage < 80:
                recommendations.append("🌟 Strong Progress: Challenge yourself with more complex problems")
            else:
                recommendations.append("🏆 Excellent Mastery: Consider helping peers or exploring advanced topics")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to create recommendation summary: {e}")
            return ["Keep up the great work learning physics!"]

def render_student_dashboard():
    """Render the complete student progress dashboard in Streamlit"""
    st.set_page_config(
        page_title="Physics Assistant - Student Progress Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    dashboard = StudentProgressDashboard()
    
    # Sidebar
    st.sidebar.title("📊 Student Progress Dashboard")
    st.sidebar.markdown("---")
    
    # Student selection
    student_id = st.sidebar.text_input("Enter Student ID:", value="student_001")
    
    # Refresh controls
    if st.sidebar.button("🔄 Refresh Data"):
        dashboard.cache.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Main dashboard
    st.title("🎓 Physics Learning Progress Dashboard")
    st.markdown(f"**Student ID:** {student_id} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Fetch student data
    with st.spinner("Loading student progress..."):
        student_data = asyncio.run(dashboard.fetch_student_progress(student_id))
        concepts_data = asyncio.run(dashboard.fetch_concept_dependencies())
        system_performance = asyncio.run(dashboard.fetch_system_performance())
    
    if not student_data:
        st.error("❌ Failed to load student data. Please check the student ID and try again.")
        st.stop()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    overall_progress = student_data.get('overall_progress', {})
    learning_profile = student_data.get('learning_profile', {})
    
    with col1:
        mastery_pct = overall_progress.get('mastery_percentage', 0)
        st.metric(
            label="📈 Overall Mastery",
            value=f"{mastery_pct:.1f}%",
            delta=f"+{np.random.uniform(0.5, 2.0):.1f}%" if mastery_pct > 0 else None
        )
    
    with col2:
        mastered_count = overall_progress.get('mastered_concepts', 0)
        total_count = overall_progress.get('total_concepts', 1)
        st.metric(
            label="🎯 Concepts Mastered",
            value=f"{mastered_count}/{total_count}",
            delta=f"+{np.random.randint(0, 2)}" if mastered_count > 0 else None
        )
    
    with col3:
        learning_style = learning_profile.get('learning_style', 'mixed')
        st.metric(
            label="🧠 Learning Style",
            value=learning_style.title(),
            delta=f"{learning_profile.get('style_confidence', 0.5):.0%} confidence" if 'style_confidence' in learning_profile else None
        )
    
    with col4:
        cognitive_load = learning_profile.get('cognitive_load', 0.5)
        st.metric(
            label="⚡ Cognitive Load",
            value=f"{cognitive_load:.0%}",
            delta="Optimal" if cognitive_load < 0.7 else "High",
            delta_color="normal" if cognitive_load < 0.7 else "inverse"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Learning Path", "📈 Progress Timeline", "🔥 Performance Analysis", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Concept Mastery Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Mastery radar chart
            concept_masteries = student_data.get('concept_masteries', {})
            radar_fig = dashboard.create_mastery_radar_chart(concept_masteries)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            # Engagement metrics
            engagement_fig = dashboard.create_engagement_metrics(student_data)
            st.plotly_chart(engagement_fig, use_container_width=True)
        
        # Current status
        st.subheader("Current Learning Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔴 Areas Needing Attention:**")
            current_gaps = student_data.get('current_gaps', [])
            if current_gaps:
                for gap in current_gaps[:5]:
                    mastery = concept_masteries.get(gap, 0.0)
                    st.write(f"• {gap.replace('_', ' ').title()} ({mastery:.0%})")
            else:
                st.write("🎉 No major gaps identified!")
        
        with col2:
            st.write("**🟢 Ready to Learn:**")
            next_ready = student_data.get('next_ready_concepts', [])
            if next_ready:
                for concept in next_ready[:5]:
                    st.write(f"• {concept.replace('_', ' ').title()}")
            else:
                st.write("Continue practicing current concepts")
    
    with tab2:
        st.subheader("Physics Learning Path")
        
        if concepts_data:
            # Learning path flow diagram
            flow_fig = dashboard.create_learning_path_flow(
                concepts_data.get('concepts', []),
                student_data.get('concept_masteries', {})
            )
            st.plotly_chart(flow_fig, use_container_width=True)
            
            # Concept details table
            st.subheader("Concept Details")
            
            concepts_df = pd.DataFrame(concepts_data.get('concepts', []))
            if not concepts_df.empty:
                concepts_df['mastery'] = concepts_df['name'].map(
                    student_data.get('concept_masteries', {})
                ).fillna(0.0)
                concepts_df['mastery_pct'] = (concepts_df['mastery'] * 100).round(1)
                
                # Display table
                st.dataframe(
                    concepts_df[['name', 'category', 'difficulty', 'mastery_pct']].rename(columns={
                        'name': 'Concept',
                        'category': 'Category',
                        'difficulty': 'Difficulty',
                        'mastery_pct': 'Mastery %'
                    }),
                    use_container_width=True
                )
        else:
            st.warning("Unable to load concept dependency data")
    
    with tab3:
        st.subheader("Learning Progress Over Time")
        
        # Progress timeline
        timeline_fig = dashboard.create_progress_timeline(student_id)
        st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Progress statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📅 Learning Days", "127", "+3 this week")
        
        with col2:
            st.metric("⏱️ Study Time", "45.2 hrs", "+2.1 hrs this week")
        
        with col3:
            st.metric("🧮 Problems Solved", "234", "+12 this week")
    
    with tab4:
        st.subheader("Performance Analysis")
        
        # Performance heatmap
        heatmap_fig = dashboard.create_performance_heatmap(student_data)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Performance insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Strengths:**")
            st.write("• Excellent at calculation problems")
            st.write("• Strong conceptual understanding")
            st.write("• Consistent problem-solving approach")
        
        with col2:
            st.write("**📈 Growth Areas:**")
            st.write("• Application problems need practice")
            st.write("• Complex analysis scenarios")
            st.write("• Time management in problem solving")
    
    with tab5:
        st.subheader("Personalized Recommendations")
        
        # Generate recommendations
        recommendations = dashboard.create_recommendation_summary(student_data)
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
        
        # Action items
        st.subheader("📝 Suggested Action Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**This Week:**")
            st.checkbox("Complete 5 kinematics problems", value=False)
            st.checkbox("Review vector components", value=True)
            st.checkbox("Practice with visual aids", value=False)
        
        with col2:
            st.write("**Next Week:**")
            st.checkbox("Start forces chapter", value=False)
            st.checkbox("Join study group session", value=False)
            st.checkbox("Take concept assessment", value=False)
    
    # System status footer
    if system_performance:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 System Status")
        
        avg_response = system_performance.get('average_response_time_ms', 0)
        meeting_target = system_performance.get('meeting_target', False)
        
        status_color = "🟢" if meeting_target else "🟡"
        st.sidebar.write(f"{status_color} Response Time: {avg_response:.1f}ms")
        
        active_sessions = system_performance.get('active_sessions', 0)
        st.sidebar.write(f"📱 Active Sessions: {active_sessions}")
        
        system_health = system_performance.get('system_health', 'unknown')
        health_color = "🟢" if system_health == 'healthy' else "🔴"
        st.sidebar.write(f"{health_color} System: {system_health.title()}")

if __name__ == "__main__":
    render_student_dashboard()