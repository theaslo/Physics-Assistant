import streamlit as st
from typing import Optional, Dict
from config import Config
from services.api_client import PhysicsAPIClient

class AgentManager:
    """Manages physics agent selection and configuration"""
    
    def __init__(self):
        self.api_client = PhysicsAPIClient()
        self.agents = self._get_available_agents()
    
    def _get_available_agents(self) -> Dict[str, Dict]:
        """Get available agents from API or fallback to config"""
        try:
            if self.api_client.is_connected():
                api_agents = self.api_client.list_available_agents()
                agents_dict = {}
                
                for agent in api_agents.get('available_agents', []):
                    agent_id = agent['agent_id']
                    agent_info = self.api_client.get_agent_info(agent_id)
                    agents_dict[agent_id] = agent_info
                
                return agents_dict
            else:
                # Fallback to config agents if API not available
                return {
                    'forces_agent': {
                        'name': 'Forces Agent',
                        'icon': '⚖️',
                        'description': 'Specialized in force analysis, free body diagrams, and Newton\'s laws'
                    },
                    'kinematics_agent': {
                        'name': 'Kinematics Agent',
                        'icon': '🚀',
                        'description': 'Expert in motion analysis, projectile motion, and kinematics equations'
                    },
                    'math_agent': {
                        'name': 'Math Agent',
                        'icon': '🔢',
                        'description': 'Expert in mathematical calculations, algebra, and computational problems'
                    },
                    'momentum_agent': {
                        'name': 'Momentum Agent',
                        'icon': '💥',
                        'description': 'Specialized in momentum, impulse, and collision analysis'
                    },
                    'energy_agent': {
                        'name': 'Energy Agent',
                        'icon': '⚡',
                        'description': 'Expert in work, energy, power, and conservation principles'
                    },
                    'angular_motion_agent': {
                        'name': 'Angular Motion Agent',
                        'icon': '🌀',
                        'description': 'Specialized in rotational motion, angular momentum, and torque'
                    }
                }
        except Exception as e:
            st.warning(f"Could not load agents from API: {str(e)}")
            return Config.PHYSICS_AGENTS if hasattr(Config, 'PHYSICS_AGENTS') else {}
    
    def render_agent_selector(self) -> Optional[str]:
        """Render agent selection interface"""
        st.markdown("**Select a Physics Agent:**")
        
        # Create agent options with icons and descriptions
        agent_options = []
        agent_mapping = {}
        
        for agent_id, agent_info in self.agents.items():
            display_name = f"{agent_info['icon']} {agent_info['name']}"
            agent_options.append(display_name)
            agent_mapping[display_name] = agent_id
        
        # Add "None" option
        agent_options.insert(0, "🚫 No Agent Selected")
        agent_mapping["🚫 No Agent Selected"] = None
        
        # Use the selectbox with proper state management
        # Let the selectbox key handle the state, don't interfere with index
        if "agent_selector" not in st.session_state:
            st.session_state["agent_selector"] = "🚫 No Agent Selected"
            
        selected_display = st.selectbox(
            "Choose Agent:",
            agent_options,
            label_visibility="collapsed",
            key="agent_selector"
        )
        
        
        selected_agent = agent_mapping[selected_display]
        
        # Show agent description if selected
        if selected_agent and selected_agent in self.agents:
            agent_info = self.agents[selected_agent]
            st.info(f"**{agent_info['name']}**: {agent_info['description']}")
            
            # Show agent capabilities
            self._show_agent_capabilities(selected_agent)
        
        return selected_agent
    
    def _show_agent_capabilities(self, agent_id: str):
        """Show capabilities for selected agent"""
        # First try to get capabilities from API
        api_capabilities = None
        if self.api_client.is_connected():
            try:
                result = self.api_client.get_agent_capabilities(agent_id)
                if 'capabilities' in result:
                    api_capabilities = result.get('available_tools', [])
            except Exception:
                pass
        
        # Get fallback capabilities
        fallback_capabilities = self._get_agent_capabilities(agent_id)
        
        # Use API capabilities if available, otherwise fallback
        capabilities = fallback_capabilities
        
        if capabilities:
            with st.expander(f"💡 {self.agents[agent_id]['name']} Capabilities"):
                # Show API connection status
                if self.api_client.is_connected():
                    st.success("🟢 Connected to API server")
                    if api_capabilities:
                        st.write("**Available MCP Tools:**")
                        for tool in api_capabilities:
                            st.write(f"• {tool}")
                        st.write("**Physics Capabilities:**")
                else:
                    st.warning("🟡 API server offline - showing general capabilities")
                
                for capability in capabilities:
                    st.write(f"• {capability}")
    
    def _get_agent_capabilities(self, agent_id: str) -> list:
        """Get capabilities for a specific agent"""
        capabilities_map = {
            "kinematics_agent": [
                "Position, velocity, and acceleration analysis",
                "1D motion with constant acceleration",
                "2D projectile motion problems",
                "Graphical motion analysis",
                "Kinematic equation solving"
            ],
            "forces_agent": [
                "Free body diagram analysis",
                "Newton's first, second, and third laws",
                "Force vector addition and decomposition",
                "Static and kinetic friction problems",
                "Inclined plane analysis",
                "Tension and normal force calculations"
            ],
            "energy_agent": [
                "Work calculations (W = F·d)",
                "Kinetic and potential energy",
                "Conservation of mechanical energy",
                "Work-energy theorem applications",
                "Power calculations (P = W/t)",
                "Energy efficiency problems"
            ],
            "momentum_agent": [
                "Linear momentum calculations (p = mv)",
                "Conservation of momentum",
                "Collision analysis (elastic and inelastic)",
                "Impulse-momentum theorem (J = Δp)",
                "Center of mass problems",
                "Explosion and recoil problems"
            ],
            "angular_motion_agent": [
                "Angular position, velocity, acceleration",
                "Rotational kinematics equations",
                "Moment of inertia calculations",
                "Torque analysis (τ = r × F)",
                "Rotational energy and angular momentum",
                "Rolling motion problems"
            ],
            "math_agent": [
                "Trigonometric functions and identities",
                "Vector operations and decomposition",
                "Algebraic equation solving",
                "Unit conversions",
                "Scientific notation handling",
                "Graph interpretation and analysis"
            ]
        }
        
        return capabilities_map.get(agent_id, [])
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict]:
        """Get information about a specific agent"""
        if agent_id in self.agents:
            return self.agents[agent_id]
        return None
    
    def get_all_agents(self) -> Dict:
        """Get all available agents"""
        return self.agents
    
    def is_agent_available(self, agent_id: str) -> bool:
        """Check if an agent is available"""
        return agent_id in self.agents