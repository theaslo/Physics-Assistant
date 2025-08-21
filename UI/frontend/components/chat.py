import streamlit as st
import time
from typing import List, Dict, Optional
from config import Config
from services.api_client import PhysicsAPIClient

class ChatInterface:
    """Manages the chat interface for physics assistance"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.api_client = PhysicsAPIClient()
        self.agent_info = self.api_client.get_agent_info(agent_id)
    
    def render(self):
        """Render the complete chat interface"""        
        # Chat header
        self._render_chat_header()
        
        # Chat history display
        self._render_chat_history()
        
        # Input interface
        self._render_input_interface()
    
    def _render_chat_header(self):
        """Render chat header with agent info"""
        agent_name = self.agent_info.get('name', 'Physics Agent')
        agent_icon = self.agent_info.get('icon', '🤖')
        
        st.markdown(f"""
        <div style='background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h3 style='margin: 0; color: #1f1f1f;'>{agent_icon} {agent_name}</h3>
            <p style='margin: 0; color: #666;'>{self.agent_info.get('description', 'Physics tutoring assistant')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_chat_history(self):
        """Render chat message history"""
        # Use agent-specific chat history
        chat_history_key = f'chat_history_{self.agent_id}'
        if not st.session_state.get(chat_history_key):
            # Show welcome message
            self._show_welcome_message()
        else:
            # Display chat messages
            chat_container = st.container()
            chat_history_key = f'chat_history_{self.agent_id}'
            with chat_container:
                for message in st.session_state[chat_history_key]:
                    self._render_message(message)
    
    def _show_welcome_message(self):
        """Show welcome message for new chat session"""
        agent_name = self.agent_info.get('name', 'Physics Agent')
        
        with st.chat_message("assistant", avatar=self.agent_info.get('icon', '🤖')):
            st.markdown(f"""
            Hello! I'm the **{agent_name}**. I'm here to help you with physics problems and concepts.
            
            **I can help you with:**
            {self._get_agent_help_topics()}
            
            **Example Questions You Can Ask:**
            {self._get_example_questions()}
            
            **How to get started:**
            - Ask me a specific physics question
            - Upload an image of a physics problem  
            - Request help with a particular concept
            
            What would you like to work on today?
            """)
    
    def _get_agent_help_topics(self) -> str:
        """Get formatted help topics for the current agent"""
        help_topics = {
            "kinematics": "• Position, velocity, and acceleration\n• Motion graphs and equations\n• Projectile motion",
            "forces": "• Newton's laws of motion\n• Free body diagrams\n• Friction and tension problems",
            "energy": "• Work and energy calculations\n• Conservation of energy\n• Power and efficiency",
            "momentum": "• Linear momentum\n• Collision analysis\n• Impulse and momentum conservation",
            "rotation": "• Rotational motion\n• Torque and angular momentum\n• Moment of inertia",
            "math_helper": "• Vector operations\n• Trigonometry\n• Unit conversions and algebra"
        }
        
        return help_topics.get(self.agent_id, "• General physics concepts\n• Problem solving\n• Mathematical calculations")
    
    def _get_example_questions(self) -> str:
        """Get example questions for the current agent"""
        # Try to get example questions from agent capabilities (from API)
        try:
            capabilities = self.api_client.get_agent_capabilities(self.agent_id)
            if capabilities and 'metadata' in capabilities:
                examples = capabilities.get('metadata', {}).get('example_problems', [])
                if examples:
                    # Show first 5 examples
                    formatted_examples = []
                    for i, example in enumerate(examples[:5], 1):
                        formatted_examples.append(f"{i}. *{example}*")
                    return "\n".join(formatted_examples)
        except Exception:
            # If API fails, continue to fallback
            pass
        
        # Fallback examples for each agent type
        example_questions = {
            "math_agent": [
                "*Solve x² + 5x + 6 = 0 using the quadratic formula*",
                "*What is sin(45°) and cos(45°)?*", 
                "*Calculate log₁₀(100) and ln(e²)*"
            ],
            "forces_agent": [
                "*A 5kg box on a 30° incline with friction coefficient 0.3*",
                "*Add forces: 10N at 30°, 15N at 120°, 8N at 270°*",
                "*Calculate spring force with k=200 N/m, compressed 0.05m*"
            ],
            "kinematics_agent": [
                "*Car accelerates from rest at 3 m/s² for 5 seconds*",
                "*Ball thrown at 30 m/s at 45° from 10m height*",
                "*Object dropped from 50m - how long to fall?*"
            ],
            "momentum_agent": [
                "*Calculate momentum of 5kg object moving at 10 m/s*",
                "*2kg ball at 8 m/s collides with 3kg ball at rest*",
                "*Car crash: 1500kg at 20 m/s hits 1200kg at 15 m/s*"
            ],
            "energy_agent": [
                "*Calculate kinetic energy of 2kg object at 5 m/s*",
                "*Ball lifted 10m high - what's the potential energy?*",
                "*Work done pushing 100N force over 5m distance*"
            ],
            "angular_motion_agent": [
                "*Calculate moment of inertia of 2kg rod, 1.5m long*",
                "*Cylinder rolls down 30° incline, mass=5kg, radius=0.3m*",
                "*Figure skater spins faster when pulling arms in*"
            ]
        }
        
        examples = example_questions.get(self.agent_id, ["*Ask me any physics question!*"])
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_examples.append(f"{i}. {example}")
        return "\n".join(formatted_examples)
    
    def _render_message(self, message: Dict):
        """Render a single chat message"""
        role = message.get('role', 'user')
        content = message.get('content', '')
        timestamp = message.get('timestamp', time.time())
        
        if role == 'user':
            with st.chat_message("user", avatar="👨‍🎓"):
                st.markdown(content)
                if 'image' in message:
                    st.image(message['image'], caption="Uploaded problem")
        else:
            with st.chat_message("assistant", avatar=self.agent_info.get('icon', '🤖')):
                st.markdown(content)
                
                # Show any additional content (equations, plots, etc.)
                if 'latex' in message:
                    st.latex(message['latex'])
                if 'plot' in message:
                    st.plotly_chart(message['plot'], use_container_width=True)
                if 'dataframe' in message:
                    st.dataframe(message['dataframe'])
    
    def _render_input_interface(self):
        """Render the input interface with various input methods"""
        # Text input area
        user_input = st.chat_input("Ask your physics question here...")
        
        if user_input:
            try:
                self._handle_user_input(user_input)
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                st.exception(e)
        
        # Additional input options
        with st.expander("📎 Additional Input Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                # File uploader for problem images
                uploaded_file = st.file_uploader(
                    "Upload problem image",
                    type=Config.ALLOWED_FILE_TYPES,
                    help="Upload an image of your physics problem"
                )
                
                if uploaded_file and st.button("Analyze Image"):
                    self._handle_image_upload(uploaded_file)
            
            with col2:
                # LaTeX input for equations
                latex_input = st.text_area(
                    "Enter LaTeX equation",
                    placeholder="e.g., F = ma",
                    help="Enter mathematical equations in LaTeX format"
                )
                
                if latex_input and st.button("Add Equation"):
                    self._handle_latex_input(latex_input)
        
        # Quick action buttons
        self._render_quick_actions()
    
    def _render_quick_actions(self):
        """Render quick action buttons for common physics topics"""
        st.markdown("**Quick Actions:**")
        
        quick_actions = self._get_quick_actions()
        
        if quick_actions:
            cols = st.columns(len(quick_actions))
            for i, (action_text, action_query) in enumerate(quick_actions):
                with cols[i]:
                    if st.button(action_text, use_container_width=True):
                        self._handle_user_input(action_query)
    
    def _get_quick_actions(self) -> List[tuple]:
        """Get quick action buttons for the current agent"""
        actions_map = {
            "kinematics": [
                ("📈 Motion Graphs", "Help me understand motion graphs"),
                ("🎯 Projectile Motion", "Explain projectile motion"),
                ("⚡ Acceleration", "How do I calculate acceleration?")
            ],
            "forces": [
                ("📋 Free Body Diagram", "How do I draw a free body diagram?"),
                ("⚖️ Newton's Laws", "Explain Newton's laws of motion"),
                ("🏔️ Inclined Plane", "Help with inclined plane problems")
            ],
            "energy": [
                ("⚡ Work Formula", "What is the formula for work?"),
                ("🔋 Energy Conservation", "Explain conservation of energy"),
                ("💪 Power Calculation", "How do I calculate power?")
            ],
            "momentum": [
                ("💥 Collisions", "Help with collision problems"),
                ("📏 Momentum Formula", "What is the momentum formula?"),
                ("🎱 Conservation", "Explain momentum conservation")
            ],
            "rotation": [
                ("🌀 Angular Motion", "Explain angular motion"),
                ("🔧 Torque", "What is torque and how to calculate it?"),
                ("⚙️ Moment of Inertia", "Help with moment of inertia")
            ],
            "math_helper": [
                ("📐 Vectors", "Help with vector operations"),
                ("🔢 Unit Conversion", "Convert units for me"),
                ("📊 Trigonometry", "Help with trig functions")
            ]
        }
        
        return actions_map.get(self.agent_id, [])
    
    def _handle_user_input(self, user_input: str):
        """Process user text input"""
        # Add user message to chat history with enhanced logging
        user_message = {
            'role': 'user',
            'content': user_input,
            'timestamp': time.time(),
            'agent_id': self.agent_id,
            'metadata': {
                'interaction_type': 'text_message',
                'user_agent': st.context.headers.get('user-agent', 'unknown') if hasattr(st, 'context') else 'unknown'
            }
        }
        
        # Use agent-specific chat history
        chat_history_key = f'chat_history_{self.agent_id}'
        if chat_history_key not in st.session_state:
            st.session_state[chat_history_key] = []
        
        st.session_state[chat_history_key].append(user_message)
        
        # Log to enhanced session data manager if available
        try:
            from services.database_client import get_data_manager
            from config import Config
            session_manager = get_data_manager(Config.DATABASE_API_URL)
            session_manager.save_chat_message(user_message)
        except Exception as e:
            # Fallback to basic session state logging
            pass
        
        # Get response from agent
        with st.spinner("Thinking..."):
            response = self._get_agent_response(user_input)
        
        
        # Add agent response to chat history with enhanced logging
        agent_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': time.time(),
            'agent_id': self.agent_id,
            'agent_icon': self.agent_info.get('icon', '🤖'),
            'response_time': time.time() - (user_message.get('timestamp', time.time())),
            'metadata': {
                'api_client_used': True,
                'agent_type': 'physics_api',
                'interaction_type': 'text_message'
            }
        }
        
        # Use agent-specific chat history
        chat_history_key = f'chat_history_{self.agent_id}'
        st.session_state[chat_history_key].append(agent_message)
        
        # Log to enhanced session data manager if available
        try:
            from services.database_client import get_data_manager
            from config import Config
            session_manager = get_data_manager(Config.DATABASE_API_URL)
            session_manager.save_chat_message(agent_message)
        except Exception as e:
            # Fallback to basic session state logging
            pass
        
        # Force UI to update to show the new message
        st.rerun()
    
    def _handle_image_upload(self, uploaded_file):
        """Process uploaded image"""
        # Add image message to chat history
        user_message = {
            'role': 'user',
            'content': "I've uploaded a physics problem image for analysis.",
            'image': uploaded_file,
            'timestamp': time.time()
        }
        
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        st.session_state['chat_history'].append(user_message)
        
        # Analyze image (placeholder)
        with st.spinner("Analyzing image..."):
            response = self._analyze_image(uploaded_file)
        
        agent_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': time.time()
        }
        
        # Use agent-specific chat history
        chat_history_key = f'chat_history_{self.agent_id}'
        st.session_state[chat_history_key].append(agent_message)
    
    def _handle_latex_input(self, latex_input: str):
        """Process LaTeX equation input"""
        user_message = {
            'role': 'user',
            'content': f"Help me understand this equation: {latex_input}",
            'latex': latex_input,
            'timestamp': time.time()
        }
        
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        st.session_state['chat_history'].append(user_message)
        
        # Get explanation for equation
        with st.spinner("Analyzing equation..."):
            response = self._explain_equation(latex_input)
        
        agent_message = {
            'role': 'assistant',
            'content': response,
            'timestamp': time.time()
        }
        
        # Use agent-specific chat history
        chat_history_key = f'chat_history_{self.agent_id}'
        st.session_state[chat_history_key].append(agent_message)
    
    def _get_agent_response(self, user_input: str) -> str:
        """Get response from FastAPI server using the agent"""
        try:
            # Check if API is connected
            if not self.api_client.is_connected():
                return "⚠️ Sorry, I'm unable to connect to the physics assistant server. Please ensure the FastAPI server is running and try again."
            
            # Always ensure agent exists - the API handles duplicate creation gracefully
            result = self.api_client.create_agent(self.agent_id)
            if not result.get('success', False):
                return f"❌ Failed to initialize agent: {result.get('error', 'Unknown error')}"
            
            # Send message to agent via API
            response = self.api_client.send_message(
                agent_id=self.agent_id,
                message=user_input,
                user_id=st.session_state.get('username', 'anonymous')
            )
            
            if response.get('success', False):
                # API returns 'solution' field, not 'content'
                content = response.get('solution', response.get('content', 'Solution generated'))
                
                # Add metadata information if available
                tools_used = response.get('tools_used', [])
                reasoning = response.get('reasoning', '')
                
                # Format the response with additional info
                formatted_response = content
                
                if tools_used:
                    formatted_response += f"\n\n🔧 **Tools used:** {', '.join(tools_used)}"
                
                if reasoning and reasoning != content:
                    formatted_response += f"\n\n💭 **Approach:** {reasoning}"
                
                return formatted_response
            else:
                error_msg = response.get('error', 'Unknown error occurred')
                return f"❌ Sorry, I encountered an error while solving your problem: {error_msg}"
                
        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}. Please try again or check if the API server is running."
    
    def _analyze_image(self, uploaded_file) -> str:
        """Analyze uploaded physics problem image (placeholder)"""
        return "I can see you've uploaded an image! In the full implementation, I would analyze the physics problem shown in your image and provide step-by-step solutions. This feature requires integration with computer vision capabilities."
    
    def _explain_equation(self, latex_input: str) -> str:
        """Explain a LaTeX equation (placeholder)"""
        return f"Great! You've entered the equation: {latex_input}. In the full implementation, I would provide a detailed explanation of this equation, its variables, units, and how to apply it to solve physics problems."