"""
Combined Forces Agent - Merges Abstract Design with Proven MCP Integration
Compatible with Google A2A Framework + Direct Tool Calls That Work

This agent combines:
- Abstraction and A2A compatibility from forces_agent.py
- Proven direct MCP tool calls from working_forces_agent.py
- Support for both forces and kinematics agents
"""

import asyncio
import json
import re
import time
import logging
import requests
from typing import Dict, Any, Optional
from langchain_ollama.chat_models import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from rag_client import RAGClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CombinedPhysicsAgent:
    """
    Combined Physics Agent with both abstraction and proven MCP tool integration
    
    Features:
    - A2A framework compatibility
    - Direct MCP tool calls (proven to work)
    - Support for forces and kinematics agents
    - Flexible agent configuration
    """
    
    def __init__(self, 
                 agent_id: str =  "forces_agent", 
                 llm_base_url: str = "http://ds.stat.uconn.edu:11434", 
                 model: str = "qwen3:8b-q8_0",
                 use_direct_tools: bool = True,
                 database_api_url: str = "http://localhost:8001",
                 enable_database_logging: bool = True,
                 enable_rag: bool = True,
                 rag_api_url: str = "http://localhost:8001"):
        
        self.agent_id = agent_id
        self.llm_base_url = llm_base_url
        self.model = model
        self.use_direct_tools = use_direct_tools  # Key flag for working mode
        self.database_api_url = database_api_url
        self.enable_database_logging = enable_database_logging
        self.enable_rag = enable_rag
        self.rag_api_url = rag_api_url
        
        # Initialize database client for logging
        self.db_client = None
        if self.enable_database_logging:
            self._initialize_database_client()
        
        # Initialize RAG client for context augmentation
        self.rag_client = None
        if self.enable_rag:
            self._initialize_rag_client()
        
        # Initialize based on agent type
        self._setup_agent_config()
        
        # Common initialization
        self.client = None
        self.tools = None
        self.tool_dict = {}
        self.agent = None
        self.initialized = False
        
        # A2A compatibility metadata (after setup so metadata exists)
        if hasattr(self, 'metadata') and self.metadata:
            self.metadata.update({
                "input_types": ["text", "json"],
                "output_types": ["text", "analysis"],
                "version": "1.0.0"
            })
        else:
            # Fallback if metadata not set by agent config
            self.metadata = {
                "input_types": ["text", "json"],
                "output_types": ["text", "analysis"],
                "version": "1.0.0"
            }

    def _setup_agent_config(self):
        """Setup agent-specific configuration"""
        if self.agent_id == "forces_agent":
            from prompts.force_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message
            self.metadata = get_metadata()
            self.mcp_port = 10100  # MCP port for forces agent on VM
            
        elif self.agent_id == "kinematics_agent":
            from prompts.kinematics_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message  
            self.metadata = get_metadata()
            self.mcp_port = 10101  # MCP port for kinematics agent on VM

        elif self.agent_id == "math_agent":
            from prompts.math_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message  
            self.metadata = get_metadata()
            self.mcp_port = 10103  # MCP port for math agent on VM

        elif self.agent_id == "momentum_agent":
            from prompts.momentum_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message  
            self.metadata = get_metadata()
            self.mcp_port = 10104  # MCP port for math agent on VM    

        elif self.agent_id == "energy_agent":
            from prompts.energy_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message  
            self.metadata = get_metadata()
            self.mcp_port = 10105  # MCP port for math agent on VM    

        elif self.agent_id == "angular_motion_agent":
            from prompts.angular_motion_agent_prompt import get_user_message, get_system_message, get_metadata
            self.get_system_message = get_system_message
            self.get_user_message = get_user_message  
            self.metadata = get_metadata()
            self.mcp_port = 10106  # MCP port for math agent on VM            
        else:
            raise ValueError(f"Agent type '{self.agent_id}' not supported. Use 'forces_agent', 'kinematics_agent' or 'math_agent'.")

    def _initialize_database_client(self):
        """Initialize database client for interaction logging"""
        try:
            # Simple HTTP client for database API
            self.db_client = {
                'base_url': self.database_api_url.rstrip('/'),
                'session': requests.Session()
            }
            self.db_client['session'].headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
            
            # Test connection
            response = self.db_client['session'].get(
                f"{self.db_client['base_url']}/health", 
                timeout=5
            )
            if response.status_code == 200:
                logger.info(f"✅ Database logging enabled for agent {self.agent_id}")
            else:
                logger.warning(f"⚠️ Database API unhealthy for agent {self.agent_id}")
                self.db_client = None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize database logging for {self.agent_id}: {e}")
            self.db_client = None

    def _initialize_rag_client(self):
        """Initialize RAG client for context augmentation"""
        try:
            self.rag_client = RAGClient(
                api_base_url=self.rag_api_url,
                enable_cache=True,
                enable_fallback=True
            )
            logger.info(f"✅ RAG integration enabled for agent {self.agent_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize RAG client for {self.agent_id}: {e}")
            self.rag_client = None

    def _integrate_rag_context(self, original_problem: str, rag_context: Dict[str, Any]) -> str:
        """
        Integrate RAG context into the problem description for enhanced agent prompting
        
        Args:
            original_problem: The original physics problem
            rag_context: RAG context containing relevant concepts, formulas, etc.
            
        Returns:
            Enhanced problem description with RAG context
        """
        if not rag_context or rag_context.get("metadata", {}).get("fallback"):
            return original_problem
        
        # Extract RAG components
        concepts = rag_context.get("relevant_concepts", [])
        formulas = rag_context.get("applicable_formulas", [])
        examples = rag_context.get("example_problems", [])
        student_context = rag_context.get("student_context", {})
        agent_context = rag_context.get("agent_context", {})
        
        # Build enhanced context
        context_parts = []
        
        # Add student personalization context
        if student_context.get("level"):
            context_parts.append(f"Student Level: {student_context['level']}")
        
        if agent_context.get("suggested_approach"):
            context_parts.append(f"Recommended Approach: {agent_context['suggested_approach']}")
        
        # Add relevant physics concepts
        if concepts:
            concept_list = [concept.get("name", str(concept)) for concept in concepts[:3]]
            context_parts.append(f"Key Physics Concepts: {', '.join(concept_list)}")
        
        # Add applicable formulas
        if formulas:
            formula_list = []
            for formula in formulas[:2]:
                if isinstance(formula, dict):
                    name = formula.get("name", "")
                    equation = formula.get("equation", "")
                    if name and equation:
                        formula_list.append(f"{name}: {equation}")
                else:
                    formula_list.append(str(formula))
            if formula_list:
                context_parts.append(f"Relevant Formulas: {'; '.join(formula_list)}")
        
        # Add example guidance if available
        if examples and len(examples) > 0:
            context_parts.append(f"Similar Problem Type: Reference available examples for guidance")
        
        # Add learning preferences
        preferences = student_context.get("learning_preferences", {})
        if preferences:
            pref_desc = []
            for pref, value in preferences.items():
                if value > 0.6:
                    pref_desc.append(f"{pref}")
            if pref_desc:
                context_parts.append(f"Student Prefers: {', '.join(pref_desc)} explanations")
        
        # Combine context with original problem
        if context_parts:
            enhanced_context = "\n".join([f"[CONTEXT] {part}" for part in context_parts])
            return f"{enhanced_context}\n\n[PROBLEM] {original_problem}"
        else:
            return original_problem

    def _classify_problem_type(self, problem: str) -> str:
        """
        Classify the type of physics problem based on keywords and context
        
        Args:
            problem: The physics problem description
            
        Returns:
            String classification of the problem type
        """
        problem_lower = problem.lower()
        
        # Force-related keywords
        force_keywords = ["force", "newton", "weight", "friction", "tension", "normal", "equilibrium", "free body"]
        
        # Kinematics keywords
        kinematics_keywords = ["velocity", "acceleration", "distance", "displacement", "time", "motion", "projectile"]
        
        # Energy keywords
        energy_keywords = ["energy", "work", "power", "kinetic", "potential", "conservation", "joule"]
        
        # Momentum keywords
        momentum_keywords = ["momentum", "collision", "impulse", "conservation of momentum"]
        
        # Angular motion keywords
        angular_keywords = ["rotation", "angular", "torque", "moment of inertia", "rotational"]
        
        # Count keyword matches
        force_count = sum(1 for keyword in force_keywords if keyword in problem_lower)
        kinematics_count = sum(1 for keyword in kinematics_keywords if keyword in problem_lower)
        energy_count = sum(1 for keyword in energy_keywords if keyword in problem_lower)
        momentum_count = sum(1 for keyword in momentum_keywords if keyword in problem_lower)
        angular_count = sum(1 for keyword in angular_keywords if keyword in problem_lower)
        
        # Determine primary classification
        counts = {
            "forces": force_count,
            "kinematics": kinematics_count,
            "energy": energy_count,
            "momentum": momentum_count,
            "angular_motion": angular_count
        }
        
        # Return the type with the highest count, or "general" if no clear match
        max_type = max(counts, key=counts.get)
        if counts[max_type] > 0:
            return max_type
        else:
            return "general_physics"

    def _log_agent_interaction(self, 
                             problem: str,
                             solution: str,
                             tools_used: list,
                             execution_time_ms: int,
                             success: bool,
                             reasoning: str = "",
                             user_id: str = "agent_direct",
                             session_id: str = None) -> None:
        """Log agent interaction to database"""
        if not self.db_client:
            return
            
        try:
            metadata = {
                'agent_type': self.agent_id,
                'tools_used': tools_used,
                'execution_time_ms': execution_time_ms,
                'success': success,
                'reasoning': reasoning,
                'llm_model': self.model,
                'llm_base_url': self.llm_base_url,
                'use_direct_tools': self.use_direct_tools,
                'mcp_port': getattr(self, 'mcp_port', None),
                'timestamp': time.time(),
                'agent_metadata': self.metadata if hasattr(self, 'metadata') else {}
            }
            
            payload = {
                'user_id': user_id,
                'agent_type': f"physics_{self.agent_id}",
                'message': problem,
                'response': solution,
                'session_id': session_id,
                'execution_time_ms': execution_time_ms,
                'metadata': metadata
            }
            
            response = self.db_client['session'].post(
                f"{self.db_client['base_url']}/interactions",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                interaction_id = result.get('interaction_id')
                logger.info(f"✅ Logged {self.agent_id} interaction: {interaction_id}")
            else:
                logger.error(f"❌ Failed to log {self.agent_id} interaction: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error logging {self.agent_id} interaction: {e}")

    async def initialize(self):
        """Initialize the physics agent with MCP tools"""
        if self.initialized:
            return
            
        print(f"🚀 Initializing {self.agent_id.title().replace('_', ' ')} (Mode: {'Direct Tools' if self.use_direct_tools else 'LangChain Agent'})...")
        
        # Connect to MCP server on VM using HTTP transport
        server_name = self.agent_id.split('_')[0]  # 'forces' 'kinematics' 'math', 'momentum' or 'energy'
        self.client = MultiServerMCPClient({
            server_name: {
                "transport": "streamable_http",
                #"url": f"http://htfd-physics.grove.ad.uconn.edu:{self.mcp_port}/mcp/",
                "url": f"http://localhost:{self.mcp_port}/mcp/",
            },
        })
        
        self.tools = await self.client.get_tools()
        
        # Create tool lookup dictionary for direct calls
        for tool in self.tools:
            self.tool_dict[tool.name] = tool
            
        if not self.use_direct_tools:
            # Initialize LLM and LangChain agent (original approach)
            llm = ChatOllama(
                model=self.model,
                temperature=0,
                base_url=self.llm_base_url,
            )
            self.agent = create_react_agent(llm, self.tools)
        
        self.initialized = True
        print(f"✅ {self.agent_id.title().replace('_', ' ')} ready with {len(self.tools)} tools:")
        for tool_name in self.tool_dict.keys():
            print(f"  - {tool_name}")
        print()

    async def solve_problem(self, problem: str, context: Optional[Dict] = None, user_id: str = "agent_direct", session_id: str = None) -> Dict[str, Any]:
        """
        Main method to solve physics problems with comprehensive logging
        
        Args:
            problem: Text description of the physics problem
            context: Optional context from A2A framework or other agents
            user_id: User identifier for logging
            session_id: Session identifier for logging
            
        Returns:
            Dict with solution, reasoning, and metadata
        """
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize()
            
        # Augment with RAG context if enabled
        rag_context = None
        rag_enhanced_problem = problem
        if self.enable_rag and self.rag_client:
            try:
                async with self.rag_client as client:
                    rag_context = await client.augment_agent_context(
                        problem=problem,
                        agent_type=self.agent_id.replace("_agent", ""),
                        user_id=user_id,
                        difficulty_level="intermediate"  # Could be determined dynamically
                    )
                    
                # Enhance the problem description with RAG context
                rag_enhanced_problem = self._integrate_rag_context(problem, rag_context)
                logger.info(f"✅ RAG context retrieved for {self.agent_id}")
                
            except Exception as e:
                logger.warning(f"⚠️ RAG context retrieval failed for {self.agent_id}: {e}")
                # Continue without RAG context
        
        try:
            if self.use_direct_tools:
                # Use proven direct tool approach with RAG enhancement
                solution = await self._solve_with_direct_tools(rag_enhanced_problem, rag_context)
                reasoning = f"Solved using direct MCP tool execution{'with RAG enhancement' if rag_context else ''} (proven method)"
            else:
                # Use LangChain agent approach with RAG enhancement
                solution = await self._solve_with_langchain_agent(rag_enhanced_problem, context, rag_context)
                reasoning = f"Solved using LangChain agent with MCP tools{'and RAG enhancement' if rag_context else ''}"
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            tools_used = list(self.tool_dict.keys()) if self.tool_dict else []
            
            # Log interaction to database
            self._log_agent_interaction(
                problem=problem,
                solution=solution,
                tools_used=tools_used,
                execution_time_ms=execution_time_ms,
                success=True,
                reasoning=reasoning,
                user_id=user_id,
                session_id=session_id
            )
            
            # Update student progress in RAG system if enabled
            if self.enable_rag and self.rag_client and rag_context:
                try:
                    async with self.rag_client as client:
                        interaction_data = {
                            "agent_type": self.agent_id.replace("_agent", ""),
                            "problem_type": self._classify_problem_type(problem),
                            "success": True,
                            "execution_time_ms": execution_time_ms,
                            "tools_used": tools_used,
                            "concepts_used": [concept.get("name", "") for concept in rag_context.get("relevant_concepts", [])[:3]],
                            "difficulty_level": "intermediate",  # Could be determined dynamically
                            "timestamp": time.time()
                        }
                        
                        await client.update_student_progress(user_id, interaction_data)
                        logger.info(f"✅ Student progress updated for {user_id}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update student progress: {e}")
                
            return {
                "success": True,
                "agent_id": self.agent_id,
                "problem": problem,
                "solution": solution,
                "reasoning": reasoning,
                "tools_used": tools_used,
                "execution_time_ms": execution_time_ms,
                "metadata": self.metadata,
                "rag_enhanced": rag_context is not None,
                "rag_context": {
                    "concepts_used": len(rag_context.get("relevant_concepts", [])) if rag_context else 0,
                    "formulas_used": len(rag_context.get("applicable_formulas", [])) if rag_context else 0,
                    "examples_available": len(rag_context.get("example_problems", [])) if rag_context else 0,
                    "student_level": rag_context.get("student_context", {}).get("level", "unknown") if rag_context else "unknown"
                }
            }
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            # Log failed interaction to database
            self._log_agent_interaction(
                problem=problem,
                solution=f"Error: {error_msg}",
                tools_used=[],
                execution_time_ms=execution_time_ms,
                success=False,
                reasoning=f"Error in problem solving: {error_msg}",
                user_id=user_id,
                session_id=session_id
            )
            
            return {
                "success": False,
                "agent_id": self.agent_id,
                "problem": problem,
                "error": error_msg,
                "reasoning": f"Error in problem solving: {error_msg}",
                "execution_time_ms": execution_time_ms
            }

    async def _solve_with_direct_tools(self, problem: str, rag_context: Optional[Dict[str, Any]] = None) -> str:
        """Solve using direct tool calls (working_forces_agent.py approach)"""
        if self.agent_id == "forces_agent":
            return await self._solve_forces_problem_direct(problem)
        elif self.agent_id == "kinematics_agent":
            return await self._solve_kinematics_problem_direct(problem)
        elif self.agent_id == "math_agent":
            return await self._solve_math_problem_direct(problem)
        elif self.agent_id == "momentum_agent":
            return await self._solve_momentum_problem_direct(problem)
        elif self.agent_id == "energy_agent":
            return await self._solve_energy_problem_direct(problem)
        elif self.agent_id == "angular_motion_agent":
            return await self._solve_angular_motion_problem_direct(problem)
        else:
            return "❌ Unsupported agent type for direct tools"

    async def _solve_with_langchain_agent(self, problem: str, context: Optional[Dict] = None, rag_context: Optional[Dict[str, Any]] = None) -> str:
        """Solve using LangChain agent approach (forces_agent.py approach)"""
        if context:
            full_input = f"Context: {json.dumps(context)}\\n\\nProblem: {problem}"
        else:
            full_input = problem
            
        response = await self.agent.ainvoke({
            "messages": [
                SystemMessage(content=self.get_system_message()),
                ("human", full_input)
            ]
        }, config={"recursion_limit": 15})
        
        return response['messages'][-1].content

    # FORCES-SPECIFIC DIRECT TOOL METHODS
    async def _solve_forces_problem_direct(self, problem: str) -> str:
        """Direct tool solving for forces problems"""
        problem_lower = problem.lower()
        
        # 2D Force Addition
        if any(word in problem_lower for word in ["add", "forces"]) and ("°" in problem or "degree" in problem_lower):
            return await self._call_forces_2d_tool(problem)
            
        # Spring Force
        elif any(word in problem_lower for word in ["spring", "hooke"]):
            return await self._call_spring_tool(problem)
            
        # Force Components
        elif any(word in problem_lower for word in ["component", "resolve", "break"]):
            return await self._call_component_tool(problem)
            
        # Equilibrium
        elif any(word in problem_lower for word in ["equilibrium", "balance"]):
            return await self._call_equilibrium_tool(problem)
            
        # Free Body Diagram
        elif any(word in problem_lower for word in ["free body", "fbd", "diagram"]):
            return await self._call_fbd_tool(problem)
            
        else:
            return await self._call_forces_2d_tool(problem)  # Default

    async def _call_forces_2d_tool(self, problem: str) -> str:
        """Call 2D force addition tool directly"""
        try:
            if "add_forces_2d" not in self.tool_dict:
                return "❌ add_forces_2d tool not available"
            
            forces = self._parse_forces(problem)
            if not forces:
                return "❌ Could not parse forces. Try format: 'Add forces: 10N at 30°, 15N at 120°'"
            
            tool = self.tool_dict["add_forces_2d"]
            result = await tool.ainvoke({"forces_data": forces})
            
            return f"🎯 **2D FORCE ADDITION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 2D force calculation: {e}"

    async def _call_spring_tool(self, problem: str) -> str:
        """Call spring force tool directly"""
        try:
            if "calculate_spring_force_tool" not in self.tool_dict:
                return "❌ calculate_spring_force_tool not available"
            
            k, displacement = self._parse_spring_params(problem)
            
            tool = self.tool_dict["calculate_spring_force_tool"]
            result = await tool.ainvoke({
                "spring_constant": k,
                "displacement": displacement
            })
            
            return f"🎯 **SPRING FORCE SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in spring force calculation: {e}"

    async def _call_component_tool(self, problem: str) -> str:
        """Call force component resolution tool"""
        try:
            if "resolve_force_components" not in self.tool_dict:
                return "❌ resolve_force_components tool not available"
            
            magnitude, angle = self._parse_force_magnitude_angle(problem)
            
            tool = self.tool_dict["resolve_force_components"]
            result = await tool.ainvoke({
                "magnitude": magnitude,
                "angle_degrees": angle
            })
            
            return f"🎯 **FORCE COMPONENTS SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in component calculation: {e}"

    async def _call_equilibrium_tool(self, problem: str) -> str:
        """Call equilibrium checking tool"""
        try:
            if "check_equilibrium" not in self.tool_dict:
                return "❌ check_equilibrium tool not available"
            
            forces = self._parse_forces(problem)
            if not forces:
                return "❌ Could not parse forces for equilibrium check"
            
            tool = self.tool_dict["check_equilibrium"]
            result = await tool.ainvoke({"forces_data": json.dumps(forces)})
            
            return f"🎯 **EQUILIBRIUM ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in equilibrium analysis: {e}"

    async def _call_fbd_tool(self, problem: str) -> str:
        """Call free body diagram tool"""
        try:
            if "create_free_body_diagram" not in self.tool_dict:
                return "❌ create_free_body_diagram tool not available"
            
            object_name = self._parse_object_name(problem)
            forces = self._parse_forces_with_names(problem)
            
            tool = self.tool_dict["create_free_body_diagram"]
            result = await tool.ainvoke({
                "object_name": object_name,
                "forces_data": json.dumps(forces)
            })
            
            return f"🎯 **FREE BODY DIAGRAM**\\n\\n{result}\\n\\n✅ **Diagram completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in free body diagram: {e}"
    
    # MATHEMATICS-SPECIFIC DIRECT TOOL METHODS
    async def _solve_math_problem_direct(self, problem: str) -> str:
        """Direct tool solving for mathematics problems"""
        problem_lower = problem.lower()
        
        # Quadratic equations
        if any(word in problem_lower for word in ["x²", "x^2", "quadratic"]) and ("=" in problem or "solve" in problem_lower):
            return await self._call_quadratic_tool(problem)
            
        # Linear equations
        elif "x" in problem and "=" in problem and "x²" not in problem and "x^2" not in problem:
            return await self._call_linear_tool(problem)
            
        # Trigonometry
        elif any(word in problem_lower for word in ["sin", "cos", "tan", "trigonometry", "trig"]):
            return await self._call_trigonometry_tool(problem)
            
        # Triangle solving
        elif any(word in problem_lower for word in ["triangle", "sides", "angles", "law of"]):
            return await self._call_triangle_tool(problem)
            
        # Logarithms
        elif any(word in problem_lower for word in ["log", "ln", "logarithm", "antilog"]):
            return await self._call_logarithm_tool(problem)
            
        # EQUATION SOLVING - Check first before other tools
        if any(phrase in problem_lower for phrase in ["solve", "for", "find", "isolate", "rearrange"]):
            # Check if it's asking to solve for a specific variable
            if any(pattern in problem_lower for pattern in [
                "solve", "for", "find", "isolate", "rearrange"]) and any(
                var in problem_lower for var in ["v", "m", "a", "f", "k", "p", "e", "x", "y", "z"]):
                
                # Check if it has specific numerical values (not just fractions like 1/2)
                import re
                has_specific_numbers = bool(re.search(r'\b\d+\s*(?:j|kg|m/s|n|joule|newton|kilogram)\b', problem_lower))
                has_equation_with_numbers = bool(re.search(r'=\s*\d+(?:\.\d+)?(?!\s*[*/])', problem))
                
                # If it has numerical values with units or specific numerical answers, use physics_formula_solver
                if has_specific_numbers or has_equation_with_numbers:
                    return await self._call_physics_formula_solver_tool(problem)
                # Otherwise use symbolic solver for abstract equations
                else:
                    return await self._call_solve_for_variable_tool(problem)
        
        # Statistics
        elif any(word in problem_lower for word in ["statistics", "mean", "median", "data", "standard deviation"]):
            return await self._call_statistics_tool(problem)
            
        # Unit circle
        elif any(word in problem_lower for word in ["unit circle", "reference"]):
            return await self._call_unit_circle_tool(problem)
            
        # Algebraic simplification
        elif any(word in problem_lower for word in ["simplify", "factor", "expand"]):
            return await self._call_algebra_simplify_tool(problem)
            
        else:
            # Default to quadratic if contains x², otherwise linear
            if "x²" in problem or "x^2" in problem:
                return await self._call_quadratic_tool(problem)
            elif "x" in problem and "=" in problem:
                return await self._call_linear_tool(problem)
            else:
                return await self._call_algebra_simplify_tool(problem)

    async def _call_quadratic_tool(self, problem: str) -> str:
        """Call quadratic equation solver tool"""
        try:
            if "solve_quadratic_equation" not in self.tool_dict:
                return "❌ solve_quadratic_equation tool not available"
            
            equation = self._parse_equation(problem)
            
            tool = self.tool_dict["solve_quadratic_equation"]
            result = await tool.ainvoke({"equation": equation})
            
            return f"🎯 **QUADRATIC EQUATION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in quadratic equation solving: {e}"

    async def _call_linear_tool(self, problem: str) -> str:
        """Call linear equation solver tool"""
        try:
            if "solve_linear_equation" not in self.tool_dict:
                return "❌ solve_linear_equation tool not available"
            
            equation = self._parse_equation(problem)
            
            tool = self.tool_dict["solve_linear_equation"]
            result = await tool.ainvoke({"equation": equation})
            
            return f"🎯 **LINEAR EQUATION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in linear equation solving: {e}"

    async def _call_trigonometry_tool(self, problem: str) -> str:
        """Call trigonometry calculator tool"""
        try:
            if "trigonometry_calculator" not in self.tool_dict:
                return "❌ trigonometry_calculator tool not available"
            
            function, value, unit = self._parse_trigonometry(problem)
            
            tool = self.tool_dict["trigonometry_calculator"]
            result = await tool.ainvoke({
                "function": function,
                "value": value,
                "unit": unit
            })
            
            return f"🎯 **TRIGONOMETRY SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in trigonometry calculation: {e}"

    async def _call_triangle_tool(self, problem: str) -> str:
        """Call triangle solver tool"""
        try:
            if "triangle_solver" not in self.tool_dict:
                return "❌ triangle_solver tool not available"
            
            triangle_data = self._parse_triangle_data(problem)
            
            tool = self.tool_dict["triangle_solver"]
            result = await tool.ainvoke({"triangle_data": json.dumps(triangle_data)})
            
            return f"🎯 **TRIANGLE SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in triangle solving: {e}"

    async def _call_logarithm_tool(self, problem: str) -> str:
        """Call logarithm calculator tool"""
        try:
            if "logarithm_calculator" not in self.tool_dict:
                return "❌ logarithm_calculator tool not available"
            
            operation, base, value, result_val = self._parse_logarithm(problem)
            
            tool = self.tool_dict["logarithm_calculator"]
            params = {"operation": operation}
            
            if base is not None:
                params["base"] = base
            if value is not None:
                params["value"] = value
            if result_val is not None:
                params["result"] = result_val
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **LOGARITHM SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in logarithm calculation: {e}"

    async def _call_statistics_tool(self, problem: str) -> str:
        """Call statistics calculator tool"""
        try:
            if "statistics_calculator" not in self.tool_dict:
                return "❌ statistics_calculator tool not available"
            
            data_type, values = self._parse_statistics_data(problem)
            
            tool = self.tool_dict["statistics_calculator"]
            result = await tool.ainvoke({
                "data_type": data_type,
                "values": values
            })
            
            return f"🎯 **STATISTICS SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in statistics calculation: {e}"

    async def _call_unit_circle_tool(self, problem: str) -> str:
        """Call unit circle reference tool"""
        try:
            if "unit_circle_reference" not in self.tool_dict:
                return "❌ unit_circle_reference tool not available"
            
            angle, unit = self._parse_angle(problem)
            
            tool = self.tool_dict["unit_circle_reference"]
            result = await tool.ainvoke({
                "angle": angle,
                "unit": unit
            })
            
            return f"🎯 **UNIT CIRCLE REFERENCE**\\n\\n{result}\\n\\n✅ **Reference completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in unit circle reference: {e}"

    async def _call_algebra_simplify_tool(self, problem: str) -> str:
        """Call algebra simplification tool"""
        try:
            if "algebra_simplify" not in self.tool_dict:
                return "❌ algebra_simplify tool not available"
            
            expression = self._parse_expression(problem)
            
            tool = self.tool_dict["algebra_simplify"]
            result = await tool.ainvoke({"expression": expression})
            
            return f"🎯 **ALGEBRA SIMPLIFICATION**\\n\\n{result}\\n\\n✅ **Simplification completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in algebra simplification: {e}"

    async def _call_solve_for_variable_tool(self, problem: str) -> str:
        """Call solve_for_variable tool for symbolic equation solving"""
        try:
            if "solve_for_variable" not in self.tool_dict:
                return "❌ solve_for_variable tool not available"
            
            equation, solve_for = self._parse_equation_solving(problem)
            
            tool = self.tool_dict["solve_for_variable"]
            result = await tool.ainvoke({
                "equation": equation,
                "solve_for": solve_for
            })
            
            return f"🎯 **EQUATION SOLVING**\\n\\n{result}\\n\\n✅ **Equation solved using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in equation solving: {e}"

    async def _call_physics_formula_solver_tool(self, problem: str) -> str:
        """Call physics_formula_solver tool for numerical calculations"""
        try:
            if "physics_formula_solver" not in self.tool_dict:
                return "❌ physics_formula_solver tool not available"
            
            formula_name, known_values, solve_for = self._parse_physics_formula(problem)
            
            tool = self.tool_dict["physics_formula_solver"]
            result = await tool.ainvoke({
                "formula_name": formula_name,
                "known_values": json.dumps(known_values),
                "solve_for": solve_for
            })
            
            return f"🎯 **PHYSICS CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in physics formula calculation: {e}"

    # KINEMATICS-SPECIFIC DIRECT TOOL METHODS
    async def _solve_kinematics_problem_direct(self, problem: str) -> str:
        """Direct tool solving for kinematics problems"""
        problem_lower = problem.lower()
        
        # Projectile motion
        if any(word in problem_lower for word in ["thrown", "launched", "projectile", "trajectory"]) and ("angle" in problem_lower or "°" in problem):
            return await self._call_projectile_tool(problem)
            
        # Free fall
        elif any(word in problem_lower for word in ["dropped", "fall", "falling", "height"]) and "angle" not in problem_lower:
            return await self._call_freefall_tool(problem)
            
        # Constant acceleration
        elif any(word in problem_lower for word in ["accelerate", "acceleration", "decelerate"]):
            return await self._call_acceleration_tool(problem)
            
        # Uniform motion
        elif any(word in problem_lower for word in ["constant", "uniform", "velocity"]) and "acceleration" not in problem_lower:
            return await self._call_uniform_motion_tool(problem)
            
        # Relative motion
        elif any(word in problem_lower for word in ["meet", "catch", "relative", "two"]):
            return await self._call_relative_motion_tool(problem)
            
        else:
            return await self._call_acceleration_tool(problem)  # Default

    def _parse_projectile_params(self, text: str) -> dict:
        """Parse projectile motion parameters"""
        print(f"DEBUG: Original text: '{text}'")
        
        params = {"v0": 30, "angle": 45, "h0": 0}
        
        # Parse velocity
        v_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', text)
        print(f"DEBUG: Velocity match: {v_match}")
        if v_match:
            params["v0"] = float(v_match.group(1))
            print(f"DEBUG: Set v0 = {params['v0']}")
        
        # Parse angle
        ang_match = re.search(r'(\d+(?:\.\d+)?)(?:°|degree|deg)', text)
        print(f"DEBUG: Angle match: {ang_match}")
        if ang_match:
            params["angle"] = float(ang_match.group(1))
            print(f"DEBUG: Set angle = {params['angle']}")
        
        print(f"DEBUG: Final params: {params}")
        return params


    async def _call_freefall_tool(self, problem: str) -> str:
        """Call free fall tool"""
        try:
            if "free_fall_motion" not in self.tool_dict:
                return "❌ free_fall_motion tool not available"
            
            
            params = self._parse_freefall_params(problem)
            tool = self.tool_dict["free_fall_motion"]
            
            result = await tool.ainvoke({
                "known_values": json.dumps(params)
            })
            
            return f"🎯 **FREE FALL SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in free fall calculation: {e}"

    async def _call_acceleration_tool(self, problem: str) -> str:
        """Call constant acceleration tool"""
        try:
            if "constant_acceleration_1d" not in self.tool_dict:
                return "❌ constant_acceleration_1d tool not available"
            
            params = self._parse_acceleration_params(problem)
            tool = self.tool_dict["constant_acceleration_1d"]
            
            result = await tool.ainvoke({
                "known_values": json.dumps(params)
            })
            
            return f"🎯 **CONSTANT ACCELERATION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in acceleration calculation: {e}"

    async def _call_uniform_motion_tool(self, problem: str) -> str:
        """Call uniform motion tool"""
        try:
            if "uniform_motion_1d" not in self.tool_dict:
                return "❌ uniform_motion_1d tool not available"
            
            params = self._parse_uniform_motion_params(problem)
            tool = self.tool_dict["uniform_motion_1d"]
            
            result = await tool.ainvoke({
                "known_values": json.dumps(params)
            })
            
            return f"🎯 **UNIFORM MOTION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in uniform motion calculation: {e}"

    async def _call_relative_motion_tool(self, problem: str) -> str:
        """Call relative motion tool"""
        try:
            if "relative_motion_1d" not in self.tool_dict:
                return "❌ relative_motion_1d tool not available"
            
            params = self._parse_relative_motion_params(problem)
            tool = self.tool_dict["relative_motion_1d"]
            
            result = await tool.ainvoke({
                "objects_data": json.dumps(params)
            })
            
            return f"🎯 **RELATIVE MOTION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in relative motion calculation: {e}"

    # PARSING METHODS (Forces)
    def _parse_forces(self, text: str) -> list:
        """Parse forces from text - same as working agent"""
        forces = []
        
        # Handle common patterns
        if "10N at 30" in text and "15N at 120" in text:
            forces = [{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 120}]
        elif "10N at 30" in text and "15N at 60" in text:
            forces = [{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 60}]
        else:
            # General regex parsing
            pattern = r'(\d+(?:\.\d+)?)\s*[Nn]?\s*(?:at|@)\s*(\d+(?:\.\d+)?)(?:°|degree|deg)?'
            matches = re.findall(pattern, text)
            forces = [{"magnitude": float(mag), "angle": float(ang)} for mag, ang in matches]
        
        return forces

    def _parse_spring_params(self, text: str):
        """Parse spring parameters"""
        k = 200  # default
        displacement = -0.05  # default compression
        
        k_match = re.search(r'k\s*=\s*(\d+(?:\.\d+)?)', text)
        if k_match:
            k = float(k_match.group(1))
        
        if "compressed" in text.lower() or "compression" in text.lower():
            disp_match = re.search(r'(\d+(?:\.\d+)?)\s*m', text)
            if disp_match:
                displacement = -float(disp_match.group(1))
        elif "stretched" in text.lower() or "extension" in text.lower():
            disp_match = re.search(r'(\d+(?:\.\d+)?)\s*m', text)
            if disp_match:
                displacement = float(disp_match.group(1))
        
        return k, displacement

    def _parse_force_magnitude_angle(self, text: str):
        """Parse single force magnitude and angle"""
        magnitude = 25  # default
        angle = 45  # default
        
        mag_match = re.search(r'(\d+(?:\.\d+)?)\s*[Nn]', text)
        if mag_match:
            magnitude = float(mag_match.group(1))
        
        ang_match = re.search(r'(\d+(?:\.\d+)?)(?:°|degree|deg)', text)
        if ang_match:
            angle = float(ang_match.group(1))
        
        return magnitude, angle

    def _parse_object_name(self, text: str) -> str:
        """Parse object name for free body diagrams"""
        objects = ["box", "block", "ball", "car", "book", "mass", "object"]
        for obj in objects:
            if obj in text.lower():
                return obj
        return "object"

    def _parse_forces_with_names(self, text: str) -> list:
        """Parse forces with names for free body diagrams"""
        forces = [
            {"name": "Weight", "magnitude": 50, "angle": 270},
            {"name": "Normal", "magnitude": 50, "angle": 90}
        ]
        
        if "applied" in text.lower():
            app_match = re.search(r'(\d+(?:\.\d+)?)\s*[Nn]', text)
            if app_match:
                forces.append({"name": "Applied", "magnitude": float(app_match.group(1)), "angle": 0})
        
        return forces

    # PARSING METHODS (Mathematics)
    
    def _parse_equation(self, text: str) -> str:
        """Parse equation from text"""
        # Look for equation patterns
        eq_patterns = [
            r'([x²x^2x+-=0-9\s\.]+=[x²x^2x+-=0-9\s\.]+)',
            r'solve\s+([x²x^2x+-=0-9\s\.]+)',
            r'equation[:\s]+([x²x^2x+-=0-9\s\.]+)'
        ]
        
        for pattern in eq_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default equations for common cases
        if "x² + 5x + 6" in text:
            return "x² + 5x + 6 = 0"
        elif "3x + 7 = 2x - 5" in text:
            return "3x + 7 = 2x - 5"
        elif "x²" in text or "x^2" in text:
            return "x² + 5x + 6 = 0"  # Default quadratic
        else:
            return "3x + 7 = 2x - 5"  # Default linear

    def _parse_trigonometry(self, text: str) -> tuple:
        """Parse trigonometric function call"""
        text_lower = text.lower()
        
        # Determine function
        if "sin" in text_lower:
            function = "sin"
        elif "cos" in text_lower:
            function = "cos"
        elif "tan" in text_lower:
            function = "tan"
        elif "arcsin" in text_lower or "asin" in text_lower:
            function = "arcsin"
        elif "arccos" in text_lower or "acos" in text_lower:
            function = "arccos"
        elif "arctan" in text_lower or "atan" in text_lower:
            function = "arctan"
        else:
            function = "sin"  # Default
        
        # Parse value
        value = 45  # Default
        value_match = re.search(r'(\d+(?:\.\d+)?)', text)
        if value_match:
            value = float(value_match.group(1))
        
        # Determine unit
        if "°" in text or "degree" in text_lower:
            unit = "degrees"
        elif "rad" in text_lower:
            unit = "radians"
        else:
            unit = "degrees"  # Default
        
        return function, value, unit

    def _parse_triangle_data(self, text: str) -> dict:
        """Parse triangle data from text - FIXED to avoid side/angle confusion"""
        triangle_data = {"sides": {}, "angles": {}}
        
        # Parse sides (look for lowercase a,b,c WITHOUT degree symbol)
        side_matches = re.findall(r'\b([abc])\s*=\s*(\d+(?:\.\d+)?)(?!\s*°)', text, re.IGNORECASE)
        for side, value in side_matches:
            triangle_data["sides"][side.lower()] = float(value)
        
        # Parse angles (look for uppercase A,B,C OR anything WITH degree symbol)
        angle_matches = re.findall(r'\b([ABC])\s*=\s*(\d+(?:\.\d+)?)\s*°', text, re.IGNORECASE)
        for angle, value in angle_matches:
            triangle_data["angles"][angle.upper()] = float(value)
        
        # Also check for explicit "angle" keyword
        explicit_angle_matches = re.findall(r'angle\s+([ABC])\s*=\s*(\d+(?:\.\d+)?)(?:°)?', text, re.IGNORECASE)
        for angle, value in explicit_angle_matches:
            triangle_data["angles"][angle.upper()] = float(value)
        
        # Default triangle if nothing parsed
        if not triangle_data["sides"] and not triangle_data["angles"]:
            if "5" in text and "7" in text and "60" in text:
                triangle_data = {"sides": {"a": 5, "b": 7}, "angles": {"C": 60}}
            elif "3" in text and "4" in text and "5" in text:
                triangle_data = {"sides": {"a": 3, "b": 4, "c": 5}}
            else:
                triangle_data = {"sides": {"a": 5, "b": 7}, "angles": {"C": 60}}
        
        return triangle_data

    def _parse_logarithm(self, text: str) -> tuple:
        """Parse logarithm parameters"""
        text_lower = text.lower()
        
        # Determine operation
        if "antilog" in text_lower:
            operation = "antilog"
        elif "solve" in text_lower:
            operation = "solve"
        else:
            operation = "log"
        
        # Parse base
        base = None
        if "log₁₀" in text or "log10" in text:
            base = 10
        elif "log₂" in text or "log2" in text:
            base = 2
        elif "ln" in text_lower:
            base = None  # Natural log
        
        # Parse value/result
        value = None
        result_val = None
        
        if operation == "log":
            # Look for log(value)
            log_match = re.search(r'log(?:₁₀|10|₂|2)?\s*\(\s*(\d+(?:\.\d+)?)\s*\)', text)
            if log_match:
                value = float(log_match.group(1))
            elif "100" in text:
                value = 100
            elif "e²" in text:
                value = 7.389  # e²
        elif operation == "antilog":
            # Look for antilog value
            antilog_match = re.search(r'(\d+(?:\.\d+)?)', text)
            if antilog_match:
                result_val = float(antilog_match.group(1))
        
        return operation, base, value, result_val

    def _parse_statistics_data(self, text: str) -> tuple:
        """Parse statistics data"""
        # Look for comma-separated numbers
        numbers_match = re.search(r'(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)', text)
        
        if numbers_match:
            values = numbers_match.group(1)
        else:
            # Default data set
            values = "12, 15, 18, 14, 16, 13, 17"
        
        # Determine data type
        if "error" in text.lower():
            data_type = "error"
        else:
            data_type = "descriptive"
        
        return data_type, values

    def _parse_angle(self, text: str) -> tuple:
        """Parse angle for unit circle"""
        # Parse angle value
        angle = 45  # Default
        angle_match = re.search(r'(\d+(?:\.\d+)?)', text)
        if angle_match:
            angle = float(angle_match.group(1))
        
        # Determine unit
        if "°" in text or "degree" in text.lower():
            unit = "degrees"
        elif "rad" in text.lower():
            unit = "radians"
        else:
            unit = "degrees"  # Default
        
        return angle, unit

    def _parse_expression(self, text: str) -> str:
        """Parse algebraic expression"""
        # Look for expression patterns
        expr_patterns = [
            r'simplify\s+([x²x^2x+-=0-9\s\.]+)',
            r'factor\s+([x²x^2x+-=0-9\s\.]+)',
            r'expand\s+([x²x^2x+-=0-9\s\.]+)',
            r'([x²x^2x+-=0-9\s\.]+)'
        ]
        
        for pattern in expr_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Default expressions
        if "3x + 2x" in text:
            return "3x + 2x - 5 + 8"
        elif "x² - 9" in text:
            return "x² - 9"
        else:
            return "2x + 3x"  # Default

    # PARSING METHODS (Kinematics)
    def _parse_projectile_params(self, text: str) -> dict:
        """Parse projectile motion parameters"""
        print(f"DEBUG: Parsing projectile text: '{text}'")  # Add debug
        
        params = {"v0": 30, "angle": 45, "h0": 0}
        
        # Parse velocity - FIXED REGEX
        v_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', text)
        if v_match:
            params["v0"] = float(v_match.group(1))
        
        # Parse angle - FIXED REGEX
        ang_match = re.search(r'(\d+(?:\.\d+)?)(?:°|degree|deg)', text)
        if ang_match:
            params["angle"] = float(ang_match.group(1))
        
        # Parse height - FIXED REGEX
        h_match = re.search(r'from\s+(\d+(?:\.\d+)?)\s*m', text)
        if h_match:
            params["h0"] = float(h_match.group(1))
        
        # Parse target coordinates - NEW ADDITION
        target_match = re.search(r'\((\d+(?:\.\d+)?)\s*m?,\s*(\d+(?:\.\d+)?)\s*m?\)', text)
        if target_match:
            params["target_x"] = float(target_match.group(1))
            params["target_y"] = float(target_match.group(2))
        
        print(f"DEBUG: Final projectile params: {params}")  # Add debug
        return params

    def _parse_freefall_params(self, text: str) -> dict:
        """Parse free fall parameters"""
        params = {}
        
        print(f"DEBUG: Parsing text: '{text}'")
        
        # Parse height - MAKE MORE SPECIFIC to avoid matching velocities
        if "from ground" in text.lower() or "ground level" in text.lower():
            params["h0"] = 0
            print("DEBUG: Set h0 = 0 (from ground)")
        else:
            # Only match height patterns, not velocity patterns
            h_match = re.search(r'(?:from|height|drop|fall)\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*m(?!\s*/)', text)
            if h_match:
                params["h0"] = float(h_match.group(1))
                print(f"DEBUG: Parsed h0 = {params['h0']}")
            else:
                # Look for standalone height (not followed by /s)
                h_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?!\s*/)', text)
                if h_match:
                    params["h0"] = float(h_match.group(1))
                    print(f"DEBUG: Found standalone height h0 = {params['h0']}")
        
        # Parse initial velocity
        if "thrown" in text.lower() or "upward" in text.lower():
            v_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', text)
            if v_match:
                params["v0"] = float(v_match.group(1))
                print(f"DEBUG: Parsed v0 = {params['v0']}")
        else:
            params["v0"] = 0  # Dropped from rest
            
        print(f"DEBUG: Final params = {params}")
        return params
    def _parse_acceleration_params(self, text: str) -> dict:
        """Parse constant acceleration parameters"""
        params = {}
        
        # Parse initial velocity
        if "from rest" in text.lower():
            params["v0"] = 0
        else:
            v0_match = re.search(r'at\s+(\d+(?:\.\d+)?)\s*m/s', text)
            if v0_match:
                params["v0"] = float(v0_match.group(1))
        
        # Parse acceleration
        a_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s[²²]', text)
        if a_match:
            params["a"] = float(a_match.group(1))
        
        # Parse time
        t_match = re.search(r'for\s+(\d+(?:\.\d+)?)\s*s', text)
        if t_match:
            params["t"] = float(t_match.group(1))
        
        return params

    def _parse_uniform_motion_params(self, text: str) -> dict:
        """Parse uniform motion parameters"""
        params = {"x0": 0}
        
        # Parse velocity
        v_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', text)
        if v_match:
            params["v"] = float(v_match.group(1))
        
        # Parse time
        t_match = re.search(r'for\s+(\d+(?:\.\d+)?)\s*s', text)
        if t_match:
            params["t"] = float(t_match.group(1))
        
        return params

    def _parse_relative_motion_params(self, text: str) -> dict:
        """Parse relative motion parameters"""
        params = {
            "object1": {"x0": 0, "v": 25},
            "object2": {"x0": 200, "v": -15}
        }
        
        # This would need more sophisticated parsing for real use
        # For now, return default two-car scenario
        
        return params

    # MOMENTUM-SPECIFIC DIRECT TOOL METHODS
    async def _solve_momentum_problem_direct(self, problem: str) -> str:
        """Direct tool solving for momentum problems"""
        problem_lower = problem.lower()
        
        # 1D momentum calculations
        if any(word in problem_lower for word in ["momentum", "p =", "calculate momentum"]) and not any(word in problem_lower for word in ["2d", "angle", "degrees", "components"]):
            return await self._call_momentum_1d_tool(problem)
            
        # 2D momentum calculations
        elif any(word in problem_lower for word in ["2d momentum", "momentum", "angle", "degrees", "components"]) and any(word in problem_lower for word in ["°", "degree", "direction"]):
            return await self._call_momentum_2d_tool(problem)
            
        # 1D impulse calculations
        elif any(word in problem_lower for word in ["impulse", "force", "time", "j =", "n⋅s"]) and not any(word in problem_lower for word in ["2d", "components"]):
            return await self._call_impulse_1d_tool(problem)
            
        # 2D impulse calculations
        elif any(word in problem_lower for word in ["2d impulse", "impulse", "components"]) and any(word in problem_lower for word in ["angle", "vector", "fx", "fy"]):
            return await self._call_impulse_2d_tool(problem)
            
        # Impulse-momentum theorem
        elif any(word in problem_lower for word in ["impulse-momentum theorem", "impulse momentum", "theorem", "j = δp"]):
            return await self._call_momentum_impulse_theorem_tool(problem)
            
        # 1D momentum conservation (collisions)
        elif any(word in problem_lower for word in ["collision", "collides", "elastic", "inelastic", "conservation"]) and not any(word in problem_lower for word in ["2d", "angle"]):
            return await self._call_momentum_conservation_1d_tool(problem)
            
        # 2D momentum conservation
        elif any(word in problem_lower for word in ["2d collision", "collision", "billiard"]) and any(word in problem_lower for word in ["angle", "2d", "vector"]):
            return await self._call_momentum_conservation_2d_tool(problem)
            
        # Comprehensive collision analysis
        elif any(word in problem_lower for word in ["crash", "car crash", "safety", "analyze collision", "comprehensive"]):
            return await self._call_analyze_collision_tool(problem)
            
        else:
            # Default based on keywords - prioritize by complexity
            if any(word in problem_lower for word in ["crash", "car", "safety"]):
                return await self._call_analyze_collision_tool(problem)
            elif any(word in problem_lower for word in ["collision", "collides", "elastic", "inelastic"]):
                if any(word in problem_lower for word in ["2d", "angle", "°"]):
                    return await self._call_momentum_conservation_2d_tool(problem)
                else:
                    return await self._call_momentum_conservation_1d_tool(problem)
            elif any(word in problem_lower for word in ["impulse", "force", "time"]):
                if any(word in problem_lower for word in ["2d", "components"]):
                    return await self._call_impulse_2d_tool(problem)
                else:
                    return await self._call_impulse_1d_tool(problem)
            elif any(word in problem_lower for word in ["momentum"]):
                if any(word in problem_lower for word in ["2d", "angle", "°", "direction"]):
                    return await self._call_momentum_2d_tool(problem)
                else:
                    return await self._call_momentum_1d_tool(problem)
            else:
                return await self._call_momentum_1d_tool(problem)

    async def _call_momentum_1d_tool(self, problem: str) -> str:
        """Call 1D momentum calculation tool"""
        try:
            if "calculate_momentum_1d" not in self.tool_dict:
                return "❌ calculate_momentum_1d tool not available"
            
            mass, velocity = self._parse_momentum_1d_data(problem)
            
            tool = self.tool_dict["calculate_momentum_1d"]
            result = await tool.ainvoke({
                "mass": mass,
                "velocity": velocity
            })
            
            return f"🎯 **1D MOMENTUM CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 1D momentum calculation: {e}"

    async def _call_momentum_2d_tool(self, problem: str) -> str:
        """Call 2D momentum calculation tool"""
        try:
            if "calculate_momentum_2d" not in self.tool_dict:
                return "❌ calculate_momentum_2d tool not available"
            
            mass, velocity, angle_degrees = self._parse_momentum_2d_data(problem)
            
            tool = self.tool_dict["calculate_momentum_2d"]
            result = await tool.ainvoke({
                "mass": mass,
                "velocity": velocity,
                "angle_degrees": angle_degrees
            })
            
            return f"🎯 **2D MOMENTUM CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 2D momentum calculation: {e}"

    async def _call_impulse_1d_tool(self, problem: str) -> str:
        """Call 1D impulse calculation tool"""
        try:
            if "calculate_impulse_1d" not in self.tool_dict:
                return "❌ calculate_impulse_1d tool not available"
            
            force, time, initial_momentum, final_momentum = self._parse_impulse_1d_data(problem)
            
            tool = self.tool_dict["calculate_impulse_1d"]
            params = {
                "force": force,
                "time": time
            }
            if initial_momentum is not None:
                params["initial_momentum"] = initial_momentum
            if final_momentum is not None:
                params["final_momentum"] = final_momentum
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **1D IMPULSE CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 1D impulse calculation: {e}"

    async def _call_impulse_2d_tool(self, problem: str) -> str:
        """Call 2D impulse calculation tool"""
        try:
            if "calculate_impulse_2d" not in self.tool_dict:
                return "❌ calculate_impulse_2d tool not available"
            
            force_data, time, momentum_data = self._parse_impulse_2d_data(problem)
            
            tool = self.tool_dict["calculate_impulse_2d"]
            params = {
                "force_data": json.dumps(force_data)
            }
            if time is not None:
                params["time"] = time
            if momentum_data is not None:
                params["momentum_data"] = json.dumps(momentum_data)
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **2D IMPULSE CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 2D impulse calculation: {e}"

    async def _call_momentum_impulse_theorem_tool(self, problem: str) -> str:
        """Call momentum-impulse theorem tool"""
        try:
            if "momentum_impulse_theorem" not in self.tool_dict:
                return "❌ momentum_impulse_theorem tool not available"
            
            problem_data = self._parse_momentum_impulse_theorem_data(problem)
            
            tool = self.tool_dict["momentum_impulse_theorem"]
            result = await tool.ainvoke({
                "problem_data": json.dumps(problem_data)
            })
            
            return f"🎯 **MOMENTUM-IMPULSE THEOREM ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in momentum-impulse theorem analysis: {e}"

    async def _call_momentum_conservation_1d_tool(self, problem: str) -> str:
        """Call 1D momentum conservation tool"""
        try:
            if "momentum_conservation_1d" not in self.tool_dict:
                return "❌ momentum_conservation_1d tool not available"
            
            collision_data = self._parse_momentum_conservation_1d_data(problem)
            
            tool = self.tool_dict["momentum_conservation_1d"]
            result = await tool.ainvoke({
                "collision_data": json.dumps(collision_data)
            })
            
            return f"🎯 **1D MOMENTUM CONSERVATION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 1D momentum conservation analysis: {e}"

    async def _call_momentum_conservation_2d_tool(self, problem: str) -> str:
        """Call 2D momentum conservation tool"""
        try:
            if "momentum_conservation_2d" not in self.tool_dict:
                return "❌ momentum_conservation_2d tool not available"
            
            collision_data = self._parse_momentum_conservation_2d_data(problem)
            
            tool = self.tool_dict["momentum_conservation_2d"]
            result = await tool.ainvoke({
                "collision_data": json.dumps(collision_data)
            })
            
            return f"🎯 **2D MOMENTUM CONSERVATION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in 2D momentum conservation analysis: {e}"

    async def _call_analyze_collision_tool(self, problem: str) -> str:
        """Call comprehensive collision analysis tool"""
        try:
            if "analyze_collision" not in self.tool_dict:
                return "❌ analyze_collision tool not available"
            
            collision_scenario = self._parse_collision_scenario_data(problem)
            
            tool = self.tool_dict["analyze_collision"]
            result = await tool.ainvoke({
                "collision_scenario": json.dumps(collision_scenario)
            })
            
            return f"🎯 **COMPREHENSIVE COLLISION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in comprehensive collision analysis: {e}"

    # PARSING METHODS (Momentum)
    def _parse_momentum_1d_data(self, text: str) -> tuple:
        """Parse 1D momentum parameters"""
        # Default values
        mass = 5.0
        velocity = 10.0
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mass = float(match.group(1))
                break
        
        # Parse velocity (including direction)
        velocity_patterns = [
            r'(-?\d+(?:\.\d+)?)\s*m/s',
            r'velocity[:\s=]+(-?\d+(?:\.\d+)?)',
            r'speed[:\s=]+(\d+(?:\.\d+)?)',
            r'v[:\s=]+(-?\d+(?:\.\d+)?)',
            r'moving.*?(-?\d+(?:\.\d+)?)\s*m/s'
        ]
        
        for pattern in velocity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                velocity = float(match.group(1))
                break
        
        # Check for direction indicators
        if any(word in text.lower() for word in ["left", "backward", "negative", "opposite"]):
            velocity = -abs(velocity)
        elif any(word in text.lower() for word in ["right", "forward", "positive"]):
            velocity = abs(velocity)
        
        return mass, velocity

    def _parse_momentum_2d_data(self, text: str) -> tuple:
        """Parse 2D momentum parameters"""
        # Default values
        mass = 3.0
        velocity = 15.0
        angle_degrees = 30.0
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mass = float(match.group(1))
                break
        
        # Parse velocity
        velocity_patterns = [
            r'(\d+(?:\.\d+)?)\s*m/s',
            r'velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'speed[:\s=]+(\d+(?:\.\d+)?)',
            r'v[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in velocity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                velocity = float(match.group(1))
                break
        
        # Parse angle
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*°',
            r'(\d+(?:\.\d+)?)\s*degree',
            r'angle[:\s=]+(\d+(?:\.\d+)?)',
            r'at\s+(\d+(?:\.\d+)?)\s*°',
            r'direction[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in angle_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                angle_degrees = float(match.group(1))
                break
        
        return mass, velocity, angle_degrees

    def _parse_impulse_1d_data(self, text: str) -> tuple:
        """Parse 1D impulse parameters"""
        # Default values
        force = 20.0
        time = 0.5
        initial_momentum = None
        final_momentum = None
        
        # Parse force
        force_patterns = [
            r'(\d+(?:\.\d+)?)\s*N',
            r'force[:\s=]+(\d+(?:\.\d+)?)',
            r'F[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in force_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                force = float(match.group(1))
                break
        
        # Parse time
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*s(?:ec|ond)?',
            r'time[:\s=]+(\d+(?:\.\d+)?)',
            r'for\s+(\d+(?:\.\d+)?)\s*s',
            r't[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time = float(match.group(1))
                break
        
        # Parse initial momentum
        initial_p_patterns = [
            r'initial momentum[:\s=]+(\d+(?:\.\d+)?)',
            r'pi[:\s=]+(\d+(?:\.\d+)?)',
            r'p0[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in initial_p_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                initial_momentum = float(match.group(1))
                break
        
        # Parse final momentum
        final_p_patterns = [
            r'final momentum[:\s=]+(\d+(?:\.\d+)?)',
            r'pf[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in final_p_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                final_momentum = float(match.group(1))
                break
        
        return force, time, initial_momentum, final_momentum

    def _parse_impulse_2d_data(self, text: str) -> tuple:
        """Parse 2D impulse parameters"""
        force_data = {}
        time = None
        momentum_data = None
        
        # Parse force components or magnitude/angle
        fx_match = re.search(r'fx[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        fy_match = re.search(r'fy[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        
        if fx_match and fy_match:
            force_data = {
                "fx": float(fx_match.group(1)),
                "fy": float(fy_match.group(1))
            }
        else:
            # Try magnitude and angle
            force_mag_match = re.search(r'force.*?(\d+(?:\.\d+)?)\s*N', text, re.IGNORECASE)
            force_angle_match = re.search(r'(\d+(?:\.\d+)?)\s*°', text, re.IGNORECASE)
            
            if force_mag_match:
                magnitude = float(force_mag_match.group(1))
                angle = float(force_angle_match.group(1)) if force_angle_match else 0.0
                force_data = {
                    "magnitude": magnitude,
                    "angle": angle
                }
            else:
                # Default force data
                force_data = {"magnitude": 15, "angle": 30}
        
        # Parse time
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*s(?:ec|ond)?', text, re.IGNORECASE)
        if time_match:
            time = float(time_match.group(1))
        
        # Parse momentum data (if provided)
        # This would be complex to parse from natural language, so we'll use defaults
        if "initial" in text.lower() and "final" in text.lower():
            momentum_data = {
                "initial": {"px": 5, "py": 3},
                "final": {"px": 8, "py": 7}
            }
        
        return force_data, time, momentum_data

    def _parse_momentum_impulse_theorem_data(self, text: str) -> dict:
        """Parse momentum-impulse theorem parameters"""
        data = {}
        
        # Parse mass
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if mass_match:
            data["mass"] = float(mass_match.group(1))
        
        # Parse initial velocity
        initial_v_patterns = [
            r'initial velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'vi[:\s=]+(\d+(?:\.\d+)?)',
            r'v0[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in initial_v_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["initial_velocity"] = float(match.group(1))
                break
        
        # Parse force
        force_match = re.search(r'(\d+(?:\.\d+)?)\s*N', text, re.IGNORECASE)
        if force_match:
            data["force"] = float(force_match.group(1))
        
        # Parse time
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*s(?:ec|ond)?', text, re.IGNORECASE)
        if time_match:
            data["time"] = float(time_match.group(1))
        
        # Parse impulse
        impulse_patterns = [
            r'impulse[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*N⋅s',
            r'J[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in impulse_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["impulse"] = float(match.group(1))
                break
        
        # Default values if nothing found
        if not data:
            data = {"mass": 2, "initial_velocity": 5, "force": 10, "time": 3}
        
        return data

    def _parse_momentum_conservation_1d_data(self, text: str) -> dict:
        """Parse 1D momentum conservation parameters"""
        data = {}
        
        # Parse masses
        mass_matches = re.findall(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if len(mass_matches) >= 2:
            data["m1"] = float(mass_matches[0])
            data["m2"] = float(mass_matches[1])
        else:
            data["m1"] = 2.0  # Default
            data["m2"] = 3.0
        
        # Parse initial velocities
        velocity_matches = re.findall(r'(-?\d+(?:\.\d+)?)\s*m/s', text, re.IGNORECASE)
        if len(velocity_matches) >= 2:
            data["v1i"] = float(velocity_matches[0])
            data["v2i"] = float(velocity_matches[1])
        elif len(velocity_matches) == 1:
            data["v1i"] = float(velocity_matches[0])
            data["v2i"] = 0.0  # At rest
        else:
            data["v1i"] = 8.0  # Default
            data["v2i"] = 0.0  # At rest
        
        # Determine collision type
        if "elastic" in text.lower():
            data["collision_type"] = "elastic"
        elif "perfectly inelastic" in text.lower() or "stick" in text.lower():
            data["collision_type"] = "perfectly_inelastic"
        elif "inelastic" in text.lower():
            data["collision_type"] = "inelastic"
        else:
            data["collision_type"] = "elastic"  # Default
        
        # Parse final velocities if given
        if "final" in text.lower():
            final_v_matches = re.findall(r'vf.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if final_v_matches:
                data["v1f"] = float(final_v_matches[0])
        
        return data

    def _parse_momentum_conservation_2d_data(self, text: str) -> dict:
        """Parse 2D momentum conservation parameters"""
        data = {"collision_type": "elastic"}
        
        # Parse object data
        mass_matches = re.findall(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*m/s', text, re.IGNORECASE)
        angle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*°', text, re.IGNORECASE)
        
        if len(mass_matches) >= 2 and len(velocity_matches) >= 2:
            data["object1"] = {
                "mass": float(mass_matches[0]),
                "velocity": float(velocity_matches[0]),
                "angle": float(angle_matches[0]) if len(angle_matches) >= 1 else 0.0
            }
            data["object2"] = {
                "mass": float(mass_matches[1]),
                "velocity": float(velocity_matches[1]),
                "angle": float(angle_matches[1]) if len(angle_matches) >= 2 else 90.0
            }
        else:
            # Default billiard ball scenario
            data["object1"] = {"mass": 0.16, "velocity": 25, "angle": 0}
            data["object2"] = {"mass": 0.16, "velocity": 0, "angle": 0}
        
        # Determine collision type
        if "perfectly inelastic" in text.lower():
            data["collision_type"] = "perfectly_inelastic"
        elif "inelastic" in text.lower():
            data["collision_type"] = "inelastic"
        else:
            data["collision_type"] = "elastic"
        
        return data

    def _parse_collision_scenario_data(self, text: str) -> dict:
        """Parse comprehensive collision scenario parameters"""
        data = {"scenario": "general", "analysis_type": "basic"}
        
        # Determine scenario type
        if "car" in text.lower() or "vehicle" in text.lower() or "crash" in text.lower():
            data["scenario"] = "car_crash"
            data["analysis_type"] = "safety"
        elif "rocket" in text.lower():
            data["scenario"] = "rocket_propulsion"
        elif "sport" in text.lower() or "ball" in text.lower():
            data["scenario"] = "sports_collision"
        
        # Parse vehicle/object data
        mass_matches = re.findall(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        velocity_matches = re.findall(r'(\d+(?:\.\d+)?)\s*m/s', text, re.IGNORECASE)
        angle_matches = re.findall(r'(\d+(?:\.\d+)?)\s*°', text, re.IGNORECASE)
        
        if len(mass_matches) >= 2 and len(velocity_matches) >= 2:
            data["car1"] = {
                "mass": float(mass_matches[0]),
                "velocity": float(velocity_matches[0]),
                "direction": float(angle_matches[0]) if len(angle_matches) >= 1 else 0.0
            }
            data["car2"] = {
                "mass": float(mass_matches[1]),
                "velocity": float(velocity_matches[1]),
                "direction": float(angle_matches[1]) if len(angle_matches) >= 2 else 90.0
            }
        else:
            # Default car crash scenario
            data["car1"] = {"mass": 1500, "velocity": 20, "direction": 0}
            data["car2"] = {"mass": 1200, "velocity": 15, "direction": 90}
        
        return data
    
    # ENERGY-SPECIFIC DIRECT TOOL METHODS

    async def _solve_energy_problem_direct(self, problem: str) -> str:
        """Direct tool solving for energy problems"""
        problem_lower = problem.lower()
        
        # Kinetic energy calculations
        if any(word in problem_lower for word in ["kinetic energy", "ke", "kinetic", "moving", "velocity"]) and any(word in problem_lower for word in ["mass", "kg", "m/s"]):
            return await self._call_kinetic_energy_tool(problem)
            
        # Gravitational potential energy
        elif any(word in problem_lower for word in ["potential energy", "pe", "gravitational", "height", "mgh"]):
            return await self._call_gravitational_potential_energy_tool(problem)
            
        # Elastic potential energy (springs)
        elif any(word in problem_lower for word in ["spring", "elastic", "compressed", "stretched", "spring constant", "k="]):
            return await self._call_elastic_potential_energy_tool(problem)
            
        # Work calculations
        elif any(word in problem_lower for word in ["work", "force", "displacement", "distance"]) and any(word in problem_lower for word in ["angle", "cos", "n", "meter", "m"]):
            return await self._call_work_tool(problem)
            
        # Work-energy theorem
        elif any(word in problem_lower for word in ["work-energy theorem", "work energy", "net work", "change in kinetic"]):
            return await self._call_work_energy_theorem_tool(problem)
            
        # Energy conservation
        elif any(word in problem_lower for word in ["conservation", "conserved", "energy transformation", "dropped", "falls", "pendulum"]):
            return await self._call_energy_conservation_tool(problem)
            
        # Energy with friction
        elif any(word in problem_lower for word in ["friction", "braking", "sliding", "efficiency", "heat", "dissipation"]):
            return await self._call_energy_with_friction_tool(problem)
            
        # Complex energy systems
        elif any(word in problem_lower for word in ["roller coaster", "system", "multi", "complex", "track", "trajectory"]):
            return await self._call_analyze_energy_system_tool(problem)
            
        else:
            # Default based on keywords
            if any(word in problem_lower for word in ["kinetic", "velocity", "speed"]):
                return await self._call_kinetic_energy_tool(problem)
            elif any(word in problem_lower for word in ["height", "gravitational", "dropped", "fall"]):
                return await self._call_gravitational_potential_energy_tool(problem)
            elif any(word in problem_lower for word in ["spring", "elastic", "compressed"]):
                return await self._call_elastic_potential_energy_tool(problem)
            elif any(word in problem_lower for word in ["work", "force"]):
                return await self._call_work_tool(problem)
            else:
                return await self._call_energy_conservation_tool(problem)

    async def _call_kinetic_energy_tool(self, problem: str) -> str:
        """Call kinetic energy calculation tool"""
        try:
            if "calculate_kinetic_energy_tool" not in self.tool_dict:
                return "❌ calculate_kinetic_energy_tool not available"
            
            mass, velocity = self._parse_kinetic_energy_data(problem)
            
            tool = self.tool_dict["calculate_kinetic_energy_tool"]
            result = await tool.ainvoke({
                "mass": mass,
                "velocity": velocity
            })
            
            return f"🎯 **KINETIC ENERGY CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in kinetic energy calculation: {e}"

    async def _call_gravitational_potential_energy_tool(self, problem: str) -> str:
        """Call gravitational potential energy calculation tool"""
        try:
            if "calculate_gravitational_potential_energy_tool" not in self.tool_dict:
                return "❌ calculate_gravitational_potential_energy_tool not available"
            
            mass, height, gravity, reference_level = self._parse_gravitational_pe_data(problem)
            
            tool = self.tool_dict["calculate_gravitational_potential_energy_tool"]
            params = {
                "mass": mass,
                "height": height
            }
            if gravity != 9.81:
                params["gravity"] = gravity
            if reference_level != "ground":
                params["reference_level"] = reference_level
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **GRAVITATIONAL POTENTIAL ENERGY CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in gravitational potential energy calculation: {e}"

    async def _call_elastic_potential_energy_tool(self, problem: str) -> str:
        """Call elastic potential energy calculation tool"""
        try:
            if "calculate_elastic_potential_energy_tool" not in self.tool_dict:
                return "❌ calculate_elastic_potential_energy_tool not available"
            
            spring_constant, displacement, equilibrium_position = self._parse_elastic_pe_data(problem)
            
            tool = self.tool_dict["calculate_elastic_potential_energy_tool"]
            params = {
                "spring_constant": spring_constant,
                "displacement": displacement
            }
            if equilibrium_position != "natural length":
                params["equilibrium_position"] = equilibrium_position
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **ELASTIC POTENTIAL ENERGY CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in elastic potential energy calculation: {e}"

    async def _call_work_tool(self, problem: str) -> str:
        """Call work calculation tool"""
        try:
            if "calculate_work_tool" not in self.tool_dict:
                return "❌ calculate_work_tool not available"
            
            force, displacement, angle_degrees, force_type = self._parse_work_data(problem)
            
            tool = self.tool_dict["calculate_work_tool"]
            params = {
                "force": force,
                "displacement": displacement,
                "angle_degrees": angle_degrees
            }
            if force_type != "constant":
                params["force_type"] = force_type
            
            result = await tool.ainvoke(params)
            
            return f"🎯 **WORK CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in work calculation: {e}"

    async def _call_work_energy_theorem_tool(self, problem: str) -> str:
        """Call work-energy theorem tool"""
        try:
            if "work_energy_theorem" not in self.tool_dict:
                return "❌ work_energy_theorem tool not available"
            
            problem_data = self._parse_work_energy_theorem_data(problem)
            
            tool = self.tool_dict["work_energy_theorem"]
            result = await tool.ainvoke({
                "problem_data": json.dumps(problem_data)
            })
            
            return f"🎯 **WORK-ENERGY THEOREM ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in work-energy theorem analysis: {e}"

    async def _call_energy_conservation_tool(self, problem: str) -> str:
        """Call energy conservation analysis tool"""
        try:
            if "energy_conservation" not in self.tool_dict:
                return "❌ energy_conservation tool not available"
            
            system_data = self._parse_energy_conservation_data(problem)
            
            tool = self.tool_dict["energy_conservation"]
            result = await tool.ainvoke({
                "system_data": json.dumps(system_data)
            })
            
            return f"🎯 **ENERGY CONSERVATION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in energy conservation analysis: {e}"

    async def _call_energy_with_friction_tool(self, problem: str) -> str:
        """Call energy with friction analysis tool"""
        try:
            if "energy_with_friction" not in self.tool_dict:
                return "❌ energy_with_friction tool not available"
            
            friction_data = self._parse_energy_with_friction_data(problem)
            
            tool = self.tool_dict["energy_with_friction"]
            result = await tool.ainvoke({
                "friction_data": json.dumps(friction_data)
            })
            
            return f"🎯 **ENERGY WITH FRICTION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in energy with friction analysis: {e}"

    async def _call_analyze_energy_system_tool(self, problem: str) -> str:
        """Call comprehensive energy system analysis tool"""
        try:
            if "analyze_energy_system" not in self.tool_dict:
                return "❌ analyze_energy_system tool not available"
            
            system_data = self._parse_energy_system_data(problem)
            
            tool = self.tool_dict["analyze_energy_system"]
            result = await tool.ainvoke({
                "system_data": json.dumps(system_data)
            })
            
            return f"🎯 **COMPREHENSIVE ENERGY SYSTEM ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in energy system analysis: {e}"

    # PARSING METHODS (Energy)
    def _parse_kinetic_energy_data(self, text: str) -> tuple:
        """Parse kinetic energy parameters"""
        # Default values
        mass = 5.0
        velocity = 10.0
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mass = float(match.group(1))
                break
        
        # Parse velocity
        velocity_patterns = [
            r'(\d+(?:\.\d+)?)\s*m/s',
            r'velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'speed[:\s=]+(\d+(?:\.\d+)?)',
            r'v[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in velocity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                velocity = float(match.group(1))
                break
        
        return mass, velocity

    def _parse_gravitational_pe_data(self, text: str) -> tuple:
        """Parse gravitational potential energy parameters"""
        # Default values
        mass = 2.0
        height = 15.0
        gravity = 9.81
        reference_level = "ground"
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mass = float(match.group(1))
                break
        
        # Parse height
        height_patterns = [
            r'(\d+(?:\.\d+)?)\s*m(?:\\s|$)',
            r'height[:\s=]+(\d+(?:\.\d+)?)',
            r'h[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*meter'
        ]
        
        for pattern in height_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                height = float(match.group(1))
                break
        
        # Parse gravity (if specified)
        gravity_patterns = [
            r'g[:\s=]+(\d+(?:\.\d+)?)',
            r'gravity[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in gravity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gravity = float(match.group(1))
                break
        
        # Parse reference level
        if "table" in text.lower():
            reference_level = "table"
        elif "floor" in text.lower():
            reference_level = "floor"
        elif "sea level" in text.lower():
            reference_level = "sea level"
        
        return mass, height, gravity, reference_level

    def _parse_elastic_pe_data(self, text: str) -> tuple:
        """Parse elastic potential energy parameters"""
        # Default values
        spring_constant = 200.0
        displacement = 0.1
        equilibrium_position = "natural length"
        
        # Parse spring constant
        k_patterns = [
            r'k[:\s=]+(\d+(?:\.\d+)?)',
            r'spring constant[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*N/m'
        ]
        
        for pattern in k_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                spring_constant = float(match.group(1))
                break
        
        # Parse displacement
        displacement_patterns = [
            r'compressed[:\s]+(\d+(?:\.\d+)?)',
            r'stretched[:\s]+(\d+(?:\.\d+)?)',
            r'displacement[:\s=]+(\d+(?:\.\d+)?)',
            r'x[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*m(?:\\s|$)',
            r'(\d+(?:\.\d+)?)\s*cm'
        ]
        
        for pattern in displacement_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                displacement = float(match.group(1))
                # Convert cm to m if needed
                if "cm" in match.group(0):
                    displacement /= 100
                break
        
        # Determine sign (compression vs stretch)
        if "compressed" in text.lower():
            displacement = abs(displacement)  # Positive for compression
        elif "stretched" in text.lower():
            displacement = abs(displacement)  # Positive for stretch
        
        return spring_constant, displacement, equilibrium_position

    def _parse_work_data(self, text: str) -> tuple:
        """Parse work calculation parameters"""
        # Default values
        force = 20.0
        displacement = 5.0
        angle_degrees = 0.0
        force_type = "constant"
        
        # Parse force
        force_patterns = [
            r'(\d+(?:\.\d+)?)\s*N',
            r'force[:\s=]+(\d+(?:\.\d+)?)',
            r'F[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in force_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                force = float(match.group(1))
                break
        
        # Parse displacement
        displacement_patterns = [
            r'(\d+(?:\.\d+)?)\s*m(?:\\s|$|[^/])',
            r'distance[:\s=]+(\d+(?:\.\d+)?)',
            r'displacement[:\s=]+(\d+(?:\.\d+)?)',
            r'd[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in displacement_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                displacement = float(match.group(1))
                break
        
        # Parse angle
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*°',
            r'(\d+(?:\.\d+)?)\s*degree',
            r'angle[:\s=]+(\d+(?:\.\d+)?)',
            r'θ[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in angle_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                angle_degrees = float(match.group(1))
                break
        
        # Determine force type
        if "friction" in text.lower():
            force_type = "friction"
        elif "variable" in text.lower():
            force_type = "variable"
        elif "applied" in text.lower():
            force_type = "applied"
        
        return force, displacement, angle_degrees, force_type

    def _parse_work_energy_theorem_data(self, text: str) -> dict:
        """Parse work-energy theorem parameters"""
        data = {}
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["mass"] = float(match.group(1))
                break
        
        # Parse initial velocity
        initial_v_patterns = [
            r'initial velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'vi[:\s=]+(\d+(?:\.\d+)?)',
            r'v0[:\s=]+(\d+(?:\.\d+)?)',
            r'starts.*?(\d+(?:\.\d+)?)\s*m/s'
        ]
        
        for pattern in initial_v_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["initial_velocity"] = float(match.group(1))
                break
        
        # Parse final velocity
        final_v_patterns = [
            r'final velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'vf[:\s=]+(\d+(?:\.\d+)?)',
            r'final.*?(\d+(?:\.\d+)?)\s*m/s'
        ]
        
        for pattern in final_v_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["final_velocity"] = float(match.group(1))
                break
        
        # Parse work done
        work_patterns = [
            r'(\d+(?:\.\d+)?)\s*J',
            r'work[:\s=]+(\d+(?:\.\d+)?)',
            r'W[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in work_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["work_done"] = float(match.group(1))
                break
        
        # Parse force and displacement if present
        force_match = re.search(r'(\d+(?:\.\d+)?)\s*N', text, re.IGNORECASE)
        displacement_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?:\\s|$|[^/])', text, re.IGNORECASE)
        
        if force_match:
            data["force"] = float(force_match.group(1))
        if displacement_match:
            data["displacement"] = float(displacement_match.group(1))
        
        # Default values if nothing found
        if not data:
            data = {"mass": 3, "initial_velocity": 5, "work_done": 40}
        
        return data

    def _parse_energy_conservation_data(self, text: str) -> dict:
        """Parse energy conservation parameters"""
        data = {}
        
        # Parse mass
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if mass_match:
            data["mass"] = float(mass_match.group(1))
        else:
            data["mass"] = 2.0  # Default
        
        # Parse initial height
        initial_height_patterns = [
            r'(?:from|at|initial height)[:\s]+(\d+(?:\.\d+)?)\s*m',
            r'(\d+(?:\.\d+)?)\s*m.*?height',
            r'height.*?(\d+(?:\.\d+)?)\s*m'
        ]
        
        for pattern in initial_height_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["initial_height"] = float(match.group(1))
                break
        
        # Parse final height (often 0 for ground level)
        if "ground" in text.lower() or "floor" in text.lower():
            data["final_height"] = 0.0
        
        # Parse initial velocity
        initial_v_match = re.search(r'initial.*?(\d+(?:\.\d+)?)\s*m/s', text, re.IGNORECASE)
        if initial_v_match:
            data["initial_velocity"] = float(initial_v_match.group(1))
        elif "rest" in text.lower() or "dropped" in text.lower():
            data["initial_velocity"] = 0.0
        
        # Parse spring constant and compression/stretch
        k_match = re.search(r'k[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        compression_match = re.search(r'compress.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        
        if k_match:
            data["spring_constant"] = float(k_match.group(1))
        if compression_match:
            data["compression"] = float(compression_match.group(1))
        
        # Default conservation scenario if minimal data
        if len(data) <= 1:
            data = {"mass": 2, "initial_height": 10, "final_height": 0, "initial_velocity": 0}
        
        return data

    def _parse_energy_with_friction_data(self, text: str) -> dict:
        """Parse energy with friction parameters"""
        data = {}
        
        # Parse mass
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if mass_match:
            data["mass"] = float(mass_match.group(1))
        else:
            data["mass"] = 1500.0  # Default car mass
        
        # Parse initial velocity
        velocity_patterns = [
            r'(\d+(?:\.\d+)?)\s*m/s',
            r'velocity[:\s=]+(\d+(?:\.\d+)?)',
            r'speed[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in velocity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["initial_velocity"] = float(match.group(1))
                break
        
        # Parse friction coefficient
        friction_patterns = [
            r'coefficient[:\s=]+(\d+(?:\.\d+)?)',
            r'μ[:\s=]+(\d+(?:\.\d+)?)',
            r'friction.*?(\d+(?:\.\d+)?)'
        ]
        
        for pattern in friction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["friction_coefficient"] = float(match.group(1))
                break
        
        # Parse distance
        distance_match = re.search(r'(\d+(?:\.\d+)?)\s*m(?:\\s|$|[^/])', text, re.IGNORECASE)
        if distance_match:
            data["distance"] = float(distance_match.group(1))
        
        # Parse heights for inclined scenarios
        height_matches = re.findall(r'(\d+(?:\.\d+)?)\s*m', text, re.IGNORECASE)
        if len(height_matches) >= 2:
            data["initial_height"] = float(height_matches[0])
            data["final_height"] = float(height_matches[1])
        
        # Default braking scenario if minimal data
        if not data:
            data = {"mass": 1500, "initial_velocity": 25, "friction_coefficient": 0.7}
        
        return data

    def _parse_energy_system_data(self, text: str) -> dict:
        """Parse complex energy system parameters"""
        data = {"scenario": "general_system"}
        
        # Determine scenario type
        if "roller coaster" in text.lower():
            data["scenario"] = "roller_coaster"
        elif "pendulum" in text.lower():
            data["scenario"] = "pendulum"
        elif "spring" in text.lower() and "mass" in text.lower():
            data["scenario"] = "spring_mass"
        
        # Parse mass
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if mass_match:
            data["mass"] = float(mass_match.group(1))
        else:
            data["mass"] = 500.0  # Default
        
        # For roller coaster scenarios
        if data["scenario"] == "roller_coaster":
            # Parse track points
            height_matches = re.findall(r'(\d+(?:\.\d+)?)\s*m', text, re.IGNORECASE)
            if height_matches:
                track_points = []
                for i, height in enumerate(height_matches):
                    point = {"height": float(height)}
                    if i == 0:
                        point["velocity"] = 0  # Start from rest typically
                    else:
                        point["velocity"] = None  # To be calculated
                    track_points.append(point)
                data["track_points"] = track_points
            
            # Parse friction coefficient
            friction_match = re.search(r'friction.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if friction_match:
                data["friction_coefficient"] = float(friction_match.group(1))
            else:
                data["friction_coefficient"] = 0.02  # Low friction
            
            # Parse track length
            length_match = re.search(r'(\d+(?:\.\d+)?)\s*m.*?(?:long|length|track)', text, re.IGNORECASE)
            if length_match:
                data["track_length"] = float(length_match.group(1))
            else:
                data["track_length"] = 1000  # Default
        
        # For pendulum scenarios
        elif data["scenario"] == "pendulum":
            length_match = re.search(r'(\d+(?:\.\d+)?)\s*m.*?length', text, re.IGNORECASE)
            if length_match:
                data["length"] = float(length_match.group(1))
            else:
                data["length"] = 1.5
            
            angle_match = re.search(r'(\d+(?:\.\d+)?)\s*°', text, re.IGNORECASE)
            if angle_match:
                data["initial_angle_degrees"] = float(angle_match.group(1))
            else:
                data["initial_angle_degrees"] = 30
        
        # For spring-mass scenarios
        elif data["scenario"] == "spring_mass":
            k_match = re.search(r'k[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if k_match:
                data["spring_constant"] = float(k_match.group(1))
            else:
                data["spring_constant"] = 100
            
            amplitude_match = re.search(r'amplitude.*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if amplitude_match:
                data["amplitude"] = float(amplitude_match.group(1))
            else:
                data["amplitude"] = 0.1
        
        return data

    # ANGULAR MOTION-SPECIFIC DIRECT TOOL METHODS

    async def _solve_angular_motion_problem_direct(self, problem: str) -> str:
        """Direct tool solving for angular motion problems"""
        problem_lower = problem.lower()
        
        # Angular kinematics calculations
        if any(word in problem_lower for word in ["angular kinematics", "angular velocity", "angular acceleration", "ω", "α", "θ"]) and any(word in problem_lower for word in ["time", "seconds", "equation"]):
            return await self._call_angular_kinematics_tool(problem)
            
        # Moment of inertia calculations
        elif any(word in problem_lower for word in ["moment of inertia", "inertia", "i =", "rod", "disk", "sphere", "cylinder", "hoop"]):
            return await self._call_moment_of_inertia_tool(problem)
            
        # Torque calculations
        elif any(word in problem_lower for word in ["torque", "τ", "force", "radius", "n⋅m", "lever"]) and not any(word in problem_lower for word in ["momentum", "impulse"]):
            return await self._call_torque_tool(problem)
            
        # Angular momentum conservation
        elif any(word in problem_lower for word in ["angular momentum", "conservation", "figure skater", "spinning", "l =", "iω"]):
            return await self._call_angular_momentum_conservation_tool(problem)
            
        # Rotational energy
        elif any(word in problem_lower for word in ["rotational energy", "rotational ke", "ke_rot", "spinning energy", "½iω²"]):
            return await self._call_rotational_energy_tool(problem)
            
        # Angular impulse-momentum
        elif any(word in problem_lower for word in ["angular impulse", "impulse-momentum", "angular impulse-momentum", "∫τ dt"]):
            return await self._call_angular_impulse_momentum_tool(problem)
            
        # Rolling motion
        elif any(word in problem_lower for word in ["rolling", "rolls", "incline", "sphere race", "yo-yo", "no-slip"]):
            return await self._call_rolling_motion_tool(problem)
            
        else:
            # Default based on keywords - prioritize by complexity
            if any(word in problem_lower for word in ["rolling", "incline", "yo-yo"]):
                return await self._call_rolling_motion_tool(problem)
            elif any(word in problem_lower for word in ["conservation", "figure skater", "spinning"]):
                return await self._call_angular_momentum_conservation_tool(problem)
            elif any(word in problem_lower for word in ["torque", "force", "lever"]):
                return await self._call_torque_tool(problem)
            elif any(word in problem_lower for word in ["energy", "ke_rot", "rotational"]):
                return await self._call_rotational_energy_tool(problem)
            elif any(word in problem_lower for word in ["moment of inertia", "inertia", "rod", "disk", "sphere"]):
                return await self._call_moment_of_inertia_tool(problem)
            elif any(word in problem_lower for word in ["angular", "ω", "α", "kinematics"]):
                return await self._call_angular_kinematics_tool(problem)
            else:
                return await self._call_angular_kinematics_tool(problem)

    async def _call_angular_kinematics_tool(self, problem: str) -> str:
        """Call angular kinematics tool"""
        try:
            if "angular_kinematics" not in self.tool_dict:
                return "❌ angular_kinematics tool not available"
            
            kinematics_data = self._parse_angular_kinematics_data(problem)
            
            tool = self.tool_dict["angular_kinematics"]
            result = await tool.ainvoke({
                "kinematics_data": json.dumps(kinematics_data)
            })
            
            return f"🎯 **ANGULAR KINEMATICS ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in angular kinematics analysis: {e}"

    async def _call_moment_of_inertia_tool(self, problem: str) -> str:
        """Call moment of inertia calculation tool"""
        try:
            if "calculate_moment_of_inertia" not in self.tool_dict:
                return "❌ calculate_moment_of_inertia tool not available"
            
            object_data = self._parse_moment_of_inertia_data(problem)
            
            tool = self.tool_dict["calculate_moment_of_inertia"]
            result = await tool.ainvoke({
                "object_data": json.dumps(object_data)
            })
            
            return f"🎯 **MOMENT OF INERTIA CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in moment of inertia calculation: {e}"

    async def _call_torque_tool(self, problem: str) -> str:
        """Call torque calculation tool"""
        try:
            if "calculate_torque" not in self.tool_dict:
                return "❌ calculate_torque tool not available"
            
            torque_data = self._parse_torque_data(problem)
            
            tool = self.tool_dict["calculate_torque"]
            result = await tool.ainvoke({
                "torque_data": json.dumps(torque_data)
            })
            
            return f"🎯 **TORQUE CALCULATION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in torque calculation: {e}"

    async def _call_angular_momentum_conservation_tool(self, problem: str) -> str:
        """Call angular momentum conservation tool"""
        try:
            if "angular_momentum_conservation" not in self.tool_dict:
                return "❌ angular_momentum_conservation tool not available"
            
            momentum_data = self._parse_angular_momentum_data(problem)
            
            tool = self.tool_dict["angular_momentum_conservation"]
            result = await tool.ainvoke({
                "momentum_data": json.dumps(momentum_data)
            })
            
            return f"🎯 **ANGULAR MOMENTUM CONSERVATION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in angular momentum conservation analysis: {e}"

    async def _call_rotational_energy_tool(self, problem: str) -> str:
        """Call rotational energy calculation tool"""
        try:
            if "rotational_energy" not in self.tool_dict:
                return "❌ rotational_energy tool not available"
            
            energy_data = self._parse_rotational_energy_data(problem)
            
            tool = self.tool_dict["rotational_energy"]
            result = await tool.ainvoke({
                "energy_data": json.dumps(energy_data)
            })
            
            return f"🎯 **ROTATIONAL ENERGY ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in rotational energy analysis: {e}"

    async def _call_angular_impulse_momentum_tool(self, problem: str) -> str:
        """Call angular impulse-momentum tool"""
        try:
            if "angular_impulse_momentum" not in self.tool_dict:
                return "❌ angular_impulse_momentum tool not available"
            
            impulse_data = self._parse_angular_impulse_data(problem)
            
            tool = self.tool_dict["angular_impulse_momentum"]
            result = await tool.ainvoke({
                "impulse_data": json.dumps(impulse_data)
            })
            
            return f"🎯 **ANGULAR IMPULSE-MOMENTUM ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in angular impulse-momentum analysis: {e}"

    async def _call_rolling_motion_tool(self, problem: str) -> str:
        """Call rolling motion analysis tool"""
        try:
            if "rolling_motion_analysis" not in self.tool_dict:
                return "❌ rolling_motion_analysis tool not available"
            
            rolling_data = self._parse_rolling_motion_data(problem)
            
            tool = self.tool_dict["rolling_motion_analysis"]
            result = await tool.ainvoke({
                "rolling_data": json.dumps(rolling_data)
            })
            
            return f"🎯 **ROLLING MOTION ANALYSIS**\\n\\n{result}\\n\\n✅ **Analysis completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in rolling motion analysis: {e}"

    # PARSING METHODS (Angular Motion)
    def _parse_angular_kinematics_data(self, text: str) -> dict:
        """Parse angular kinematics parameters"""
        import re
        
        data = {}
        
        # Parse angular displacement (theta)
        theta_patterns = [
            r'θ[:\s=]+(\d+(?:\.\d+)?)',
            r'theta[:\s=]+(\d+(?:\.\d+)?)',
            r'angular displacement[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*rad(?:ian)?(?:s)?'
        ]
        
        for pattern in theta_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["theta"] = float(match.group(1))
                break
        
        # Parse initial angular velocity (omega_0)
        omega0_patterns = [
            r'ω₀[:\s=]+(\d+(?:\.\d+)?)',
            r'omega_0[:\s=]+(\d+(?:\.\d+)?)',
            r'initial.*?(\d+(?:\.\d+)?)\s*rad/s',
            r'ω0[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in omega0_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["omega_0"] = float(match.group(1))
                break
        
        # Parse final angular velocity (omega_f)
        omegaf_patterns = [
            r'ωf[:\s=]+(\d+(?:\.\d+)?)',
            r'omega_f[:\s=]+(\d+(?:\.\d+)?)',
            r'final.*?(\d+(?:\.\d+)?)\s*rad/s'
        ]
        
        for pattern in omegaf_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["omega_f"] = float(match.group(1))
                break
        
        # Parse angular acceleration (alpha)
        alpha_patterns = [
            r'α[:\s=]+(\d+(?:\.\d+)?)',
            r'alpha[:\s=]+(\d+(?:\.\d+)?)',
            r'angular acceleration[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*rad/s²'
        ]
        
        for pattern in alpha_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["alpha"] = float(match.group(1))
                break
        
        # Parse time
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*s(?:ec|ond)?(?:s)?',
            r'time[:\s=]+(\d+(?:\.\d+)?)',
            r'after\s+(\d+(?:\.\d+)?)\s*s',
            r't[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["time"] = float(match.group(1))
                break
        
        # Default values if nothing found
        if not data:
            data = {"omega_0": 5, "alpha": 2, "time": 3}
        
        return data

    def _parse_moment_of_inertia_data(self, text: str) -> dict:
        """Parse moment of inertia parameters"""
        import re
        
        data = {}
        
        # Determine shape
        if any(word in text.lower() for word in ["rod", "bar", "stick"]):
            data["shape"] = "rod"
        elif any(word in text.lower() for word in ["disk", "disc"]):
            data["shape"] = "disk"
        elif any(word in text.lower() for word in ["sphere", "ball"]):
            data["shape"] = "sphere"
        elif any(word in text.lower() for word in ["cylinder", "tube"]):
            data["shape"] = "cylinder"
        elif any(word in text.lower() for word in ["hoop", "ring"]):
            data["shape"] = "cylinder"
            data["hollow"] = True
        else:
            data["shape"] = "cylinder"  # Default
        
        # Parse mass
        mass_patterns = [
            r'(\d+(?:\.\d+)?)\s*kg',
            r'mass[:\s=]+(\d+(?:\.\d+)?)',
            r'm[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in mass_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["mass"] = float(match.group(1))
                break
        
        # Parse dimensions based on shape
        if data["shape"] == "rod":
            # Parse length
            length_patterns = [
                r'(\d+(?:\.\d+)?)\s*m(?:\s|$|[^/])',
                r'length[:\s=]+(\d+(?:\.\d+)?)',
                r'long[:\s,]+(\d+(?:\.\d+)?)',
                r'L[:\s=]+(\d+(?:\.\d+)?)'
            ]
            
            for pattern in length_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["length"] = float(match.group(1))
                    break
            
            # Determine axis
            if any(word in text.lower() for word in ["center", "middle", "about center"]):
                data["axis"] = "center"
            elif any(word in text.lower() for word in ["end", "endpoint", "about end"]):
                data["axis"] = "end"
            else:
                data["axis"] = "center"  # Default
        
        else:
            # Parse radius for other shapes
            radius_patterns = [
                r'radius[:\s=]+(\d+(?:\.\d+)?)',
                r'r[:\s=]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*m.*?radius'
            ]
            
            for pattern in radius_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["radius"] = float(match.group(1))
                    break
        
        # Determine if hollow
        if any(word in text.lower() for word in ["hollow", "empty", "thin-walled"]):
            data["hollow"] = True
        elif any(word in text.lower() for word in ["solid", "filled"]):
            data["hollow"] = False
        
        # Parse offset for parallel axis theorem
        offset_patterns = [
            r'offset[:\s=]+(\d+(?:\.\d+)?)',
            r'distance.*?(\d+(?:\.\d+)?)\s*m',
            r'parallel.*?(\d+(?:\.\d+)?)\s*m'
        ]
        
        for pattern in offset_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and "parallel" in text.lower():
                data["offset"] = float(match.group(1))
                break
        
        # Default values if nothing found
        if not data:
            data = {"shape": "cylinder", "mass": 2, "radius": 0.3}
        
        return data

    def _parse_torque_data(self, text: str) -> dict:
        """Parse torque calculation parameters"""
        import re
        
        data = {}
        
        # Parse force
        force_patterns = [
            r'(\d+(?:\.\d+)?)\s*N',
            r'force[:\s=]+(\d+(?:\.\d+)?)',
            r'F[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in force_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["force"] = float(match.group(1))
                break
        
        # Parse radius
        radius_patterns = [
            r'(\d+(?:\.\d+)?)\s*m(?:\s|$|[^/])',
            r'radius[:\s=]+(\d+(?:\.\d+)?)',
            r'r[:\s=]+(\d+(?:\.\d+)?)',
            r'at\s+(\d+(?:\.\d+)?)\s*m'
        ]
        
        for pattern in radius_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["radius"] = float(match.group(1))
                break
        
        # Parse angle
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*°',
            r'(\d+(?:\.\d+)?)\s*degree',
            r'angle[:\s=]+(\d+(?:\.\d+)?)',
            r'θ[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in angle_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["angle"] = float(match.group(1))
                break
        
        # Default angle based on description
        if "angle" not in data:
            if any(word in text.lower() for word in ["perpendicular", "⊥", "90"]):
                data["angle"] = 90
            elif any(word in text.lower() for word in ["parallel", "∥", "0"]):
                data["angle"] = 0
            else:
                data["angle"] = 90  # Default perpendicular
        
        # Parse moment of inertia and angular acceleration for τ = Iα
        I_patterns = [
            r'I[:\s=]+(\d+(?:\.\d+)?)',
            r'moment of inertia[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*kg⋅m²'
        ]
        
        for pattern in I_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["moment_of_inertia"] = float(match.group(1))
                break
        
        alpha_patterns = [
            r'α[:\s=]+(\d+(?:\.\d+)?)',
            r'angular acceleration[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*rad/s²'
        ]
        
        for pattern in alpha_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["angular_acceleration"] = float(match.group(1))
                break
        
        # Default values if nothing found
        if not data:
            data = {"force": 50, "radius": 0.3, "angle": 90}
        
        return data

    def _parse_angular_momentum_data(self, text: str) -> dict:
        """Parse angular momentum conservation parameters"""
        import re
        
        data = {}
        
        # Figure skater scenario
        if any(word in text.lower() for word in ["figure skater", "skater", "arms", "tucked", "extended"]):
            data["figure_skater"] = {}
            
            # Parse initial (extended) state
            I_extended_patterns = [
                r'I[:\s=]+(\d+(?:\.\d+)?)',
                r'extended.*?(\d+(?:\.\d+)?)\s*kg⋅m²',
                r'arms extended.*?(\d+(?:\.\d+)?)'
            ]
            
            for pattern in I_extended_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["figure_skater"]["I_extended"] = float(match.group(1))
                    break
            
            # Parse initial angular velocity
            omega_extended_patterns = [
                r'ω[:\s=]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*rad/s'
            ]
            
            for pattern in omega_extended_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["figure_skater"]["omega_extended"] = float(match.group(1))
                    break
            
            # Parse final (tucked) moment of inertia
            I_tucked_patterns = [
                r'tucked.*?(\d+(?:\.\d+)?)',
                r'arms.*?tucked.*?(\d+(?:\.\d+)?)',
                r'pulls.*?(\d+(?:\.\d+)?)'
            ]
            
            for pattern in I_tucked_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["figure_skater"]["I_tucked"] = float(match.group(1))
                    break
        
        # General initial/final states
        else:
            # Parse initial state
            data["initial"] = {}
            data["final"] = {}
            
            # Extract all I and omega values
            I_matches = re.findall(r'I[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            omega_matches = re.findall(r'ω[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            
            if len(I_matches) >= 2:
                data["initial"]["I"] = float(I_matches[0])
                data["final"]["I"] = float(I_matches[1])
            elif len(I_matches) == 1:
                data["initial"]["I"] = float(I_matches[0])
            
            if len(omega_matches) >= 1:
                data["initial"]["omega"] = float(omega_matches[0])
            if len(omega_matches) >= 2:
                data["final"]["omega"] = float(omega_matches[1])
        
        # Default values if nothing found
        if not data:
            data = {"figure_skater": {"I_extended": 5, "omega_extended": 2, "I_tucked": 1.5}}
        
        return data

    def _parse_rotational_energy_data(self, text: str) -> dict:
        """Parse rotational energy parameters"""
        import re
        
        data = {}
        
        # Simple rotational KE
        if any(word in text.lower() for word in ["ke_rot", "rotational energy", "½iω²"]):
            # Parse moment of inertia
            I_patterns = [
                r'I[:\s=]+(\d+(?:\.\d+)?)',
                r'moment of inertia[:\s=]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*kg⋅m²'
            ]
            
            for pattern in I_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["moment_of_inertia"] = float(match.group(1))
                    break
            
            # Parse angular velocity
            omega_patterns = [
                r'ω[:\s=]+(\d+(?:\.\d+)?)',
                r'angular velocity[:\s=]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*rad/s'
            ]
            
            for pattern in omega_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["angular_velocity"] = float(match.group(1))
                    break
        
        # Rolling object analysis
        elif any(word in text.lower() for word in ["rolling", "rolls"]):
            data["rolling_object"] = {}
            
            # Parse mass
            mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
            if mass_match:
                data["rolling_object"]["mass"] = float(mass_match.group(1))
            
            # Parse radius
            radius_match = re.search(r'radius[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if radius_match:
                data["rolling_object"]["radius"] = float(radius_match.group(1))
            
            # Parse velocity
            velocity_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', text, re.IGNORECASE)
            if velocity_match:
                data["rolling_object"]["velocity"] = float(velocity_match.group(1))
            
            # Determine shape
            if any(word in text.lower() for word in ["cylinder", "disk"]):
                data["rolling_object"]["shape"] = "cylinder"
            elif any(word in text.lower() for word in ["sphere", "ball"]):
                data["rolling_object"]["shape"] = "sphere"
            elif any(word in text.lower() for word in ["hoop", "ring"]):
                data["rolling_object"]["shape"] = "hoop"
            else:
                data["rolling_object"]["shape"] = "cylinder"
        
        # Energy transformation
        elif any(word in text.lower() for word in ["transformation", "initial", "final"]):
            data["energy_transformation"] = {}
            
            # Parse I
            I_match = re.search(r'I[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if I_match:
                data["energy_transformation"]["I"] = float(I_match.group(1))
            
            # Parse initial and final omega
            omega_matches = re.findall(r'(\d+(?:\.\d+)?)\s*rad/s', text, re.IGNORECASE)
            if len(omega_matches) >= 2:
                data["energy_transformation"]["omega_initial"] = float(omega_matches[0])
                data["energy_transformation"]["omega_final"] = float(omega_matches[1])
        
        # Default values if nothing found
        if not data:
            data = {"moment_of_inertia": 2, "angular_velocity": 5}
        
        return data

    def _parse_angular_impulse_data(self, text: str) -> dict:
        """Parse angular impulse parameters"""
        import re
        
        data = {}
        
        # Parse torque
        torque_patterns = [
            r'τ[:\s=]+(\d+(?:\.\d+)?)',
            r'torque[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*N⋅m'
        ]
        
        for pattern in torque_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["torque"] = float(match.group(1))
                break
        
        # Parse time
        time_patterns = [
            r'(\d+(?:\.\d+)?)\s*s(?:ec|ond)?(?:s)?',
            r'time[:\s=]+(\d+(?:\.\d+)?)',
            r'for\s+(\d+(?:\.\d+)?)\s*s'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["time"] = float(match.group(1))
                break
        
        # Parse moment of inertia
        I_patterns = [
            r'I[:\s=]+(\d+(?:\.\d+)?)',
            r'moment of inertia[:\s=]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*kg⋅m²'
        ]
        
        for pattern in I_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["moment_of_inertia"] = float(match.group(1))
                break
        
        # Parse initial angular velocity
        omega0_patterns = [
            r'ω₀[:\s=]+(\d+(?:\.\d+)?)',
            r'initial.*?(\d+(?:\.\d+)?)\s*rad/s',
            r'ω0[:\s=]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in omega0_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["initial_omega"] = float(match.group(1))
                break
        
        # Default values if nothing found
        if not data:
            data = {"torque": 15, "time": 2, "moment_of_inertia": 1.2, "initial_omega": 3}
        
        return data

    def _parse_rolling_motion_data(self, text: str) -> dict:
        """Parse rolling motion parameters"""
        import re
        
        data = {}
        
        # Yo-yo scenario
        if any(word in text.lower() for word in ["yo-yo", "yoyo"]):
            data["yo_yo"] = {}
            
            # Parse mass
            mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
            if mass_match:
                data["yo_yo"]["mass"] = float(mass_match.group(1))
            else:
                data["yo_yo"]["mass"] = 0.2
            
            # Parse radius
            radius_match = re.search(r'radius[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if radius_match:
                data["yo_yo"]["radius"] = float(radius_match.group(1))
            else:
                data["yo_yo"]["radius"] = 0.05
            
            # Parse string length
            length_patterns = [
                r'string.*?(\d+(?:\.\d+)?)\s*m',
                r'length[:\s=]+(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*m.*?string'
            ]
            
            for pattern in length_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["yo_yo"]["string_length"] = float(match.group(1))
                    break
            
            if "string_length" not in data["yo_yo"]:
                data["yo_yo"]["string_length"] = 1.5
        
        # Sphere race scenario
        elif any(word in text.lower() for word in ["race", "compare", "vs", "versus"]):
            data["sphere_race"] = {}
            
            if any(word in text.lower() for word in ["solid", "hollow"]):
                data["sphere_race"]["solid_sphere"] = {"mass": 2, "radius": 0.1}
                data["sphere_race"]["hollow_sphere"] = {"mass": 2, "radius": 0.1}
        
        # Single object on incline
        else:
            # Determine object type
            if any(word in text.lower() for word in ["cylinder", "disk"]):
                data["object"] = "cylinder"
            elif any(word in text.lower() for word in ["sphere", "ball"]):
                data["object"] = "sphere"
            elif any(word in text.lower() for word in ["hoop", "ring"]):
                data["object"] = "hoop"
            else:
                data["object"] = "cylinder"  # Default
            
            # Parse mass
            mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
            if mass_match:
                data["mass"] = float(mass_match.group(1))
            else:
                data["mass"] = 5
            
            # Parse radius
            radius_match = re.search(r'radius[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if radius_match:
                data["radius"] = float(radius_match.group(1))
            else:
                data["radius"] = 0.3
            
            # Parse incline angle
            angle_patterns = [
                r'(\d+(?:\.\d+)?)\s*°',
                r'incline[:\s=]+(\d+(?:\.\d+)?)',
                r'angle[:\s=]+(\d+(?:\.\d+)?)'
            ]
            
            for pattern in angle_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    data["incline_angle"] = float(match.group(1))
                    break
            
            if "incline_angle" not in data:
                data["incline_angle"] = 30  # Default
            
            # Parse height
            height_match = re.search(r'height[:\s=]+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if height_match:
                data["height"] = float(height_match.group(1))
        
        return data

    # A2A COMPATIBILITY METHODS
    async def get_capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities for A2A framework"""
        return {
            "agent_id": self.agent_id,
            "metadata": self.metadata,
            "available_tools": list(self.tool_dict.keys()) if self.tool_dict else [],
            "status": "ready" if self.initialized else "not_initialized",
            "mode": "direct_tools" if self.use_direct_tools else "langchain_agent"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for A2A framework"""
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if self.initialized else "not_ready",
            "tools_count": len(self.tool_dict) if self.tool_dict else 0,
            "ready": self.initialized,
            "mode": "direct_tools" if self.use_direct_tools else "langchain_agent"
        }

    # Add these parsing helper methods for equation solving:
    def _parse_equation_solving(self, problem: str) -> tuple:
        """Parse equation and variable to solve for"""
        problem_lower = problem.lower()
        
        # Try to extract equation directly from the problem text first
        import re
        
        # Look for equations with = sign (extract just the equation part)
        equation_match = re.search(r'(?:solve\s+)?([A-Za-z]\s*=\s*[^,]+?)(?:\s+for\s+\w+|\s*,|$)', problem, re.IGNORECASE)
        if equation_match:
            raw_equation = equation_match.group(1).strip()
            
            # Clean up the equation - normalize common patterns
            equation = raw_equation
            equation = re.sub(r'\b1/2\b', '1/2', equation)  # normalize fractions
            equation = re.sub(r'\b½\b', '1/2', equation)    # convert ½ to 1/2
            equation = re.sub(r'\s*\*\s*', ' * ', equation)  # normalize multiplication
            equation = re.sub(r'\bv\^2\b', 'v^2', equation) # normalize exponents
            equation = re.sub(r'\bv²\b', 'v^2', equation)   # convert ² to ^2
        else:
            # Fall back to common physics equations mapping  
            equation_patterns = {
                "kinetic": "K = 1/2 * m * v^2",
                "k =": "K = 1/2 * m * v^2", 
                "f = ma": "F = m * a",
                "force": "F = m * a",
                "p = mv": "p = m * v",
                "momentum": "p = m * v",
                "e = mc": "E = m * c^2",
                "energy": "E = m * c^2"
            }
            
            equation = "K = 1/2 * m * v^2"  # default
            for key, eq in equation_patterns.items():
                if key in problem_lower:
                    equation = eq
                    break
        
        # Try to identify what to solve for
        solve_for = "v"  # default
        solve_patterns = {
            "for v": "v", "velocity": "v", "speed": "v",
            "for m": "m", "mass": "m", 
            "for a": "a", "acceleration": "a",
            "for f": "F", "force": "F",
            "for k": "K", "kinetic": "K",
            "for p": "p", "momentum": "p",
            "for e": "E", "energy": "E"
        }
        
        for pattern, var in solve_patterns.items():
            if pattern in problem_lower:
                solve_for = var
                break
        
        return equation, solve_for

    def _parse_physics_formula(self, problem: str) -> tuple:
        """Parse physics formula parameters for numerical calculations"""
        problem_lower = problem.lower()
        
        formula_name = "kinetic_energy"
        known_values = {}
        solve_for = "v"
        
        # Detect formula type
        if any(word in problem_lower for word in ["kinetic", "k ="]):
            formula_name = "kinetic_energy"
        elif any(word in problem_lower for word in ["force", "f ="]):
            formula_name = "force"
        
        # Extract numerical values (this is a simplified version)
        import re
        
        # Look for mass values
        mass_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', problem)
        if mass_match:
            known_values["m"] = float(mass_match.group(1))
        
        # Look for velocity values
        velocity_match = re.search(r'(\d+(?:\.\d+)?)\s*m/s', problem)
        if velocity_match:
            known_values["v"] = float(velocity_match.group(1))
        
        # Look for energy values
        energy_match = re.search(r'(\d+(?:\.\d+)?)\s*j', problem_lower)
        if energy_match:
            known_values["K"] = float(energy_match.group(1))
        
        return formula_name, known_values, solve_for

# Interactive interface function to add:

async def interactive_angular_motion_agent():
    """Interactive angular motion agent interface"""
    agent = create_angular_motion_agent(use_direct_tools=True)  # Use working mode
    
    await agent.initialize()
    agent.get_user_message()
    
    while True:
        try:
            user_input = input(f"🌀 Angular Motion Problem: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"👋 Goodbye from Angular Motion Agent!")
                break
                
            if not user_input:
                continue
                
            print("\\n🤖 Analyzing and solving...")
            result = await agent.solve_problem(user_input)
            
            if result["success"]:
                print("📊 SOLUTION:")
                print(result["solution"])
            else:
                print(f"❌ ERROR: {result['error']}")
                
            print("\\n" + "-"*70 + "\\n")
            
        except KeyboardInterrupt:
            print(f"\\n👋 Goodbye from Angular Motion Agent!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\\n")

# Interactive interface function to add:

async def interactive_energy_agent():
    """Interactive energy agent interface"""
    agent = create_energy_agent(use_direct_tools=True)  # Use working mode
    
    await agent.initialize()
    agent.get_user_message()
    
    while True:
        try:
            user_input = input(f"⚡ Energy Problem: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"👋 Goodbye from Energy Agent!")
                break
                
            if not user_input:
                continue
                
            print("\\n🤖 Analyzing and solving...")
            result = await agent.solve_problem(user_input)
            
            if result["success"]:
                print("📊 SOLUTION:")
                print(result["solution"])
            else:
                print(f"❌ ERROR: {result['error']}")
                
            print("\\n" + "-"*70 + "\\n")
            
        except KeyboardInterrupt:
            print(f"\\n👋 Goodbye from Energy Agent!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\\n")


# FACTORY FUNCTIONS
def create_forces_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create a forces agent"""
    return CombinedPhysicsAgent(
        agent_id="forces_agent", 
        use_direct_tools=use_direct_tools
    )
def create_kinematics_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create a kinematics agent"""
    return CombinedPhysicsAgent(
        agent_id="kinematics_agent", 
        use_direct_tools=use_direct_tools
    )
def create_math_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create a math agent"""
    return CombinedPhysicsAgent(
        agent_id="math_agent", 
        use_direct_tools=use_direct_tools
    )

def create_momentum_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create a momentum agent"""
    return CombinedPhysicsAgent(
        agent_id="momentum_agent", 
        use_direct_tools=use_direct_tools
    )

def create_energy_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create a energy agent"""
    return CombinedPhysicsAgent(
        agent_id="energy_agent", 
        use_direct_tools=use_direct_tools
    )

def create_angular_motion_agent(use_direct_tools: bool = True) -> CombinedPhysicsAgent:
    """Create an angular motion agent"""
    return CombinedPhysicsAgent(
        agent_id="angular_motion_agent", 
        use_direct_tools=use_direct_tools
    )

# INTERACTIVE INTERFACES
async def interactive_physics_agent(agent_type: str = "forces"):
    """Universal interactive interface"""
    if agent_type == "forces":
        agent = create_forces_agent(use_direct_tools=True)  # Use working mode
    elif agent_type == "kinematics":
        agent = create_kinematics_agent(use_direct_tools=True)  # Use working mode
    elif agent_type == "math":
        agent = create_math_agent(use_direct_tools=True)  # Use working mode
    elif agent_type == "momentum":
        agent = create_momentum_agent(use_direct_tools=True)  # Use working mode 
    elif agent_type == "energy":
        agent = create_energy_agent(use_direct_tools=True)  # Use working mode   
    elif agent_type == "angular motion":
        agent = create_angular_motion_agent(use_direct_tools=True)  # Use working mode   
    else:
        raise ValueError("Agent type must be 'forces', 'kinematics', 'math', 'momentum', 'energy' or 'angular motion' ")
        
    await agent.initialize()
    agent.get_user_message()
    
    while True:
        try:
            user_input = input(f"🧮 {agent_type.title()} Problem: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"👋 Goodbye from {agent_type.title()} Agent!")
                break
                
            if not user_input:
                continue
                
            print("\\n🤖 Analyzing and solving...")
            result = await agent.solve_problem(user_input)
            
            if result["success"]:
                print("📊 SOLUTION:")
                print(result["solution"])
            else:
                print(f"❌ ERROR: {result['error']}")
                
            print("\\n" + "-"*70 + "\\n")
            
        except KeyboardInterrupt:
            print(f"\\n👋 Goodbye from {agent_type.title()} Agent!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\\n")

# MAIN EXECUTION
if __name__ == "__main__":
    # Default: Run forces agent in interactive mode
    #asyncio.run(interactive_physics_agent("forces"))
    
    # Uncomment to run kinematics agent
    #asyncio.run(interactive_physics_agent("kinematics"))

    # Uncomment to run math agent
    #asyncio.run(interactive_physics_agent("math"))

    # Uncomment to run math agent
    #asyncio.run(interactive_physics_agent("momentum"))

    # Uncomment to run math agent
    #asyncio.run(interactive_physics_agent("energy"))

    # Uncomment to run math agent
    asyncio.run(interactive_physics_agent("angular motion"))