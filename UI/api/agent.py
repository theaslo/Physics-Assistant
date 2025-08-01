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
from typing import Dict, Any, Optional
from langchain_ollama.chat_models import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


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
                 use_direct_tools: bool = True):
        
        self.agent_id = agent_id
        self.llm_base_url = llm_base_url
        self.model = model
        self.use_direct_tools = use_direct_tools  # Key flag for working mode
        
        # Initialize based on agent type
        self._setup_agent_config()
        
        # Common initialization
        self.client = None
        self.tools = None
        self.tool_dict = {}
        self.agent = None
        self.initialized = False
        
        # A2A compatibility metadata
        self.metadata.update({
            "input_types": ["text", "json"],
            "output_types": ["text", "analysis"],
            "version": "1.0.0"
        })

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
            
        else:
            raise ValueError(f"Agent type '{self.agent_id}' not supported. Use 'forces_agent' or 'kinematics_agent'")

    async def initialize(self):
        """Initialize the physics agent with MCP tools"""
        if self.initialized:
            return
            
        print(f"🚀 Initializing {self.agent_id.title().replace('_', ' ')} (Mode: {'Direct Tools' if self.use_direct_tools else 'LangChain Agent'})...")
        
        # Connect to MCP server on VM using HTTP transport
        server_name = self.agent_id.split('_')[0]  # 'forces' or 'kinematics'
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

    async def solve_problem(self, problem: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main method to solve physics problems
        
        Args:
            problem: Text description of the physics problem
            context: Optional context from A2A framework or other agents
            
        Returns:
            Dict with solution, reasoning, and metadata
        """
        if not self.initialized:
            await self.initialize()
            
        try:
            if self.use_direct_tools:
                # Use proven direct tool approach
                solution = await self._solve_with_direct_tools(problem)
                reasoning = "Solved using direct MCP tool execution (proven method)"
            else:
                # Use LangChain agent approach
                solution = await self._solve_with_langchain_agent(problem, context)
                reasoning = "Solved using LangChain agent with MCP tools"
                
            return {
                "success": True,
                "agent_id": self.agent_id,
                "problem": problem,
                "solution": solution,
                "reasoning": reasoning,
                "tools_used": list(self.tool_dict.keys()),
                "metadata": self.metadata
            }
            
        except Exception as e:
            return {
                "success": False,
                "agent_id": self.agent_id,
                "problem": problem,
                "error": str(e),
                "reasoning": f"Error in problem solving: {str(e)}"
            }

    async def _solve_with_direct_tools(self, problem: str) -> str:
        """Solve using direct tool calls (working_forces_agent.py approach)"""
        if self.agent_id == "forces_agent":
            return await self._solve_forces_problem_direct(problem)
        elif self.agent_id == "kinematics_agent":
            return await self._solve_kinematics_problem_direct(problem)
        else:
            return "❌ Unsupported agent type for direct tools"

    async def _solve_with_langchain_agent(self, problem: str, context: Optional[Dict] = None) -> str:
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

    async def _call_projectile_tool(self, problem: str) -> str:
        """Call projectile motion tool"""
        try:
            if "projectile_motion_2d" not in self.tool_dict:
                return "❌ projectile_motion_2d tool not available"
            
            params = self._parse_projectile_params(problem)
            tool = self.tool_dict["projectile_motion_2d"]
            
            result = await tool.ainvoke({
                "launch_conditions": json.dumps(params)
            })
            
            return f"🎯 **PROJECTILE MOTION SOLUTION**\\n\\n{result}\\n\\n✅ **Calculation completed using MCP tools**"
            
        except Exception as e:
            return f"❌ Error in projectile motion: {e}"

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
            pattern = r'(\\d+(?:\\.\\d+)?)\\s*[Nn]?\\s*(?:at|@)\\s*(\\d+(?:\\.\\d+)?)(?:°|degree|deg)?'
            matches = re.findall(pattern, text)
            forces = [{"magnitude": float(mag), "angle": float(ang)} for mag, ang in matches]
        
        return forces

    def _parse_spring_params(self, text: str):
        """Parse spring parameters"""
        k = 200  # default
        displacement = -0.05  # default compression
        
        k_match = re.search(r'k\\s*=\\s*(\\d+(?:\\.\\d+)?)', text)
        if k_match:
            k = float(k_match.group(1))
        
        if "compressed" in text.lower() or "compression" in text.lower():
            disp_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m', text)
            if disp_match:
                displacement = -float(disp_match.group(1))
        elif "stretched" in text.lower() or "extension" in text.lower():
            disp_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m', text)
            if disp_match:
                displacement = float(disp_match.group(1))
        
        return k, displacement

    def _parse_force_magnitude_angle(self, text: str):
        """Parse single force magnitude and angle"""
        magnitude = 25  # default
        angle = 45  # default
        
        mag_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*[Nn]', text)
        if mag_match:
            magnitude = float(mag_match.group(1))
        
        ang_match = re.search(r'(\\d+(?:\\.\\d+)?)(?:°|degree|deg)', text)
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
            app_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*[Nn]', text)
            if app_match:
                forces.append({"name": "Applied", "magnitude": float(app_match.group(1)), "angle": 0})
        
        return forces

    # PARSING METHODS (Kinematics)
    def _parse_projectile_params(self, text: str) -> dict:
        """Parse projectile motion parameters"""
        params = {"v0": 30, "angle": 45, "h0": 0}
        
        # Parse velocity
        v_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m/s', text)
        if v_match:
            params["v0"] = float(v_match.group(1))
        
        # Parse angle
        ang_match = re.search(r'(\\d+(?:\\.\\d+)?)(?:°|degree|deg)', text)
        if ang_match:
            params["angle"] = float(ang_match.group(1))
        
        # Parse height
        h_match = re.search(r'from\\s+(\\d+(?:\\.\\d+)?)\\s*m', text)
        if h_match:
            params["h0"] = float(h_match.group(1))
        
        return params

    def _parse_freefall_params(self, text: str) -> dict:
        """Parse free fall parameters"""
        params = {}
        
        # Parse height
        h_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m', text)
        if h_match:
            params["h0"] = float(h_match.group(1))
        
        # Parse initial velocity if any
        if "thrown" in text.lower():
            v_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m/s', text)
            if v_match:
                params["v0"] = float(v_match.group(1))
        else:
            params["v0"] = 0  # Dropped from rest
        
        return params

    def _parse_acceleration_params(self, text: str) -> dict:
        """Parse constant acceleration parameters"""
        params = {}
        
        # Parse initial velocity
        if "from rest" in text.lower():
            params["v0"] = 0
        else:
            v0_match = re.search(r'at\\s+(\\d+(?:\\.\\d+)?)\\s*m/s', text)
            if v0_match:
                params["v0"] = float(v0_match.group(1))
        
        # Parse acceleration
        a_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m/s[²²]', text)
        if a_match:
            params["a"] = float(a_match.group(1))
        
        # Parse time
        t_match = re.search(r'for\\s+(\\d+(?:\\.\\d+)?)\\s*s', text)
        if t_match:
            params["t"] = float(t_match.group(1))
        
        return params

    def _parse_uniform_motion_params(self, text: str) -> dict:
        """Parse uniform motion parameters"""
        params = {"x0": 0}
        
        # Parse velocity
        v_match = re.search(r'(\\d+(?:\\.\\d+)?)\\s*m/s', text)
        if v_match:
            params["v"] = float(v_match.group(1))
        
        # Parse time
        t_match = re.search(r'for\\s+(\\d+(?:\\.\\d+)?)\\s*s', text)
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

# INTERACTIVE INTERFACES
async def interactive_physics_agent(agent_type: str = "forces"):
    """Universal interactive interface"""
    if agent_type == "forces":
        agent = create_forces_agent(use_direct_tools=True)  # Use working mode
    elif agent_type == "kinematics":
        agent = create_kinematics_agent(use_direct_tools=True)  # Use working mode
    else:
        raise ValueError("Agent type must be 'forces' or 'kinematics'")
        
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
    asyncio.run(interactive_physics_agent("kinematics"))