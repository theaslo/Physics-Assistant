# Forces Agent - Google A2A Compatible

A comprehensive physics force calculation agent designed to be compatible with the [Google A2A (Agent-to-Agent) framework](https://github.com/a2aproject/a2a). This agent handles all types of algebra-based force calculations and can seamlessly integrate with A2A host agents.

## 🎯 Overview

The Forces Agent is a specialized physics calculation agent that:
- Handles ALL force calculation types (1D, 2D, applied forces, equilibrium)
- Integrates seamlessly with Google A2A framework
- Uses MCP (Model Context Protocol) tools for precise calculations
- Provides structured responses for agent-to-agent communication
- Supports both standalone and orchestrated operation

## 🤖 A2A Framework Compatibility

### Agent Metadata
```python
{
    "id": "forces_agent",
    "name": "Forces Agent", 
    "description": "Comprehensive physics force calculation specialist",
    "capabilities": [
        "1D_force_addition", "2D_force_addition", "force_components",
        "resultant_calculations", "free_body_diagrams", "equilibrium_analysis",
        "spring_forces", "friction_forces", "weight_calculations", 
        "tension_analysis", "inclined_planes", "vector_operations"
    ],
    "input_types": ["text", "json"],
    "output_types": ["text", "analysis"],
    "version": "1.0.0"
}
```

### A2A Interface Methods

#### `solve_force_problem(problem: str, context: Optional[Dict] = None) -> Dict[str, Any]`
Main problem-solving interface for A2A integration.

**Input:**
- `problem`: Text description of physics problem
- `context`: Optional context from other agents

**Output:**
```python
{
    "success": True,
    "agent_id": "forces_agent",
    "problem": "...",
    "solution": "...", 
    "reasoning": "...",
    "tools_used": [...],
    "metadata": {...}
}
```

#### `get_capabilities() -> Dict[str, Any]`
Returns agent capabilities for A2A discovery.

#### `health_check() -> Dict[str, Any]`
Provides health status for A2A monitoring.

## 🔧 Force Calculation Capabilities

### Vector Operations
- **1D Force Addition**: Add forces along a single axis
- **2D Force Addition**: Combine forces with magnitude and angle
- **Component Resolution**: Break forces into x/y components
- **Resultant Calculations**: Find magnitude/direction from components
- **Vector Operations**: Dot product, cross product, addition, subtraction

### Equilibrium Analysis  
- **Free Body Diagrams**: Generate complete force diagrams
- **Equilibrium Checking**: Determine if forces are balanced
- **Balancing Forces**: Calculate forces needed for equilibrium
- **Static Analysis**: Systems at rest or constant velocity

### Applied Forces
- **Spring Forces**: Hooke's Law calculations (F = -kx)
- **Friction Forces**: Static and kinetic friction (f = μN)
- **Weight Forces**: Gravitational calculations (W = mg)
- **Tension Forces**: Rope, cable, and pulley systems
- **Inclined Planes**: Complete force analysis on slopes
- **Normal Forces**: Contact and support forces

## 🛠 Technical Architecture

### MCP Integration
The agent uses Model Context Protocol tools for precise calculations:
```python
# Connected to forces MCP server
tools = [
    'add_forces_1d', 'add_forces_2d', 'resolve_force_components',
    'find_resultant_force', 'create_free_body_diagram', 'check_equilibrium',
    'calculate_spring_force_tool', 'calculate_friction_force_tool',
    'calculate_weight_force', 'analyze_forces_on_incline',
    'analyze_tension_forces', 'force_vector_operations'
]
```

### LangChain Integration
- Uses LangChain for agent orchestration
- React agent pattern for tool selection and execution
- Structured conversation flow with system prompts

## 🚀 Usage Examples

### Standalone Operation
```python
from forces_agent import ForcesAgent

# Initialize agent
agent = ForcesAgent(agent_id="physics_solver")
await agent.initialize()

# Solve problem
result = await agent.solve_force_problem(
    "Add forces: 10N at 30°, 15N at 120°"
)
print(result["solution"])
```

### A2A Integration
```python
# Host agent calling forces agent
context = {
    "requesting_agent": "physics_tutor",
    "student_level": "introductory" 
}

result = await forces_agent.solve_force_problem(
    problem="Calculate spring force with k=200 N/m, x=0.05m",
    context=context
)

# Structured response for other agents
if result["success"]:
    solution = result["solution"]
    reasoning = result["reasoning"]
```

### Example Problems
```python
problems = [
    "Add 1D forces: 15N, -8N, 20N, -5N",
    "Find resultant of 12N at 45° and 16N at 135°",
    "Create free body diagram for box on inclined plane", 
    "Calculate spring force: k=300 N/m, compressed 0.04m",
    "Analyze 6kg mass on 35° incline with μ=0.25",
    "Check equilibrium: 10N right, 10N left, 20N up, 20N down",
    "Break down 25N force at 135° into components"
]
```

## 📋 Input/Output Formats

### Input Formats
**Text Problems:**
- Natural language physics problems
- "Add forces: 10N at 30°, 15N at 120°"
- "Calculate spring force with k=200 N/m, x=0.05m"

**JSON Context (A2A):**
```json
{
    "requesting_agent": "physics_tutor",
    "student_level": "advanced",
    "problem_type": "vector_addition",
    "previous_results": {...}
}
```

### Output Formats
**Structured Response:**
```json
{
    "success": true,
    "agent_id": "forces_agent",
    "problem": "Add forces: 10N at 30°, 15N at 120°",
    "solution": "Detailed step-by-step solution...",
    "reasoning": "Used 2D force addition with MCP tools",
    "tools_used": ["add_forces_2d"],
    "metadata": {...}
}
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.11+
- uv package manager
- Ollama with llama3.2:3b model
- Forces MCP server (included)

### Installation
```bash
cd /Users/asli.tandogan_kunkel/Projects/jakub_summer_project/forces_agent

# Install dependencies
uv sync

# Run standalone
uv run forces_agent.py

# Or run main entry point
uv run main.py
```

### Configuration
```python
# Agent configuration
agent = ForcesAgent(
    agent_id="custom_forces_agent",
    llm_base_url="http://localhost:11434",  # Ollama server
    model="llama3.2:3b"  # Or your preferred model
)
```

## 🧪 Testing

### Interactive Testing
```bash
uv run forces_agent.py
# Provides interactive chat interface for testing
```

### A2A Integration Testing
```python
# Test A2A methods
capabilities = await agent.get_capabilities()
health = await agent.health_check()
result = await agent.solve_force_problem(problem, context)
```

### Example Test Problems
Run the agent with these test cases:
1. Vector addition: "Add 10N at 0°, 15N at 90°, 8N at 180°"
2. Free body diagram: "Create FBD for 5kg box with applied force"
3. Spring force: "k=150 N/m, stretched by 0.08m"
4. Inclined plane: "10kg block on 30° incline with friction"
5. Equilibrium: "Check if forces balance: 12N right, 12N left"

## 🤝 A2A Integration Guide

### For Host Agent Developers
1. **Agent Discovery**: Use `get_capabilities()` to discover force calculation abilities
2. **Health Monitoring**: Use `health_check()` for system monitoring  
3. **Problem Delegation**: Use `solve_force_problem()` for force calculations
4. **Context Passing**: Include relevant context for better solutions

### Integration Pattern
```python
class HostAgent:
    async def handle_physics_problem(self, problem):
        # Check if it's a force problem
        if self.is_force_problem(problem):
            # Delegate to forces agent
            context = self.build_context()
            result = await self.forces_agent.solve_force_problem(
                problem, context
            )
            return self.process_force_result(result)
```

## 📊 Performance & Scalability

- **Response Time**: ~2-5 seconds per problem (depends on complexity)
- **Concurrent Requests**: Supports async operation for multiple requests
- **Memory Usage**: ~200MB typical, ~500MB with complex problems
- **Tool Coverage**: 12 specialized MCP tools for comprehensive coverage

## 🔮 Future Enhancements

1. **Multi-language Support**: Support for problems in multiple languages
2. **Visual Diagrams**: Generate actual diagrams (not just text descriptions)
3. **Step-by-step Tutorials**: Educational mode with detailed explanations
4. **Batch Processing**: Handle multiple problems simultaneously
5. **Integration APIs**: REST/GraphQL endpoints for broader integration

## 📚 Related Documentation

- [Forces MCP Server Documentation](../forces_mcp/README.md)
- [Google A2A Framework](https://github.com/a2aproject/a2a)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [LangChain Agent Documentation](https://python.langchain.com/docs/modules/agents/)

## 🏷 Version History

- **v1.0.0**: Initial release with full force calculation capabilities
- A2A framework compatibility
- Comprehensive MCP tool integration
- Interactive and programmatic interfaces

---

🤖 **Ready for A2A Integration!** This forces agent is designed to seamlessly work with Google A2A framework while providing comprehensive physics force calculation capabilities.
