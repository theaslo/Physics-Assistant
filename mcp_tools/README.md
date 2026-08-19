# Physics MCP Tools 🔬⚡

**MCP (Model Context Protocol) tools for physics calculations and analysis**

A collection of specialized MCP servers that provide physics calculation capabilities to AI assistants and other applications. Perfect for educational use, homework help, research, and physics problem-solving.

---

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run the forces server (default)
uv run physics-mcp

# Run a specific server
uv run physics-mcp --run kinematics-server 

# Run a specific server and transport
uv run physics-mcp --run kinematics-server --transport streamable_http


# Run a specific server, transport and port
uv run physics-mcp --run kinematics-server --transport streamable_http --port 12000

# Get help
uv run physics-mcp --help
```

---

## 🧮 Available Servers

### 🎯 Forces Server (`forces-server`)
*Calculate and analyze forces in 1D and 2D systems*

**Capabilities:**
- **1D Force Addition**: Sum forces along a line with equilibrium analysis
- **2D Force Resolution**: Break forces into x/y components and find resultants
- **Vector Operations**: Add, subtract, dot product, cross product of force vectors
- **Free Body Diagrams**: Generate text-based force analysis and equilibrium checks
- **Specialized Forces**: Spring forces (Hooke's Law), friction, weight/gravity
- **Inclined Planes**: Complete force analysis on angled surfaces
- **Tension Systems**: Analyze rope/pulley systems and Atwood machines

**Example Tools:**
```python
# Add 2D forces
add_forces_2d([{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 120}])

# Check equilibrium 
check_equilibrium('[{"magnitude": 10, "angle": 0}, {"magnitude": 10, "angle": 180}]')

# Analyze inclined plane
analyze_forces_on_incline(mass=5.0, angle_degrees=30, coefficient_friction=0.3)
```

### 📏 Kinematics Server (`kinematics-server`)
*Motion analysis with constant acceleration*

**Capabilities:**
- **1D Kinematics**: Solve for displacement, velocity, acceleration, and time
- **Projectile Motion**: Analyze 2D motion with gravity
- **Motion Equations**: Apply kinematic equations with detailed explanations

### ⚡ Circuit Server (`circuit-server`)
*Electrical circuit analysis and calculations*

**Capabilities:**
- **Ohm's Law**: Solve voltage, current, resistance, and power relationships
- **Resistor Networks**: Calculate equivalent resistance for series, parallel, and nested networks
- **Voltage Dividers**: Determine voltage drops and series current
- **RC Circuits**: Analyze charging and discharging time constants

### 🌊 Waves Server (`waves-server`)
*Wave motion, sound, acoustics, and interference*

**Capabilities:**
- **Wave Equation**: Solve velocity, frequency, wavelength, period, angular frequency, and wave number
- **Doppler Effect**: Analyze frequency shifts for moving sources and observers
- **Sound Intensity**: Convert between intensity and decibel level
- **Standing Waves**: Calculate harmonics for strings, open pipes, and closed pipes
- **Wave Interference**: Classify constructive, destructive, and partial interference

### 🧲 Electromagnetism Server (`electromagnetism-server`)
*Electricity, magnetism, circuits, and induction*

**Capabilities:**
- **Coulomb's Law**: Calculate point-charge force magnitude and direction
- **Electric Fields**: Analyze single-charge, uniform, and multi-charge 2D fields
- **Electric Potential**: Calculate potential and potential energy
- **Capacitance**: Analyze parallel plates, charge, energy, series, and parallel combinations
- **Circuit Tools**: Apply Ohm's Law and equivalent resistance calculations
- **Magnetism**: Calculate magnetic forces and fields from currents
- **Induction**: Apply Faraday's Law for changing flux

### 🔎 Optics Server (`optics-server`)
*Geometric optics, diffraction, and interference*

**Capabilities:**
- **Snell's Law**: Calculate refraction angles and critical angles
- **Lens/Mirror Equation**: Solve image distance, focal length, object distance, and magnification
- **Diffraction**: Analyze single-slit, double-slit, and grating patterns
- **Thin Film Interference**: Evaluate coatings, soap films, and optical path differences
- **Optical Power**: Convert focal length and diopters, including combined thin lenses

---

## 📋 Installation & Setup

### Prerequisites
- Python 3.11+ 
- [uv](https://docs.astral.sh/uv/) package manager

### Install
```bash
cd /path/to/Physics-Assistant/mcp_tools
uv sync  # Install dependencies
```

---

## 🔧 Usage

### Command Line Interface

```bash
# Basic usage (defaults to forces-server, streamable_http, localhost:10100)
uv run physics-mcp

# Specify server and transport
uv run physics-mcp --run <SERVER> --transport <TRANSPORT>

# Custom host and port
uv run physics-mcp --run forces-server --host 0.0.0.0 --port 8080
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--run` | `forces-server` | Server to run: `forces-server`, `kinematics-server`, `math-server`, `momentum-server`, `energy-server`, `angular-motion-server`, `circuit-server`, `thermodynamics-server`, `waves-server`, `electromagnetism-server`, `optics-server`, `modern-physics-server` |
| `--transport` | `streamable_http` | Transport: `stdio`, `sse`, `streamable_http` |
| `--host` | `localhost` | Host address to bind server |
| `--port` | `10100` | Port number to bind server |

### Transport Types

- **`streamable_http`**: HTTP-based transport (recommended for web integration)
- **`stdio`**: Standard input/output (for direct process communication)  
- **`sse`**: Server-Sent Events (for real-time web applications)

---

## 🔗 MCP Integration

These servers implement the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), making them compatible with:

- **Claude Desktop** (via MCP configuration)
- **Custom AI applications** (via MCP client libraries)
- **Educational platforms** (via HTTP/WebSocket integration)
- **Physics simulation tools** (via programmatic access)

### Claude Desktop Configuration

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "physics-forces": {
      "command": "uv",
      "args": ["run", "physics-mcp", "--run", "forces-server"],
      "cwd": "/path/to/Physics-Assistant/mcp_tools"
    },
    "physics-kinematics": {
      "command": "uv", 
      "args": ["run", "physics-mcp", "--run", "kinematics-server"],
      "cwd": "/path/to/Physics-Assistant/mcp_tools"
    },
    "physics-waves": {
      "command": "uv",
      "args": ["run", "physics-mcp", "--run", "waves-server"],
      "cwd": "/path/to/Physics-Assistant/mcp_tools"
    },
    "physics-electromagnetism": {
      "command": "uv",
      "args": ["run", "physics-mcp", "--run", "electromagnetism-server"],
      "cwd": "/path/to/Physics-Assistant/mcp_tools"
    },
    "physics-optics": {
      "command": "uv",
      "args": ["run", "physics-mcp", "--run", "optics-server"],
      "cwd": "/path/to/Physics-Assistant/mcp_tools"
    }
  }
}
```

---

## 🎓 Educational Use Cases

### For Students
- **Homework Help**: Step-by-step force calculations with explanations
- **Concept Learning**: Visual force analysis and equilibrium checking
- **Problem Verification**: Double-check manual calculations

### For Educators  
- **Interactive Lessons**: Real-time physics calculations during class
- **Assignment Creation**: Generate problems with worked solutions
- **Concept Demonstration**: Show force resolution and vector addition

### For Developers
- **Physics Engines**: Integrate accurate physics calculations
- **Educational Apps**: Add physics problem-solving capabilities  
- **Research Tools**: Automated physics analysis workflows

---

## 🛠️ Development

### Project Structure
```
mcp_tools/
├── physics_mcp_tools/          # Main package
│   ├── __init__.py            # CLI entry point
│   ├── forces_mcp_server.py   # Forces calculation server
│   ├── kinematics_mcp_server.py # Motion analysis server
│   ├── circuit_mcp_server.py  # Electrical circuit server
│   ├── waves_mcp_server.py    # Waves and sound server
│   ├── electromagnetism_mcp_server.py # E&M calculation server
│   ├── optics_mcp_server.py   # Optics calculation server
│   └── forces_utils.py        # Shared physics utilities
├── pyproject.toml             # Package configuration
├── README.md                  # This file
└── uv.lock                   # Dependency lock file
```

### Adding New Tools

1. Add tool functions to the appropriate server file
2. Use the `@mcp.tool()` decorator
3. Provide clear docstrings with examples
4. Include input validation and error handling

Example:
```python
@mcp.tool()
async def my_physics_tool(param1: float, param2: str) -> str:
    """Brief description of what this tool does.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        
    Returns:
        str: Description of return value
    """
    # Implementation here
    return "Result with explanation"
```

### Running Tests
```bash
# Install dev dependencies
uv sync --group dev

# Run tests
python -m unittest discover -s tests -p 'test_*.py'

# Code formatting
uv run black physics_mcp_tools/
uv run isort physics_mcp_tools/
```

---

## 📚 Part of Physics-Assistant

This package is part of the larger **Physics-Assistant** project, which aims to provide comprehensive physics education and problem-solving tools.

- **Repository**: [Physics-Assistant](https://github.com/theaslo/Physics-Assistant)
- **Author**: Asli Tandogan Kunkel (doogers@uconn.edu)
- **License**: MIT

### Related Components
- `agents/` - AI agents that use these MCP tools
- `mcp_tools/` - This physics calculation toolkit ← **You are here**

---

## 🤝 Contributing

Contributions are welcome! This project is designed to help students and educators with physics education.

### Ideas for Contributions
- Additional physics domains (thermodynamics, electromagnetism, waves)
- Enhanced visualization capabilities  
- More detailed step-by-step explanations
- Educational examples and use cases
- Performance optimizations
- Additional transport protocols

### Getting Started
1. Fork the repository
2. Create a feature branch
3. Add your physics tools or improvements  
4. Test thoroughly with educational examples
5. Submit a pull request

---

## 📄 License

MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Happy Physics Problem Solving!** 🎉

*Making physics calculations accessible to AI assistants and educational tools everywhere.*
