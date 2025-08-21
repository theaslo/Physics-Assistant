# Physics Assistant

A comprehensive physics education platform designed to help college-level introductory physics students learn through interactive AI tutoring. The system combines a Streamlit-based frontend with specialized MCP (Model Context Protocol) servers running Ollama models for physics-specific assistance.

## Architecture Overview

- **Frontend**: Streamlit web application with authentication, chat interface, and educational tools
- **Backend**: MCP servers providing specialized physics agents and computational tools
- **AI Models**: Ollama-powered physics tutors for different topics (kinematics, forces, energy, momentum, rotational motion)
- **Tools**: Physics calculation utilities, visualization tools, and mathematical helpers
- **Deployment**: Docker containers for production deployment

## Key Features

### Physics Agents
- **Kinematics Agent**: 1D and 2D motion problems
- **Forces Agent**: Newton's laws, vector analysis, equilibrium
- **Energy Agent**: Work-energy theorem, conservation laws
- **Momentum Agent**: Linear momentum, collisions, impulse
- **Angular Motion Agent**: Rotational dynamics and kinematics
- **Math Helper Agent**: Trigonometry, algebra, calculus support

### Educational Tools
- Real-time LaTeX equation rendering
- Interactive physics visualizations and simulations
- Unit conversion and scientific notation handling
- Progress tracking and learning analytics
- File upload for physics diagrams and problems

### MCP Tools Integration
- Force calculation and vector analysis
- Spring force calculations (Hooke's Law)
- Friction analysis (static and kinetic)
- Inclined plane problems
- Tension and pulley systems
- Free body diagram generation

## Project Structure

```
Physics-Assistant/
├── UI/                     # Streamlit frontend application
│   ├── frontend/          # Main UI components and pages
│   └── api/               # Backend API and agent coordination
├── agents/                # Physics tutoring agents
├── mcp_tools/            # Physics calculation MCP servers
├── Docker/               # Containerization configuration
└── docker-compose.yaml  # Multi-service orchestration
```

## Development Setup

The system uses Python with UV package management and supports both local development and containerized deployment through Docker Compose.

## Important Instructions

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.