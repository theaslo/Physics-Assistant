# Physics Assistant - FastAPI + Streamlit Integration

A comprehensive physics tutoring system with a FastAPI backend hosting specialized physics agents and a Streamlit frontend providing an interactive user interface.

## 🏗️ Architecture

- **Backend**: FastAPI server hosting `CombinedPhysicsAgent` with MCP tool integration
- **Frontend**: Streamlit web application with dynamic agent selection
- **Agents**: Specialized physics agents for forces and kinematics problems
- **Tools**: MCP (Model Context Protocol) integration for direct physics calculations

## 📋 Requirements

### System Requirements
- Python 3.12+
- uv package manager (recommended) or pip
- Git

### Dependencies
- **API**: FastAPI, uvicorn, pydantic, langchain, MCP adapters
- **UI**: Streamlit, requests, authentication libraries

## 🚀 Quick Start

### Option 1: Complete System Startup (Recommended)
```bash
# Clone and navigate to the project
cd /path/to/Physics-Assistant/UI

# Start both API and UI together
python start_system.py
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start API server
python start_api.py

# Terminal 2: Start UI (in new terminal)
python start_ui.py
```

### Option 3: Individual Services
```bash
# API only
cd api/
uv pip install -e .
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# UI only
cd frontend/
uv pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

## 🔗 Access Points

Once running, access the application at:

- **🖥️ Streamlit UI**: http://localhost:8501
- **🚀 FastAPI Server**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **❤️ Health Check**: http://localhost:8000/health

## 📖 Installation Guide

### Step 1: Install Dependencies

#### Using uv (Recommended)
```bash
# Install API dependencies
cd api/
uv pip install -e .

# Install UI dependencies  
cd ../frontend/
uv pip install -r requirements.txt
```

#### Using pip
```bash
# Install API dependencies
cd api/
pip install -e .

# Install UI dependencies
cd ../frontend/
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
# Check if dependencies are installed
python -c "import fastapi, uvicorn, streamlit; print('✅ All dependencies installed')"
```

## 🤖 Available Physics Agents

### Forces Agent (`forces_agent`)
**Specialization**: Force analysis and Newton's laws
- Free body diagram creation
- Force vector addition and resolution  
- Newton's laws applications
- Equilibrium analysis
- Spring force calculations

**Example Problems**:
- "Add forces: 10N at 30°, 15N at 120°"
- "Draw a free body diagram for a box on an inclined plane"
- "Calculate spring force with k=200 N/m compressed by 0.05m"

### Kinematics Agent (`kinematics_agent`)  
**Specialization**: Motion analysis and kinematics
- Projectile motion analysis
- Free fall calculations
- Constant acceleration problems
- Uniform motion analysis
- Relative motion problems

**Example Problems**:
- "A ball is thrown at 30 m/s at 45° from 10m height"
- "Car accelerates from rest at 2 m/s² for 5 seconds"
- "Object dropped from 50m height, find time to fall"

## 💻 Usage

### Starting the System
1. **Complete System**: Run `python start_system.py` for automatic startup
2. **Manual**: Start API first, then UI in separate terminals
3. **Development**: Use individual startup scripts for debugging

### Using the Interface
1. **Login**: Access Streamlit UI and authenticate
2. **Select Agent**: Choose between Forces or Kinematics agent
3. **Ask Questions**: Type physics problems in natural language
4. **Dynamic Switching**: Change agents anytime during session
5. **View Results**: Get detailed solutions with MCP tool integration

### API Usage
```python
import requests

# Create an agent
response = requests.post("http://localhost:8000/agent/create", json={
    "agent_id": "forces_agent",
    "use_direct_tools": True
})

# Solve a problem
response = requests.post("http://localhost:8000/agent/forces_agent/solve", json={
    "problem": "Add forces: 10N at 30°, 15N at 120°"
})
```

## 🔧 Configuration

### API Configuration
- **Host**: `0.0.0.0` (configurable in startup scripts)
- **Port**: `8000` (configurable)
- **Reload**: Enabled for development
- **Log Level**: Info

### UI Configuration  
- **Host**: `0.0.0.0`
- **Port**: `8501`
- **API Base URL**: `http://localhost:8000` (configurable in `services/api_client.py`)

### Agent Configuration
Agents are configured in `/api/agent.py`:
- **MCP Ports**: Forces (10100), Kinematics (10101)
- **LLM**: Ollama with qwen3:8b-q8_0 model
- **Mode**: Direct tools (recommended) or LangChain agent

## 🛠️ Development

### Project Structure
```
Physics-Assistant/UI/
├── api/                    # FastAPI backend
│   ├── main.py            # Main API server
│   ├── agent.py           # CombinedPhysicsAgent class
│   ├── prompts/           # Agent prompts
│   └── pyproject.toml     # API dependencies
├── frontend/              # Streamlit frontend
│   ├── app.py            # Main Streamlit app
│   ├── components/       # UI components
│   ├── services/         # API client services
│   ├── config.py         # Configuration
│   └── requirements.txt  # UI dependencies
├── start_system.py       # Complete system launcher
├── start_api.py          # API server launcher
├── start_ui.py           # UI launcher
├── memory.md            # Implementation progress
└── README.md            # This file
```

### Key Features
- **Dynamic Agent Switching**: Real-time agent selection in UI
- **Async API**: FastAPI with async/await for performance
- **MCP Integration**: Direct tool calls for physics calculations
- **Error Handling**: Comprehensive error handling and fallbacks
- **Documentation**: Auto-generated API docs
- **Health Monitoring**: Service health checks and status indicators

### Adding New Agents
1. **Backend**: Add agent configuration in `agent.py`
2. **Frontend**: Update agent info in `services/api_client.py`
3. **UI**: Extend agent capabilities in `components/agents.py`

## 🧪 Testing

### Manual Testing
```bash
# Test API health
curl http://localhost:8000/health

# Test agent creation
curl -X POST http://localhost:8000/agent/create \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "forces_agent", "use_direct_tools": true}'

# Test problem solving
curl -X POST http://localhost:8000/agent/forces_agent/solve \
  -H "Content-Type: application/json" \
  -d '{"problem": "Add forces: 10N at 30°, 15N at 120°"}'
```

### UI Testing
1. Navigate to http://localhost:8501
2. Login with credentials
3. Select different agents
4. Submit physics problems
5. Verify dynamic agent switching works

## 🔍 Troubleshooting

### Common Issues

#### API Server Not Starting
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process if needed
sudo kill -9 <PID>

# Check dependencies
python -c "import fastapi, uvicorn; print('API dependencies OK')"
```

#### UI Not Connecting to API
```bash
# Verify API is running
curl http://localhost:8000/health

# Check UI dependencies
python -c "import streamlit, requests; print('UI dependencies OK')"

# Check firewall/network settings
```

#### MCP Tools Not Working
- Ensure MCP servers are running on ports 10100 (forces) and 10101 (kinematics)
- Check agent initialization logs
- Verify MCP adapter dependencies

#### Agent Creation Failures
- Check agent_id is valid (`forces_agent` or `kinematics_agent`)
- Verify MCP server connectivity
- Check API server logs for detailed errors

### Debugging Mode
```bash
# Start API with debug logging
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

# Start UI with verbose output
cd frontend/
streamlit run app.py --logger.level debug
```

## 🚦 Health Checks

### API Health
- **Endpoint**: `GET /health`
- **Expected**: `{"status": "healthy", "active_agents": N}`

### Agent Health  
- **Endpoint**: `GET /agent/{agent_id}/health`
- **Expected**: `{"status": "healthy", "ready": true}`

### UI Health
- Access http://localhost:8501
- Verify agent selection works
- Check connection indicators in sidebar

## 📝 Logs

### API Logs
- Console output shows request/response details
- Error logs include stack traces
- Health check logs show agent status

### UI Logs
- Streamlit console shows page interactions
- API connection status in UI
- Error messages displayed to users

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## 📄 License

[Add appropriate license information]

## 🆘 Support

For issues and questions:
1. Check this README
2. Review API documentation at `/docs`
3. Check logs for error details
4. Create issue in repository

---

**🎉 Happy Physics Problem Solving!**