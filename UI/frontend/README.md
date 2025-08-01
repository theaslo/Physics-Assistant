# Physics Assistant UI - Streamlit Implementation

A comprehensive Streamlit-based user interface for a Physics Assistant designed for college-level introductory physics students. The UI serves as the frontend for a distributed system utilizing MCP (Model Context Protocol) servers running Ollama models.

## 🌟 Features

### 🔐 Authentication System
- Secure login with username/password
- Role-based access (student/instructor)
- Session state management
- Demo credentials included

### 🤖 Physics Agents
- **Kinematics Agent**: 1D and 2D motion problems
- **Forces Agent**: Newton's laws and vector analysis
- **Work and Energy Agent**: Energy conservation and work calculations
- **Momentum Agent**: Collision and momentum problems
- **Rotation Agent**: Rotational motion and dynamics
- **Math Helper Agent**: Trigonometry and algebra support

### 💬 Interactive Chat Interface
- Real-time physics tutoring
- LaTeX equation rendering
- File upload for physics diagrams
- Quick action buttons for common topics
- Message history and context

### 📊 Progress Tracking
- Learning analytics dashboard
- Session statistics
- Topic mastery tracking
- Performance visualization
- Data export capabilities

### ⚙️ Settings & Customization
- UI preferences
- Learning preferences
- Data management options
- Physics constants reference

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone or download the project files to your directory**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser to:**
   ```
   http://localhost:8501
   ```

### Demo Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Student | `student1` | `password123` |
| Student | `student2` | `password123` |
| Instructor | `instructor` | `instructor123` |

## 📁 Project Structure

```
physics_assistant_ui/
├── app.py                      # Main application entry point
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── pages/                      # Streamlit pages
│   ├── login.py               # Authentication page
│   ├── progress.py            # Analytics dashboard
│   └── settings.py            # User preferences
│
├── components/                 # Reusable UI components
│   ├── auth.py                # Authentication logic
│   ├── chat.py                # Chat interface
│   └── agents.py              # Agent management
│
├── services/                   # Backend services
│   ├── mcp_client.py          # MCP server communication
│   └── data_manager.py        # Session and progress data
│
├── utils/                      # Utility modules
│   ├── constants.py           # Physics constants and formulas
│   ├── formatters.py          # Data formatting utilities
│   └── validators.py          # Input validation
│
└── assets/                     # Static assets
    ├── images/                # UI images and icons
    └── styles/                # Custom CSS (if needed)
```

## 🎯 Usage Guide

### Getting Started

1. **Login**: Use the demo credentials to access the system
2. **Select Agent**: Choose a physics agent from the sidebar
3. **Start Chatting**: Ask physics questions or upload problem images
4. **Track Progress**: View your learning analytics in the Progress page
5. **Customize**: Adjust preferences in the Settings page

### Physics Agents Overview

#### 🚀 Kinematics Agent
- Position, velocity, and acceleration problems
- Motion graphs and kinematic equations
- Projectile motion analysis

#### ⚡ Forces Agent
- Free body diagrams
- Newton's laws applications
- Friction and tension problems

#### 🔋 Work and Energy Agent
- Work calculations (W = F·d)
- Energy conservation problems
- Power and efficiency

#### 💥 Momentum Agent
- Linear momentum (p = mv)
- Collision analysis
- Conservation of momentum

#### 🌀 Rotation Agent
- Angular motion
- Torque calculations
- Moment of inertia

#### 📐 Math Helper Agent
- Vector operations
- Trigonometric calculations
- Unit conversions

### Input Methods

1. **Text Chat**: Type physics questions naturally
2. **LaTeX Equations**: Enter mathematical expressions
3. **Image Upload**: Upload physics problem diagrams
4. **Quick Actions**: Use predefined topic buttons

## ⚙️ Configuration

### Environment Variables

The application supports the following environment variables:

```bash
# MCP Server Configuration
MCP_SERVER_URL=http://localhost:8000
MCP_WEBSOCKET_URL=ws://localhost:8000/ws
MCP_API_KEY=your_api_key_here

# Database Configuration
DB_PATH=data/physics_assistant.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/physics_assistant.log
```

### Configuration File

Key settings can be modified in `config.py`:

```python
# Application Settings
APP_NAME = "Physics Assistant UI"
PAGE_TITLE = "Physics Assistant"
MAX_CHAT_HISTORY = 100
MAX_FILE_SIZE_MB = 5

# Physics Agents Configuration
PHYSICS_AGENTS = {
    "kinematics": {
        "name": "Kinematics Agent",
        "description": "1D and 2D motion problems",
        "icon": "🚀"
    },
    # ... more agents
}
```

## 🔧 Development

### Running in Development Mode

```bash
# Enable debug mode
streamlit run app.py --logger.level=debug

# Run on specific port
streamlit run app.py --server.port=8502
```

### Code Structure

The application follows a modular architecture:

- **Components**: Reusable UI elements
- **Services**: Backend communication logic
- **Utils**: Helper functions and utilities
- **Pages**: Streamlit page definitions

### Adding New Features

1. **New Physics Agent**: Add to `PHYSICS_AGENTS` in `config.py`
2. **New Page**: Create in `pages/` directory
3. **New Component**: Add to `components/` directory
4. **New Utility**: Add to `utils/` directory

## 🧪 Testing

### Manual Testing

1. Test authentication with different user roles
2. Verify agent selection and switching
3. Test chat functionality with various inputs
4. Check file upload capabilities
5. Verify progress tracking and analytics

### Automated Testing (Future)

```bash
# Unit tests (when implemented)
python -m pytest tests/

# Integration tests (when implemented)
python -m pytest tests/integration/
```

## 📦 Deployment

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push code to GitHub repository
2. Connect to Streamlit Cloud
3. Configure secrets for API keys
4. Deploy with automatic updates

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🛠️ MCP Server Integration

### Connection Setup

The application connects to MCP servers for AI physics assistance:

```python
# HTTP API endpoints
POST /api/auth/login
GET /api/agents/available
POST /api/agents/select
POST /api/chat/message

# WebSocket connection
ws://localhost:8000/ws
```

### Message Format

```json
{
    "agent": "kinematics",
    "content": "Help me solve this velocity problem",
    "user_id": "student123",
    "session_id": "session_abc123"
}
```

## 🔒 Security

### Authentication
- Bcrypt password hashing
- Session-based authentication
- Input validation and sanitization

### Data Protection
- No sensitive data in logs
- Secure session state management
- File upload validation

### Privacy
- Optional analytics collection
- User data export capabilities
- Session data management

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **MCP Server Connection**: Check server URL and status
3. **File Upload Issues**: Verify file size and type limits
4. **Session Problems**: Clear browser cache or restart app

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Support

For issues and questions:
- Check the troubleshooting section above
- Review the physics_assistant_ui_spec.md file
- Contact: physics-help@university.edu

## 📈 Roadmap

### Phase 1 ✅ (Completed)
- [x] Core Streamlit setup
- [x] Authentication system
- [x] Basic chat interface
- [x] Agent selection

### Phase 2 (In Progress)
- [ ] MCP server integration
- [ ] Real-time WebSocket communication
- [ ] Advanced physics visualization
- [ ] File upload processing

### Phase 3 (Planned)
- [ ] Learning analytics improvements
- [ ] Advanced equation rendering
- [ ] Multi-language support
- [ ] Mobile responsiveness

### Phase 4 (Future)
- [ ] Offline mode support
- [ ] Advanced AI features
- [ ] Integration with LMS systems
- [ ] Performance optimizations

## 📄 License

This project is for educational purposes. Please respect the terms of use for all dependencies and frameworks used.

## 🙏 Acknowledgments

- **Streamlit**: For the excellent web framework
- **Physics Community**: For domain expertise and requirements
- **MCP Protocol**: For enabling distributed AI systems
- **Educational Institutions**: For use case validation

---

**Note**: This is a demonstration implementation showing the complete Streamlit-based UI structure for a physics tutoring system. The MCP server integration includes placeholder responses for development and testing purposes.