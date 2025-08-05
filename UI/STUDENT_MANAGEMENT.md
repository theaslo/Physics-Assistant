# Physics Assistant - Student Management Guide

## Overview
This guide explains how to add and manage student accounts for the Physics Assistant tutoring system.

## Current System Status ✅
- **System is working** - All physics agents are functional
- **Authentication enabled** - Students need login credentials
- **5 Physics domains available**: Forces, Kinematics, Math, Momentum, Angular Motion
- **Web interface** accessible at: `http://localhost:8501`

## Quick Start - Demo Accounts

**Ready-to-use student accounts:**
```
Username: student1     Password: password123
Username: student2     Password: password123
Username: alice_smith  Password: physics2024
Username: bob_jones    Password: physics2024
```

**Instructor account:**
```
Username: instructor   Password: instructor123
```

## Adding New Students

### Method 1: Quick Manual Addition (Recommended)

1. **Edit the authentication file:**
   ```bash
   nano frontend/components/auth.py
   ```

2. **Find the `usernames` section** (around line 18)

3. **Add your student following this template:**
   ```python
   'student_username': {
       'name': 'Student Full Name',
       'password': self._hash_password('their_password'),
       'email': 'student@university.edu',
       'role': 'student',
       'course': 'PHYS101'
   },
   ```

4. **Example - Adding John Smith:**
   ```python
   'john_smith': {
       'name': 'John Smith',
       'password': self._hash_password('physics2024'),
       'email': 'john.smith@university.edu',
       'role': 'student',
       'course': 'PHYS101'
   },
   ```

5. **Save the file** and students can immediately log in

### Method 2: Using the Student Addition Tool

1. **Run the helper script:**
   ```bash
   python add_student.py
   ```

2. **Enter student information when prompted:**
   - Username (e.g., john_doe)
   - Full name (e.g., John Doe)
   - Email address
   - Password
   - Course code (default: PHYS101)

3. **Copy the generated code** into `frontend/components/auth.py`

### Method 3: Bulk Student Addition

For adding multiple students at once, copy this template and modify:

```python
# Add to the 'usernames' section in auth.py:
'sarah_wilson': {
    'name': 'Sarah Wilson',
    'password': self._hash_password('physics2024'),
    'email': 'sarah.wilson@university.edu',
    'role': 'student',
    'course': 'PHYS101'
},
'mike_chen': {
    'name': 'Mike Chen',
    'password': self._hash_password('physics2024'),
    'email': 'mike.chen@university.edu',
    'role': 'student',
    'course': 'PHYS101'
},
'emma_rodriguez': {
    'name': 'Emma Rodriguez',
    'password': self._hash_password('physics2024'),
    'email': 'emma.rodriguez@university.edu',
    'role': 'student',
    'course': 'PHYS101'
},
```

## System Startup Instructions

### Starting the Physics Assistant

1. **Start the API server:**
   ```bash
   cd /home/atk21004admin/Physics-Assistant/UI
   python start_api.py
   ```

2. **Start the web interface:**
   ```bash
   cd frontend
   streamlit run app.py
   ```

3. **Access the system:**
   - Open browser to: `http://localhost:8501`
   - Students log in with their credentials
   - Select a physics agent and start asking questions

### One-Command Startup (Alternative)
```bash
python start_system.py
```

## Student Instructions

**Share these instructions with your students:**

### How to Use the Physics Assistant

1. **Access the system:**
   - Go to: `http://localhost:8501`
   - Use the username/password provided by your instructor

2. **Choose a physics domain:**
   - **Forces Agent** ⚖️ - Force analysis, Newton's laws, free body diagrams
   - **Kinematics Agent** 🚀 - Motion analysis, projectile motion
   - **Math Agent** 🔢 - Mathematical calculations, algebra, trigonometry
   - **Momentum Agent** 💥 - Momentum, impulse, collisions
   - **Angular Motion Agent** 🌀 - Rotational motion, torque

3. **Ask questions:**
   - Type physics questions in natural language
   - Examples: "What is Newton's second law?"
   - "Solve: A 5kg box slides down a 30° incline with friction coefficient 0.3"
   - "Calculate the projectile motion for initial velocity 20 m/s at 45°"

4. **Get detailed solutions:**
   - Step-by-step explanations
   - Mathematical derivations
   - Conceptual explanations

## Security & Password Management

### Password Security
- All passwords are automatically encrypted with bcrypt
- Students cannot see other students' data
- Each student has independent chat history

### Recommended Passwords
- **Simple**: Use a common class password like `physics2024`
- **Secure**: Use unique passwords per student
- **Pattern**: Use student ID + course (e.g., `student123_phys101`)

### Password Reset
Students who forget passwords should contact the instructor. To reset:
1. Find the student in `auth.py`
2. Change their password line:
   ```python
   'password': self._hash_password('new_password'),
   ```

## Monitoring Student Usage

### View Active Sessions
Check which students are currently using the system:
```bash
curl http://localhost:8000/health
```

### Chat History
Each student's questions and answers are stored in their browser session. Data is not persistent between sessions for privacy.

## Troubleshooting

### Common Issues

**"Agent not responding"**
- Check that the API server is running: `python start_api.py`
- Verify at: `http://localhost:8000/health`

**"Cannot connect to server"**
- Restart the API server
- Check MCP servers are running (ports 10100-10106)

**"Login not working"**
- Verify username/password in `auth.py`
- Check for typos in the authentication file
- Ensure proper syntax (commas, quotes, indentation)

### Getting Help
- Check system logs in the terminal where you started the servers
- Verify all services are running: API server + Streamlit UI
- Contact system administrator if issues persist

## System Requirements

- **Python 3.8+**
- **Required packages**: streamlit, fastapi, bcrypt
- **Network**: Students need access to `localhost:8501`
- **Memory**: ~500MB RAM for full system
- **Storage**: Minimal (no persistent data stored)

## Scaling for Large Classes

### Current Capacity
- **Recommended**: Up to 50 concurrent students
- **File-based auth**: Suitable for single classes
- **Memory usage**: ~10MB per active student

### For Larger Classes (100+ students)
Consider upgrading to:
- Database-based authentication (PostgreSQL/MySQL)
- External authentication (LDAP/Active Directory)
- Load balancing for multiple API servers
- Persistent chat history storage

## File Locations

```
Physics-Assistant/UI/
├── frontend/components/auth.py          # Student accounts (EDIT THIS)
├── start_api.py                         # Start API server
├── start_system.py                      # Start everything
├── add_student.py                       # Helper tool
└── frontend/app.py                      # Main web interface
```

## Backup & Recovery

### Backup Student Data
```bash
cp frontend/components/auth.py auth_backup_$(date +%Y%m%d).py
```

### Restore from Backup
```bash
cp auth_backup_YYYYMMDD.py frontend/components/auth.py
```

---

## Summary for Instructors

1. **Current Status**: System is ready with 4 demo student accounts
2. **Add Students**: Edit `frontend/components/auth.py` file
3. **Start System**: Run `python start_system.py`
4. **Student Access**: `http://localhost:8501`
5. **Support**: Students can immediately start asking physics questions

**Need help?** Check the troubleshooting section or restart the system components.