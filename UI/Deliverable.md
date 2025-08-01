# Preface for Claude

This document outlines the project for implemeting a front end that comprises of a streamlit UI application found in `frontend` and an api, to be create, that will host the class CombinedPhysicsAgent (api/agent.py)

# Claude deliverable

1. An FastAPI server the hosts a endpoint for CombinedPhysicsAgent (api/agent.py). API should host  the correct function of 
```python
    return CombinedPhysicsAgent(
        agent_id=<USER REQUESTED AGENT>, 
        use_direct_tools=use_direct_tools
    )
```
- See /api/agent.py create_kinematics_agent() for guidance
2. Modify streamlit UI `/frontend/app.py` to use the api.

# Claude Objectives

1. A well constructed, documented and working API to serve CombinedPhysicsAgent and the user agent chosen from streamlit
2. Streamlit modifications
    - Ensure this is dynamic, meaning user can change agent anytime while logged in and the agent_id will also change to the users preference.
3. Claude must think of new ways to make this better, maybe pydantic v2 models, maybe better UI configuration. All news ideas must be approved by me, the user.
4. Must ask questions if unlcear to any objective or deliverable.

# Tools
Use the following tools
- fetch
    - to grab information from url
- WebFetch
    - given a url and context, search url
# Claude User Preferences
## History
Use the memory.md file to document which steps have been completed and how. This file should help you know where to continue if you get disconnected and need to pick up where you left off. If file exists use it. If file does not exist create this file first and update as you work on the project, step-by-step. 

### Structure of Memory
- What has been done
    - Why it was done in that manner
- What is left to be done
    - How it needs to be done
    - Why it needs to be done in that manner
- Useful remarks for future instances of Claude to use when picking up where it was left off

## Thought process 
- Always think of ways to improve or find better methods for a more impactful deliverable. Suggest ideas that could enhance the deliverable's effectiveness.

# Summary instructions
- Ensure all code changes are well-documented and tested.

## Other Preferences
- Use secure and efficient coding practices.
- Ensure the system is scalable and maintainable.
- Must use uv for python management
    - https://docs.astral.sh/uv/getting-started/installation/
        - Tool
            - WebFetch: https://docs.astral.sh/uv/getting-started/installation/ "How to use uv for python management"  

# Working with Claude Principles

1. **Clarity First**: Provide clear, concise instructions with specific implementation details.
2. **Contextual Understanding**: Ensure Claude understands the insurance industry's needs and regulatory environment.
3. **Iterative Development**: Break down complex enhancements into manageable steps.
4. **Test-Driven Development**: Define expected outcomes before implementation.
5. **Documentation**: Maintain comprehensive documentation of all changes.
6. **Error Handling**: Implement robust error handling in all new components.
7. **User Feedback**: Incorporate mechanisms for user feedback in new features.
8. **Performance Consideration**: Always consider performance implications of new features.
9. **Security First**: Implement security best practices in all new code.
10. **Maintainability**: Write code that is easy to maintain and extend.
11. Read working_with_claude.md to learn more about working with yourself in this project.