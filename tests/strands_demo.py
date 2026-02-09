#!/usr/bin/env python3
"""
Strands SDK Demo - Kinematics Agent
====================================
Simple standalone example showing Strands SDK with:
- Ollama LLM (qwen3:8b-q8_0)
- MCP Kinematics tools via streamable HTTP

Usage:
    # From inside Docker container:
    docker exec -it physics-agents-api uv run python /app/strands_demo.py

    # Or locally with uv:
    cd UI/api && uv run python ../../tests/strands_demo.py --local
"""

import os
import sys
import asyncio
from strands import Agent
from strands.models.ollama import OllamaModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamablehttp_client


# Configuration - detect environment
OLLAMA_HOST = "http://ds.stat.uconn.edu:11434"
OLLAMA_MODEL = "qwen3:8b-q8_0"

# Use Docker service name by default, localhost if --local flag
if "--local" in sys.argv:
    MCP_HOST = "localhost"
else:
    MCP_HOST = os.getenv("MCP_KINEMATICS_HOST", "mcp-kinematics")

MCP_PORT = 10101


async def main():
    print("=" * 60)
    print("Strands SDK Demo - Kinematics Agent")
    print("=" * 60)

    # Step 1: Connect to MCP server
    mcp_url = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
    print(f"\n1. Connecting to MCP server at {mcp_url}...")

    mcp_client = MCPClient(
        lambda: streamablehttp_client(url=mcp_url),
        startup_timeout=30
    )
    mcp_client.start()

    # Get available tools
    tools = mcp_client.list_tools_sync()
    print(f"   ✓ Loaded {len(tools)} tools:")
    for tool in tools:
        print(f"     - {tool.tool_name}")

    # Step 2: Create Ollama model
    print(f"\n2. Creating Ollama model ({OLLAMA_MODEL})...")
    ollama_model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=OLLAMA_MODEL,
        temperature=0.1,
    )
    print("   ✓ Ollama model ready")

    # Step 3: Create Strands agent
    print("\n3. Creating Strands agent...")
    agent = Agent(
        model=ollama_model,
        tools=tools,
        system_prompt="""You are a physics tutor specializing in kinematics.
Use the MCP tools to solve problems. Always show your work.""",
        name="kinematics_demo",
    )
    print("   ✓ Strands agent created")

    # Step 4: Solve a problem
    problem = "A ball is dropped from a height of 20 meters. How long does it take to hit the ground?"
    print(f"\n4. Solving problem:")
    print(f"   '{problem}'")
    print("\n   Working...\n")

    result = agent(problem)

    # Extract response
    solution = ""
    if result.message and result.message.get("content"):
        for block in result.message["content"]:
            if "text" in block:
                solution += block["text"]

    # Show tools used
    tools_used = []
    for msg in agent.messages:
        for block in msg.get("content", []):
            if "toolUse" in block:
                tools_used.append(block["toolUse"]["name"])

    print("-" * 60)
    print("SOLUTION:")
    print("-" * 60)
    print(solution)
    print("-" * 60)
    print(f"\nTools used: {list(set(tools_used))}")

    # Cleanup
    mcp_client.stop(None, None, None)
    agent.cleanup()

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
