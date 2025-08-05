#!/usr/bin/env python3
"""
Test script to reproduce the double question issue
"""

import sys
import os
sys.path.append('frontend')

from services.api_client import PhysicsAPIClient

def test_agent_flow():
    """Test the exact flow that happens in the UI"""
    print("=== Testing Agent Flow ===")
    
    client = PhysicsAPIClient()
    agent_id = "math_agent"
    
    print(f"1. Testing API connection...")
    if not client.is_connected():
        print("❌ API not connected")
        return
    print("✅ API connected")
    
    print(f"2. Creating agent: {agent_id}")
    result = client.create_agent(agent_id)
    print(f"   Result: {result}")
    
    if not result.get('success', False):
        print(f"❌ Agent creation failed: {result.get('error')}")
        return
    print("✅ Agent created")
    
    print(f"3. Sending first message...")
    response1 = client.send_message(
        agent_id=agent_id,
        message="What is 2+2?",
        user_id="test_user"
    )
    print(f"   Response 1: {response1.get('success')} - {response1.get('solution', response1.get('error'))[:100]}...")
    
    print(f"4. Sending second message...")
    response2 = client.send_message(
        agent_id=agent_id,
        message="What is 3+3?",
        user_id="test_user"
    )
    print(f"   Response 2: {response2.get('success')} - {response2.get('solution', response2.get('error'))[:100]}...")
    
    print(f"5. Sending third message...")
    response3 = client.send_message(
        agent_id=agent_id,
        message="What is 4+4?",
        user_id="test_user"
    )
    print(f"   Response 3: {response3.get('success')} - {response3.get('solution', response3.get('error'))[:100]}...")

if __name__ == "__main__":
    test_agent_flow()