#!/usr/bin/env python3
"""
Startup script for Physics Assistant Streamlit UI
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import requests
        print("✅ Streamlit dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies with: uv pip install -r ./frontend/requirements.txt")
        return False

def check_api_server():
    """Check if API server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("⚠️ API server responded with error")
            return False
    except Exception:
        print("❌ API server not accessible at http://localhost:8000")
        print("Please start the API server first using: python start_api.py")
        return False

def start_streamlit_app():
    """Start the Streamlit application"""
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    print("🚀 Starting Physics Assistant UI...")
    print("📍 UI will be available at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the UI\n")
    
    try:
        # Start streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Streamlit UI...")
        sys.exit(0)

def main():
    """Main startup function"""
    print("=" * 60)
    print("🔬 Physics Assistant Streamlit UI")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🔍 Checking API server connection...")
    if not check_api_server():
        print("\n💡 To start the complete system:")
        print("   1. Terminal 1: python start_api.py")
        print("   2. Terminal 2: python start_ui.py")
        sys.exit(1)
    
    start_streamlit_app()

if __name__ == "__main__":
    main()