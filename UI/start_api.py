#!/usr/bin/env python3
"""
Startup script for Physics Assistant API server
"""

import sys
import os
import subprocess
import signal
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ FastAPI dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install dependencies with: uv pip install -e ./api")
        return False

def start_api_server():
    """Start the FastAPI server"""
    # Change to API directory
    api_dir = Path(__file__).parent / "api"
    os.chdir(api_dir)
    
    print("🚀 Starting Physics Assistant API server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔧 Health check at: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Shutting down API server...")
        sys.exit(0)

def main():
    """Main startup function"""
    print("=" * 60)
    print("🔬 Physics Assistant API Server")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    start_api_server()

if __name__ == "__main__":
    main()