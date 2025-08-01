#!/usr/bin/env python3
"""
Complete system startup script for Physics Assistant
Starts both API server and Streamlit UI
"""

import sys
import subprocess
import time
import signal
from pathlib import Path

class PhysicsAssistantLauncher:
    """Manages starting and stopping the complete Physics Assistant system"""
    
    def __init__(self):
        self.api_process = None
        self.ui_process = None
        self.running = True
    
    def signal_handler(self, signum, _):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all processes"""
        self.running = False
        
        if self.ui_process:
            print("🔴 Stopping Streamlit UI...")
            self.ui_process.terminate()
            self.ui_process.wait()
        
        if self.api_process:
            print("🔴 Stopping API server...")
            self.api_process.terminate()
            self.api_process.wait()
        
        print("✅ All services stopped")
        sys.exit(0)
    
    def start_api_server(self):
        """Start the API server in background"""
        print("🚀 Starting API server...")
        api_dir = Path(__file__).parent / "api"
        
        self.api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=api_dir)
        
        # Wait for API to be ready
        print("⏳ Waiting for API server to start...")
        for _ in range(30):  # Wait up to 30 seconds
            try:
                import requests
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server is ready")
                    return True
            except:
                time.sleep(1)
        
        print("❌ API server failed to start")
        return False
    
    def start_ui(self):
        """Start the Streamlit UI in background"""
        print("🚀 Starting Streamlit UI...")
        frontend_dir = Path(__file__).parent / "frontend"
        
        self.ui_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "app.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--browser.gatherUsageStats", "false"
        ], cwd=frontend_dir)
        
        print("✅ Streamlit UI started")
        return True
    
    def monitor_processes(self):
        """Monitor the health of all processes"""
        while self.running:
            try:
                # Check API process
                if self.api_process and self.api_process.poll() is not None:
                    print("❌ API server process died")
                    self.shutdown()
                    return
                
                # Check UI process
                if self.ui_process and self.ui_process.poll() is not None:
                    print("❌ UI process died")
                    self.shutdown()
                    return
                
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                self.shutdown()
                return
    
    def start_system(self):
        """Start the complete system"""
        print("=" * 60)
        print("🔬 Physics Assistant Complete System Launcher")
        print("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start API server
            if not self.start_api_server():
                print("❌ Failed to start API server")
                sys.exit(1)
            
            # Start UI
            if not self.start_ui():
                print("❌ Failed to start UI")
                self.shutdown()
                sys.exit(1)
            
            print("\n" + "=" * 60)
            print("🎉 Physics Assistant is now running!")
            print("📍 API Server: http://localhost:8000")
            print("📍 API Docs: http://localhost:8000/docs")
            print("📍 Streamlit UI: http://127.0.0.1:8501")
            print("🔍 Health Check: http://localhost:8000/health")
            print("=" * 60)
            print("\nPress Ctrl+C to stop all services")
            
            # Monitor processes
            self.monitor_processes()
            
        except Exception as e:
            print(f"❌ Error starting system: {e}")
            self.shutdown()
            sys.exit(1)

def main():
    """Main function"""
    launcher = PhysicsAssistantLauncher()
    launcher.start_system()

if __name__ == "__main__":
    main()