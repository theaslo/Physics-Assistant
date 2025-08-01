#!/usr/bin/env python3
"""
Simple shutdown script for Physics Assistant system
No external dependencies required - uses only built-in Python modules
"""

import subprocess
import sys
import time
import signal
import os

def stop_physics_services():
    """Stop Physics Assistant services using basic commands"""
    print("=" * 60)
    print("🛑 Physics Assistant System Shutdown (Simple Mode)")
    print("=" * 60)
    
    success = True
    
    # Method 1: Kill by process name patterns
    print("\n🔍 Stopping services by process patterns...")
    
    patterns = [
        ('uvicorn', 'API Server'),
        ('streamlit', 'Streamlit UI'),
        ('start_system.py', 'System Launcher'),
        ('start_api.py', 'API Launcher'),
        ('start_ui.py', 'UI Launcher')
    ]
    
    for pattern, description in patterns:
        try:
            print(f"🔴 Stopping {description}...")
            
            # First try graceful termination
            result = subprocess.run(['pkill', '-TERM', '-f', pattern], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   Sent TERM signal to {description} processes")
                time.sleep(2)  # Give time for graceful shutdown
                
                # Check if still running and force kill if needed
                check_result = subprocess.run(['pgrep', '-f', pattern], 
                                            capture_output=True, text=True)
                if check_result.returncode == 0:
                    print(f"   Force killing stubborn {description} processes...")
                    subprocess.run(['pkill', '-KILL', '-f', pattern], 
                                 capture_output=True, text=True)
                    
                print(f"✅ {description} stopped")
            else:
                print(f"✅ No {description} processes found")
                
        except Exception as e:
            print(f"⚠️ Error stopping {description}: {e}")
            success = False
    
    # Method 2: Kill by port (if lsof is available)
    print("\n🔍 Checking ports 8000 and 8501...")
    
    for port, service in [(8000, 'API'), (8501, 'UI')]:
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"🔴 Killing {service} processes on port {port}...")
                for pid in pids:
                    if pid.strip():
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(1)
                            # Force kill if still running
                            try:
                                os.kill(int(pid), signal.SIGKILL)
                            except:
                                pass  # Process already dead
                        except:
                            pass
                print(f"✅ Port {port} freed")
            else:
                print(f"✅ Port {port} already free")
        except FileNotFoundError:
            print(f"⚠️ lsof not available, skipping port {port} check")
        except Exception as e:
            print(f"⚠️ Error checking port {port}: {e}")
    
    # Method 3: Final cleanup
    print("\n🧹 Final cleanup...")
    
    # Kill any remaining python processes with our scripts
    try:
        subprocess.run(['pkill', '-f', 'Physics-Assistant'], 
                      capture_output=True, text=True)
        print("✅ Cleanup completed")
    except:
        pass
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Physics Assistant shutdown completed!")
        print("💡 You can restart with: python start_system.py")
    else:
        print("⚠️ Shutdown completed with some warnings")
        print("💡 Check for any remaining processes manually if needed")
    print("=" * 60)
    
    return success

def show_simple_status():
    """Show service status using basic commands"""
    print("=" * 60)
    print("📊 Physics Assistant Service Status (Simple Mode)")
    print("=" * 60)
    
    # Check for running processes
    patterns = [
        ('uvicorn', 'API Server'),
        ('streamlit', 'Streamlit UI'),
        ('start_system.py', 'System Launcher')
    ]
    
    print("\nRunning Processes:")
    for pattern, description in patterns:
        try:
            result = subprocess.run(['pgrep', '-f', pattern], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                print(f"  🟢 {description}: {len(pids)} process(es) - PIDs: {', '.join(pids)}")
            else:
                print(f"  ⚪ {description}: Not running")
        except:
            print(f"  ❓ {description}: Unable to check")
    
    # Check ports
    print("\nPort Status:")
    for port, service in [(8000, 'API'), (8501, 'UI')]:
        try:
            result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                print(f"  🟢 Port {port} ({service}): In use")
            else:
                print(f"  ⚪ Port {port} ({service}): Available")
        except:
            print(f"  ❓ Port {port} ({service}): Unable to check")
    
    print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Physics Assistant shutdown")
    parser.add_argument('--status', action='store_true',
                       help='Show service status instead of stopping')
    
    args = parser.parse_args()
    
    if args.status:
        show_simple_status()
    else:
        try:
            stop_physics_services()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown interrupted by user")
            sys.exit(1)

if __name__ == "__main__":
    main()