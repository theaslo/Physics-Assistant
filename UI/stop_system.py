#!/usr/bin/env python3
"""
Shutdown script for Physics Assistant system
Stops all running API and UI services regardless of how they were started
"""

import subprocess
import sys
import time
import signal
import os
from typing import List, Dict, Optional

# Try to import psutil, but fall back to basic functionality if not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not available - using basic process management")

class PhysicsAssistantStopper:
    """Manages stopping all Physics Assistant services"""
    
    def __init__(self):
        self.verbose = True
    
    def find_physics_processes(self) -> Dict[str, List]:
        """Find all running Physics Assistant processes"""
        processes = {
            'api': [],
            'ui': [],
            'system': []
        }
        
        if HAS_PSUTIL:
            return self._find_processes_psutil()
        else:
            return self._find_processes_basic()
    
    def _find_processes_psutil(self) -> Dict[str, List]:
        """Find processes using psutil (more accurate)"""
        processes = {
            'api': [],
            'ui': [],
            'system': []
        }
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Look for API processes (uvicorn/FastAPI)
                    if any(keyword in cmdline.lower() for keyword in [
                        'uvicorn main:app',
                        'fastapi',
                        'main:app',
                        'physics-assistant/ui/api'
                    ]):
                        processes['api'].append(proc)
                    
                    # Look for UI processes (streamlit)
                    elif any(keyword in cmdline.lower() for keyword in [
                        'streamlit run app.py',
                        'streamlit run',
                        'physics-assistant/ui/frontend'
                    ]):
                        processes['ui'].append(proc)
                    
                    # Look for system launcher processes
                    elif any(keyword in cmdline.lower() for keyword in [
                        'start_system.py',
                        'start_api.py',
                        'start_ui.py'
                    ]):
                        processes['system'].append(proc)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Warning: Error scanning processes: {e}")
        
        return processes
    
    def _find_processes_basic(self) -> Dict[str, List]:
        """Find processes using basic commands (fallback)"""
        processes = {
            'api': [],
            'ui': [],
            'system': []
        }
        
        try:
            # Use ps command to find processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode != 0:
                return processes
            
            lines = result.stdout.splitlines()
            for line in lines[1:]:  # Skip header
                if any(keyword in line.lower() for keyword in [
                    'uvicorn main:app',
                    'uvicorn',
                    'fastapi'
                ]):
                    # Extract PID (second column)
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            processes['api'].append(pid)
                        except ValueError:
                            continue
                
                elif any(keyword in line.lower() for keyword in [
                    'streamlit run',
                    'streamlit'
                ]):
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            processes['ui'].append(pid)
                        except ValueError:
                            continue
                
                elif any(keyword in line.lower() for keyword in [
                    'start_system.py',
                    'start_api.py',
                    'start_ui.py'
                ]):
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            processes['system'].append(pid)
                        except ValueError:
                            continue
                            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Warning: Error scanning processes: {e}")
        
        return processes
    
    def stop_processes(self, processes: List, service_name: str) -> bool:
        """Stop a list of processes gracefully"""
        if not processes:
            if self.verbose:
                print(f"✅ No {service_name} processes found")
            return True
        
        print(f"🔴 Stopping {len(processes)} {service_name} process(es)...")
        
        if HAS_PSUTIL:
            return self._stop_processes_psutil(processes, service_name)
        else:
            return self._stop_processes_basic(processes, service_name)
    
    def _stop_processes_psutil(self, processes: List, service_name: str) -> bool:
        """Stop processes using psutil (more accurate)"""
        # First, try graceful termination (SIGTERM)
        terminated = []
        for proc in processes:
            try:
                if self.verbose:
                    print(f"   Terminating PID {proc.pid}: {proc.name()}")
                proc.terminate()
                terminated.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Wait for graceful shutdown
        if terminated:
            print(f"⏳ Waiting for {service_name} processes to terminate gracefully...")
            gone, still_alive = psutil.wait_procs(terminated, timeout=10)
            
            # Force kill any remaining processes
            if still_alive:
                print(f"💥 Force killing {len(still_alive)} stubborn {service_name} process(es)...")
                for proc in still_alive:
                    try:
                        if self.verbose:
                            print(f"   Force killing PID {proc.pid}: {proc.name()}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Final wait
                psutil.wait_procs(still_alive, timeout=5)
        
        # Verify all processes are stopped
        remaining = []
        for proc in processes:
            try:
                if proc.is_running():
                    remaining.append(proc)
            except psutil.NoSuchProcess:
                continue
        
        if remaining:
            print(f"⚠️ Warning: {len(remaining)} {service_name} process(es) may still be running")
            return False
        else:
            print(f"✅ All {service_name} processes stopped successfully")
            return True
    
    def _stop_processes_basic(self, processes: List, service_name: str) -> bool:
        """Stop processes using basic kill commands (fallback)"""
        success = True
        
        # First try SIGTERM (graceful)
        for pid in processes:
            try:
                if self.verbose:
                    print(f"   Terminating PID {pid}")
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                continue
        
        # Wait a bit for graceful shutdown
        if processes:
            print(f"⏳ Waiting for {service_name} processes to terminate gracefully...")
            time.sleep(5)
        
        # Check which processes are still running and force kill if needed
        still_alive = []
        for pid in processes:
            try:
                os.kill(pid, 0)  # Check if process exists
                still_alive.append(pid)
            except (OSError, ProcessLookupError):
                continue
        
        # Force kill remaining processes
        if still_alive:
            print(f"💥 Force killing {len(still_alive)} stubborn {service_name} process(es)...")
            for pid in still_alive:
                try:
                    if self.verbose:
                        print(f"   Force killing PID {pid}")
                    os.kill(pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    continue
            
            time.sleep(2)  # Wait for force kill to take effect
        
        # Final verification
        final_remaining = []
        for pid in processes:
            try:
                os.kill(pid, 0)
                final_remaining.append(pid)
            except (OSError, ProcessLookupError):
                continue
        
        if final_remaining:
            print(f"⚠️ Warning: {len(final_remaining)} {service_name} process(es) may still be running")
            success = False
        else:
            print(f"✅ All {service_name} processes stopped successfully")
        
        return success
    
    def stop_by_port(self, ports: List[int]) -> bool:
        """Stop processes using specific ports"""
        stopped_any = False
        
        if HAS_PSUTIL:
            return self._stop_by_port_psutil(ports)
        else:
            return self._stop_by_port_basic(ports)
    
    def _stop_by_port_psutil(self, ports: List[int]) -> bool:
        """Stop processes by port using psutil"""
        stopped_any = False
        
        for port in ports:
            try:
                # Find processes using the port
                connections = psutil.net_connections(kind='inet')
                port_procs = []
                
                for conn in connections:
                    if conn.laddr.port == port and conn.pid:
                        try:
                            proc = psutil.Process(conn.pid)
                            port_procs.append(proc)
                        except psutil.NoSuchProcess:
                            continue
                
                if port_procs:
                    print(f"🔴 Found {len(port_procs)} process(es) using port {port}")
                    if self.stop_processes(port_procs, f"port-{port}"):
                        stopped_any = True
                else:
                    if self.verbose:
                        print(f"✅ No processes found using port {port}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Warning: Error checking port {port}: {e}")
        
        return stopped_any
    
    def _stop_by_port_basic(self, ports: List[int]) -> bool:
        """Stop processes by port using basic commands"""
        stopped_any = False
        
        for port in ports:
            try:
                # Use lsof to find processes using the port
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pids = [int(pid.strip()) for pid in result.stdout.strip().split('\n') 
                           if pid.strip().isdigit()]
                    
                    if pids:
                        print(f"🔴 Found {len(pids)} process(es) using port {port}")
                        if self.stop_processes(pids, f"port-{port}"):
                            stopped_any = True
                    else:
                        if self.verbose:
                            print(f"✅ No processes found using port {port}")
                else:
                    if self.verbose:
                        print(f"✅ No processes found using port {port}")
                        
            except FileNotFoundError:
                if self.verbose:
                    print(f"⚠️ lsof not available, skipping port {port}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Warning: Error checking port {port}: {e}")
        
        return stopped_any
    
    def cleanup_system(self) -> bool:
        """Perform system cleanup"""
        success = True
        
        print("🧹 Performing system cleanup...")
        
        # Kill any remaining processes by name patterns
        patterns = [
            'uvicorn',
            'streamlit'
        ]
        
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ['pkill', '-f', pattern],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"✅ Cleaned up remaining {pattern} processes")
                elif self.verbose and result.returncode != 1:  # 1 means no processes found
                    print(f"⚠️ Warning: pkill {pattern} returned {result.returncode}")
            except FileNotFoundError:
                # pkill not available on this system
                if self.verbose:
                    print(f"⚠️ Warning: pkill not available for {pattern} cleanup")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Warning: Error cleaning {pattern}: {e}")
                success = False
        
        return success
    
    def stop_all(self, force: bool = False, cleanup: bool = True) -> bool:
        """Stop all Physics Assistant services"""
        print("=" * 60)
        print("🛑 Physics Assistant System Shutdown")
        print("=" * 60)
        
        overall_success = True
        
        # Method 1: Find and stop by process detection
        print("\n🔍 Scanning for running Physics Assistant processes...")
        processes = self.find_physics_processes()
        
        # Stop system launcher processes first
        if processes['system']:
            if not self.stop_processes(processes['system'], "system launcher"):
                overall_success = False
        
        # Stop UI processes
        if processes['ui']:
            if not self.stop_processes(processes['ui'], "Streamlit UI"):
                overall_success = False
        
        # Stop API processes
        if processes['api']:
            if not self.stop_processes(processes['api'], "FastAPI server"):
                overall_success = False
        
        # Method 2: Stop by known ports
        print("\n🔍 Checking known service ports...")
        physics_ports = [8000, 8501]  # API and UI ports
        if self.stop_by_port(physics_ports):
            print("✅ Stopped processes by port")
        
        # Method 3: Force cleanup if requested
        if force or cleanup:
            print("\n🧹 Performing cleanup...")
            if not self.cleanup_system():
                overall_success = False
        
        # Final verification
        print("\n🔍 Final verification...")
        time.sleep(2)  # Give processes time to fully shutdown
        
        final_processes = self.find_physics_processes()
        total_remaining = sum(len(procs) for procs in final_processes.values())
        
        if total_remaining == 0:
            print("✅ All Physics Assistant services stopped successfully")
            print("\n💡 You can now restart with: python start_system.py")
        else:
            print(f"⚠️ Warning: {total_remaining} process(es) may still be running")
            print("💡 Try running with --force flag: python stop_system.py --force")
            overall_success = False
        
        print("=" * 60)
        return overall_success
    
    def show_status(self):
        """Show current status of Physics Assistant services"""
        print("=" * 60)
        print("📊 Physics Assistant Service Status")
        print("=" * 60)
        
        processes = self.find_physics_processes()
        
        for service_type, procs in processes.items():
            print(f"\n{service_type.upper()} Services:")
            if procs:
                if HAS_PSUTIL:
                    for proc in procs:
                        try:
                            print(f"  🟢 PID {proc.pid}: {proc.name()} - {' '.join(proc.cmdline()[:3])}...")
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            print(f"  🔴 PID {proc.pid}: Process no longer accessible")
                else:
                    for pid in procs:
                        print(f"  🟢 PID {pid}: Running")
            else:
                print(f"  ⚪ No {service_type} processes running")
        
        # Check ports
        print(f"\nPort Status:")
        for port, service in [(8000, "API"), (8501, "UI")]:
            try:
                if HAS_PSUTIL:
                    connections = [c for c in psutil.net_connections(kind='inet') 
                                 if c.laddr.port == port]
                    if connections:
                        print(f"  🟢 Port {port} ({service}): In use")
                    else:
                        print(f"  ⚪ Port {port} ({service}): Available")
                else:
                    # Use lsof as fallback
                    result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        print(f"  🟢 Port {port} ({service}): In use")
                    else:
                        print(f"  ⚪ Port {port} ({service}): Available")
            except Exception as e:
                print(f"  ❓ Port {port} ({service}): Error checking - {e}")
        
        print("=" * 60)

def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stop Physics Assistant services")
    parser.add_argument('--force', action='store_true', 
                       help='Force kill all processes')
    parser.add_argument('--status', action='store_true',
                       help='Show service status instead of stopping')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip cleanup phase')
    
    args = parser.parse_args()
    
    stopper = PhysicsAssistantStopper()
    stopper.verbose = not args.quiet
    
    if args.status:
        stopper.show_status()
        return
    
    try:
        success = stopper.stop_all(
            force=args.force, 
            cleanup=not args.no_cleanup
        )
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during shutdown: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()