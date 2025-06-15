import os
import sys
import signal
import time
import psutil

def close_websocket_connections():
    """Close all WebSocket connections and related Python processes"""
    print("Closing WebSocket connections...")
    
    # Find and terminate Python processes that might be holding WebSocket connections
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for Python processes running the app
            if 'python' in proc.info['name'].lower() and \
               len(proc.info['cmdline']) > 1 and \
               'app.py' in ' '.join(proc.info['cmdline']):
                print(f"Terminating process {proc.info['pid']} - {proc.info['name']}")
                try:
                    # Try graceful termination first
                    proc.terminate()
                    time.sleep(1)
                    if proc.is_running():
                        # Force kill if still running
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    print("All WebSocket connections should be closed now.")
    print("You may need to wait a few minutes for Alpaca to release the connections.")

if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        print("Installing required package: psutil")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    close_websocket_connections()
