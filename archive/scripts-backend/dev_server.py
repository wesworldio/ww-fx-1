#!/usr/bin/env python3
"""
Development server - runs web server and WASM watcher together.
"""
import os
import sys
import subprocess
import signal
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = "python3.11"

processes = []


def cleanup(signum=None, frame=None):
    """Clean up all child processes"""
    print("\n\n👋 Shutting down development services...")
    
    for proc in processes:
        if proc.poll() is None:  # Process still running
            print(f"   Stopping PID {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    
    # Also stop any daemons
    try:
        subprocess.run(["make", "dev-stop"], cwd=PROJECT_ROOT, 
                      capture_output=True, timeout=5)
    except:
        pass
    
    print("✅ All services stopped")
    sys.exit(0)


def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    print("🚀 Starting development mode...")
    print("   - Web server with hot reload")
    print("   - WASM watcher with auto-rebuild")
    print("\nPress Ctrl+C to stop all services\n")
    
    # Initial WASM build
    print("📦 Building WASM module initially...")
    build_proc = subprocess.run(
        ["make", "build"],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    if build_proc.returncode != 0:
        print("⚠️  Initial WASM build failed, continuing anyway...")
    print()
    
    # Start WASM watcher (as daemon)
    print("🔨 Starting WASM watcher...")
    wasm_proc = subprocess.Popen(
        [PYTHON, "scripts/watch_wasm.py"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes.append(wasm_proc)
    print(f"   ✅ WASM watcher started (PID: {wasm_proc.pid})")
    
    # Wait a moment for watcher to initialize
    time.sleep(2)
    
    # Start web server
    print("🌐 Starting web server...")
    web_proc = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "web_server:app", 
         "--host", "0.0.0.0", "--port", "9000", "--reload"],
        cwd=PROJECT_ROOT / "src",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes.append(web_proc)
    print(f"   ✅ Web server started (PID: {web_proc.pid})")
    print("\n" + "="*60)
    print("✅ Development mode active!")
    print("   Web server: http://localhost:9000")
    print("   WASM watcher: monitoring for changes")
    print("="*60 + "\n")
    
    # Monitor processes
    try:
        while True:
            # Check if processes are still running
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    # Process died
                    output = proc.stdout.read() if proc.stdout else ""
                    print(f"\n⚠️  Process {proc.pid} exited with code {proc.returncode}")
                    if output:
                        print("Output:")
                        print(output)
                    cleanup()
                    return
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()

