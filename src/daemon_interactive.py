#!/usr/bin/env python3
"""
Daemon mode for interactive filters with auto-reload on file changes
"""
import os
import sys
import time
import signal
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from logger import get_logger

# Watchdog for file monitoring
WATCHDOG_AVAILABLE = False
Observer = None
FileSystemEventHandler = None
FileModifiedEvent = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    # Create dummy classes if watchdog not available
    class FileSystemEventHandler:
        pass
    class FileModifiedEvent:
        pass
    print("Warning: watchdog not available. Install with: pip install watchdog")
    print("Auto-reload will not work without watchdog.")


class InteractiveReloadHandler(FileSystemEventHandler):
    """Handle file system events for auto-reload"""
    
    def __init__(self, daemon):
        self.daemon = daemon
        self.logger = get_logger("daemon")
        self.last_reload = 0
        self.reload_cooldown = 2.0  # Prevent rapid reloads
        
        # Files to watch
        self.watch_patterns = [
            '*.py',
            '*.json',
        ]
        
        # Files to ignore
        self.ignore_patterns = [
            '__pycache__',
            '.git',
            '*.pyc',
            'logs/',
            'recordings/',
            '.DS_Store',
        ]
    
    def should_reload(self, file_path: str) -> bool:
        """Check if file change should trigger reload"""
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_reload < self.reload_cooldown:
            return False
        
        # Check if file matches watch patterns
        path = Path(file_path)
        if not any(path.match(pattern) for pattern in self.watch_patterns):
            return False
        
        # Check if file should be ignored
        if any(ignore in str(path) for ignore in self.ignore_patterns):
            return False
        
        # Don't reload on log file changes
        if 'logs/' in str(path):
            return False
        
        return True
    
    def on_modified(self, event):
        """Handle file modification events"""
        if isinstance(event, FileModifiedEvent):
            file_path = event.src_path
            
            if self.should_reload(file_path):
                self.logger.info(f"File changed: {file_path}, triggering reload")
                self.last_reload = time.time()
                self.daemon.reload_process()


class InteractiveDaemon:
    """Daemon manager for interactive filters"""
    
    def __init__(self, pid_file: str = "/tmp/ww_fx_interactive.pid", 
                 log_file: str = "/tmp/ww_fx_interactive.log",
                 logs_dir: str = "logs"):
        self.pid_file = Path(pid_file)
        self.log_file = Path(log_file)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger("daemon", str(self.logs_dir))
        self.process: Optional[subprocess.Popen] = None
        self.observer: Optional[Observer] = None
        self.running = False
        self.reload_count = 0
        
        # Get script directory
        self.script_dir = Path(__file__).parent.absolute()
        self.interactive_script = self.script_dir / "interactive_filters.py"
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self, width: int = 1280, height: int = 720, fps: int = 30):
        """Start the daemon"""
        if self.is_running():
            self.logger.warning("Daemon is already running")
            return False
        
        self.logger.info("Starting interactive filters daemon")
        self.logger.log_event("daemon_start", {
            "width": width,
            "height": height,
            "fps": fps,
            "pid_file": str(self.pid_file),
            "log_file": str(self.log_file)
        })
        
        # Start the process
        self._start_process(width, height, fps)
        
        # Start file watcher
        if WATCHDOG_AVAILABLE:
            self._start_file_watcher()
        
        # Save PID
        self.pid_file.write_text(str(os.getpid()))
        self.running = True
        
        self.logger.info(f"Daemon started (PID: {os.getpid()})")
        return True
    
    def _start_process(self, width: int, height: int, fps: int):
        """Start the interactive filters process"""
        cmd = [
            sys.executable,
            str(self.interactive_script),
            "--width", str(width),
            "--height", str(height),
            "--fps", str(fps)
        ]
        
        self.logger.info(f"Starting process: {' '.join(cmd)}")
        
        # Redirect output to log file
        with open(self.log_file, 'a') as log_f:
            log_f.write(f"\n{'='*50}\n")
            log_f.write(f"Process started at {datetime.now().isoformat()}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"{'='*50}\n")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=open(self.log_file, 'a'),
            stderr=subprocess.STDOUT,
            cwd=str(self.script_dir),
            env=os.environ.copy()
        )
        
        self.logger.log_event("process_started", {
            "pid": self.process.pid,
            "command": ' '.join(cmd)
        })
    
    def _start_file_watcher(self):
        """Start watching for file changes"""
        if not WATCHDOG_AVAILABLE:
            return
        
        event_handler = InteractiveReloadHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.script_dir), recursive=True)
        self.observer.start()
        
        self.logger.info(f"File watcher started on {self.script_dir}")
    
    def reload_process(self):
        """Reload the process"""
        if not self.process:
            return
        
        self.reload_count += 1
        self.logger.info(f"Reloading process (reload #{self.reload_count})")
        self.logger.log_event("process_reload", {
            "reload_count": self.reload_count,
            "old_pid": self.process.pid
        })
        
        # Stop current process
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.logger.warning("Process didn't terminate, killing...")
            self.process.kill()
            self.process.wait()
        except Exception as e:
            self.logger.error(f"Error stopping process: {e}")
        
        # Wait a moment
        time.sleep(1)
        
        # Get config for dimensions
        config_path = self.script_dir / "config.json"
        width, height, fps = 1280, 720, 30
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Could read from config if stored
            except:
                pass
        
        # Start new process
        self._start_process(width, height, fps)
    
    def stop(self):
        """Stop the daemon"""
        if not self.running:
            return
        
        self.logger.info("Stopping daemon...")
        
        # Stop file watcher
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        # Stop process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Process didn't terminate, killing...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                self.logger.error(f"Error stopping process: {e}")
            self.process = None
        
        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()
        
        self.running = False
        self.logger.info("Daemon stopped")
        self.logger.log_event("daemon_stopped", {
            "total_reloads": self.reload_count
        })
    
    def is_running(self) -> bool:
        """Check if daemon is running"""
        if not self.pid_file.exists():
            return False
        
        try:
            pid = int(self.pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (ValueError, OSError, ProcessLookupError):
            # PID file exists but process doesn't
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
    
    def status(self) -> Dict:
        """Get daemon status"""
        running = self.is_running()
        status = {
            "running": running,
            "pid_file": str(self.pid_file),
            "log_file": str(self.log_file),
            "logs_dir": str(self.logs_dir),
        }
        
        if running:
            try:
                pid = int(self.pid_file.read_text().strip())
                status["pid"] = pid
            except:
                pass
        
        if self.process:
            status["process_pid"] = self.process.pid
            status["reload_count"] = self.reload_count
        
        return status
    
    def run_forever(self):
        """Run daemon in foreground (for testing)"""
        try:
            while self.running:
                if self.process:
                    # Check if process died
                    if self.process.poll() is not None:
                        self.logger.warning("Process died, restarting...")
                        self.reload_process()
                
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Interrupted, shutting down...")
            self.stop()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WesWorld FX Interactive Daemon')
    parser.add_argument('action', choices=['start', 'stop', 'status', 'restart', 'run'],
                       help='Action to perform')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--fps', type=int, default=30, help='FPS')
    parser.add_argument('--pid-file', default='/tmp/ww_fx_interactive.pid',
                       help='PID file path')
    parser.add_argument('--log-file', default='/tmp/ww_fx_interactive.log',
                       help='Log file path')
    parser.add_argument('--logs-dir', default='logs',
                       help='JSON logs directory')
    
    args = parser.parse_args()
    
    daemon = InteractiveDaemon(
        pid_file=args.pid_file,
        log_file=args.log_file,
        logs_dir=args.logs_dir
    )
    
    if args.action == 'start':
        if daemon.start(args.width, args.height, args.fps):
            print(f"✅ Daemon started (PID: {os.getpid()})")
            print(f"   PID file: {args.pid_file}")
            print(f"   Log file: {args.log_file}")
            print(f"   JSON logs: {args.logs_dir}/")
            if WATCHDOG_AVAILABLE:
                print("   Auto-reload: Enabled")
            else:
                print("   Auto-reload: Disabled (install watchdog)")
            daemon.run_forever()
        else:
            print("❌ Failed to start daemon (already running?)")
            sys.exit(1)
    
    elif args.action == 'stop':
        if daemon.is_running():
            daemon.stop()
            print("✅ Daemon stopped")
        else:
            print("⚠️  Daemon is not running")
    
    elif args.action == 'status':
        status = daemon.status()
        if status['running']:
            print("✅ Daemon is running")
            print(f"   PID: {status.get('pid', 'Unknown')}")
            print(f"   Reloads: {status.get('reload_count', 0)}")
        else:
            print("❌ Daemon is not running")
        print(f"   PID file: {status['pid_file']}")
        print(f"   Log file: {status['log_file']}")
        print(f"   JSON logs: {status['logs_dir']}/")
    
    elif args.action == 'restart':
        if daemon.is_running():
            daemon.stop()
            time.sleep(1)
        daemon.start(args.width, args.height, args.fps)
        print("✅ Daemon restarted")
        daemon.run_forever()
    
    elif args.action == 'run':
        # Run in foreground for testing
        daemon.start(args.width, args.height, args.fps)
        try:
            daemon.run_forever()
        except KeyboardInterrupt:
            daemon.stop()


if __name__ == '__main__':
    from datetime import datetime
    main()

