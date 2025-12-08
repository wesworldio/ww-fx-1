#!/usr/bin/env python3
"""
WASM file watcher - rebuilds WASM module when source files change.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️  watchdog not available. Install with: pip install watchdog")
    print("Falling back to polling mode...")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
WASM_DIR = PROJECT_ROOT / "wasm"
BUILD_SCRIPT = WASM_DIR / "build.sh"

# Files to watch
WATCH_PATTERNS = [
    "*.cpp",
    "*.h",
    "*.hpp",
    "CMakeLists.txt",
    "build.sh"
]

WATCH_DIRS = [
    WASM_DIR / "src",
    WASM_DIR / "include",
    WASM_DIR
]


class WASMBuilder:
    """Handles WASM builds"""
    
    def __init__(self):
        self.last_build_time = 0
        self.build_cooldown = 2  # Minimum seconds between builds
        
    def build(self):
        """Build the WASM module"""
        current_time = time.time()
        if current_time - self.last_build_time < self.build_cooldown:
            return  # Skip if too soon after last build
            
        self.last_build_time = current_time
        
        print(f"\n{'='*60}")
        print(f"🔨 Rebuilding WASM module...")
        print(f"{'='*60}\n")
        
        try:
            # Make build script executable
            os.chmod(BUILD_SCRIPT, 0o755)
            
            # Run build script
            result = subprocess.run(
                [str(BUILD_SCRIPT)],
                cwd=str(WASM_DIR),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n✅ WASM build successful!")
            else:
                print(f"\n❌ WASM build failed (exit code: {result.returncode})")
                
        except Exception as e:
            print(f"\n❌ Error building WASM: {e}")
        
        print(f"{'='*60}\n")


class WASMFileHandler(FileSystemEventHandler):
    """Handles file system events for WASM files"""
    
    def __init__(self, builder):
        self.builder = builder
        self.debounce_time = 0.5
        self.last_event_time = 0
        
    def should_trigger(self, path):
        """Check if we should trigger a build"""
        # Check if file matches watch patterns
        path_obj = Path(path)
        if not any(path_obj.match(pattern) for pattern in WATCH_PATTERNS):
            return False
            
        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_event_time < self.debounce_time:
            return False
            
        self.last_event_time = current_time
        return True
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if self.should_trigger(event.src_path):
            self.builder.build()
            
    def on_created(self, event):
        if event.is_directory:
            return
            
        if self.should_trigger(event.src_path):
            self.builder.build()


def watch_polling(builder, interval=1.0):
    """Fallback polling mode if watchdog not available"""
    print("📡 Using polling mode (install watchdog for better performance)")
    
    last_mtimes = {}
    
    while True:
        try:
            changed = False
            
            for watch_dir in WATCH_DIRS:
                if not watch_dir.exists():
                    continue
                    
                for pattern in WATCH_PATTERNS:
                    for file_path in watch_dir.rglob(pattern):
                        if file_path.is_file():
                            mtime = file_path.stat().st_mtime
                            file_key = str(file_path)
                            
                            if file_key in last_mtimes:
                                if mtime > last_mtimes[file_key]:
                                    changed = True
                                    break
                            else:
                                last_mtimes[file_key] = mtime
                                
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            
            if changed:
                builder.build()
                # Update all mtimes after build
                for watch_dir in WATCH_DIRS:
                    if watch_dir.exists():
                        for pattern in WATCH_PATTERNS:
                            for file_path in watch_dir.rglob(pattern):
                                if file_path.is_file():
                                    last_mtimes[str(file_path)] = file_path.stat().st_mtime
                                    
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n👋 Stopping WASM watcher...")
            break
        except Exception as e:
            print(f"⚠️  Error in polling: {e}")
            time.sleep(interval)


def main():
    """Main entry point"""
    print("🚀 Starting WASM file watcher...")
    print(f"📁 Watching: {WASM_DIR}")
    print(f"🔍 Patterns: {', '.join(WATCH_PATTERNS)}")
    print("\nPress Ctrl+C to stop\n")
    
    # Initial build
    builder = WASMBuilder()
    builder.build()
    
    if WATCHDOG_AVAILABLE:
        # Use watchdog for efficient file watching
        event_handler = WASMFileHandler(builder)
        observer = Observer()
        
        for watch_dir in WATCH_DIRS:
            if watch_dir.exists():
                observer.schedule(event_handler, str(watch_dir), recursive=True)
        
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Stopping WASM watcher...")
            observer.stop()
        
        observer.join()
    else:
        # Fallback to polling
        watch_polling(builder)


if __name__ == "__main__":
    main()

