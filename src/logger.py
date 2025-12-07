"""
JSON logging system for WesWorld FX
All logs are written to logs/ directory as JSON files
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path


class JSONLogger:
    """Logger that writes structured JSON logs to files"""
    
    def __init__(self, component: str = "main", logs_dir: str = "logs"):
        self.component = component
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.log_file = self.logs_dir / f"{component}.jsonl"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _write_log(self, level: str, message: str, **kwargs):
        """Write a log entry to JSON file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "component": self.component,
            "level": level,
            "message": message,
            **kwargs
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Fallback to stderr if logging fails
            print(f"Logging error: {e}", file=__import__('sys').stderr)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._write_log("DEBUG", message, **kwargs)
        print(f"[DEBUG] {message}")
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._write_log("INFO", message, **kwargs)
        print(f"[INFO] {message}")
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._write_log("WARNING", message, **kwargs)
        print(f"[WARNING] {message}")
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._write_log("ERROR", message, **kwargs)
        print(f"[ERROR] {message}", file=__import__('sys').stderr)
    
    def exception(self, message: str, exc_info: Optional[Any] = None, **kwargs):
        """Log exception with traceback"""
        import traceback
        tb = traceback.format_exc() if exc_info is None else str(exc_info)
        self._write_log("ERROR", message, exception=tb, **kwargs)
        print(f"[ERROR] {message}", file=__import__('sys').stderr)
        print(tb, file=__import__('sys').stderr)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured event"""
        self._write_log("EVENT", f"Event: {event_type}", event_type=event_type, **data)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        self._write_log("PERFORMANCE", f"{operation} took {duration:.3f}s", 
                       operation=operation, duration=duration, **kwargs)


# Global loggers for different components
_loggers = {}

def get_logger(component: str = "main", logs_dir: str = "logs") -> JSONLogger:
    """Get or create a logger for a component"""
    if component not in _loggers:
        _loggers[component] = JSONLogger(component, logs_dir)
    return _loggers[component]

