#!/usr/bin/env python3
"""
Test script to verify daemon and logging functionality
"""
import os
import sys
import time
import json
from pathlib import Path

def test_logger():
    """Test JSON logger"""
    print("=" * 60)
    print("Testing JSON Logger")
    print("=" * 60)
    
    try:
        from logger import get_logger
        
        logger = get_logger("test", "logs")
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.log_event("test_event", {"test": True, "number": 42})
        logger.log_performance("test_operation", 0.123)
        
        # Check if log file was created
        log_file = Path("logs/test.jsonl")
        if log_file.exists():
            print(f"✅ Log file created: {log_file}")
            
            # Verify JSON format
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"✅ Log file has {len(lines)} entries")
                
                for i, line in enumerate(lines[-3:], 1):  # Check last 3 entries
                    try:
                        entry = json.loads(line.strip())
                        print(f"✅ Entry {i} is valid JSON: {entry.get('level', 'N/A')} - {entry.get('message', 'N/A')[:50]}")
                    except json.JSONDecodeError as e:
                        print(f"❌ Entry {i} is NOT valid JSON: {e}")
                        return False
        else:
            print(f"❌ Log file not created: {log_file}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_daemon_imports():
    """Test daemon module imports"""
    print("\n" + "=" * 60)
    print("Testing Daemon Imports")
    print("=" * 60)
    
    try:
        import daemon_interactive
        print("✅ daemon_interactive module imported successfully")
        
        # Check if watchdog is available
        if daemon_interactive.WATCHDOG_AVAILABLE:
            print("✅ Watchdog is available (auto-reload will work)")
        else:
            print("⚠️  Watchdog not available (auto-reload disabled)")
            print("   Install with: pip install watchdog")
        
        # Test daemon class
        daemon = daemon_interactive.InteractiveDaemon()
        print("✅ InteractiveDaemon class instantiated")
        
        # Test status
        status = daemon.status()
        print(f"✅ Status check works: running={status['running']}")
        
        return True
    except Exception as e:
        print(f"❌ Daemon import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_logging():
    """Test interactive filters logging integration"""
    print("\n" + "=" * 60)
    print("Testing Interactive Filters Logging")
    print("=" * 60)
    
    try:
        import interactive_filters
        
        # Check if logger is available
        if hasattr(interactive_filters, 'LOGGER_AVAILABLE'):
            if interactive_filters.LOGGER_AVAILABLE:
                print("✅ Logger is available in interactive_filters")
            else:
                print("⚠️  Logger not available (using dummy logger)")
        
        # Check if InteractiveFilterViewer has logger
        viewer = interactive_filters.InteractiveFilterViewer()
        if hasattr(viewer, 'logger'):
            print("✅ InteractiveFilterViewer has logger attribute")
        else:
            print("❌ InteractiveFilterViewer missing logger attribute")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Interactive filters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_update_checker_logging():
    """Test update checker logging integration"""
    print("\n" + "=" * 60)
    print("Testing Update Checker Logging")
    print("=" * 60)
    
    try:
        from update_checker import UpdateChecker
        
        checker = UpdateChecker()
        if hasattr(checker, 'logger'):
            print("✅ UpdateChecker has logger attribute")
            checker.logger.info("Test log from update checker")
        else:
            print("❌ UpdateChecker missing logger attribute")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Update checker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_log_files():
    """Check all log files"""
    print("\n" + "=" * 60)
    print("Checking Log Files")
    print("=" * 60)
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print(f"⚠️  Logs directory doesn't exist: {logs_dir}")
        return False
    
    log_files = list(logs_dir.glob("*.jsonl"))
    if log_files:
        print(f"✅ Found {len(log_files)} log file(s):")
        for log_file in log_files:
            size = log_file.stat().st_size
            with open(log_file, 'r') as f:
                lines = len(f.readlines())
            print(f"   - {log_file.name}: {lines} entries, {size} bytes")
    else:
        print("⚠️  No log files found (may be normal if nothing has run yet)")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("WesWorld FX - Daemon & Logging Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Logger", test_logger()))
    results.append(("Daemon Imports", test_daemon_imports()))
    results.append(("Interactive Logging", test_interactive_logging()))
    results.append(("Update Checker Logging", test_update_checker_logging()))
    results.append(("Log Files", check_log_files()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nDaemon and logging system is ready!")
        print("\nTo start daemon: make interactive-daemon")
        print("To check status: make interactive-daemon-status")
        print("To view logs: make interactive-daemon-logs-json")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

