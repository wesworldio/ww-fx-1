"""
Auto-update checker for WesWorld FX
Checks GitHub for updates and can pull/reload if needed
"""
import json
import os
import subprocess
import sys
import time
from typing import Optional, Tuple, Dict
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
try:
    from logger import get_logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    def get_logger(*args, **kwargs):
        class DummyLogger:
            def debug(self, *args, **kwargs): pass
            def info(self, *args, **kwargs): print(f"[INFO] {args[0] if args else ''}")
            def warning(self, *args, **kwargs): print(f"[WARNING] {args[0] if args else ''}")
            def error(self, *args, **kwargs): print(f"[ERROR] {args[0] if args else ''}", file=sys.stderr)
            def exception(self, *args, **kwargs): pass
            def log_event(self, *args, **kwargs): pass
        return DummyLogger()


class UpdateChecker:
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.repo_url = 'https://github.com/wesworldio/ww-fx-1'
        self.api_base = 'https://api.github.com/repos/wesworldio/ww-fx-1'
        self.last_check_time = 0
        self.update_config = self.load_update_config()
        self.logger = get_logger("update_checker", "logs")
        
    def load_update_config(self) -> Dict:
        """Load update configuration from config.json"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    update_config = config.get('updates', {})
                    # Set defaults
                    return {
                        'enabled': update_config.get('enabled', True),
                        'branch': update_config.get('branch', 'main'),
                        'check_interval': update_config.get('check_interval', 300),  # 5 minutes
                        'auto_pull': update_config.get('auto_pull', False),  # Manual by default
                        'last_commit': update_config.get('last_commit', None),
                        'last_check': update_config.get('last_check', 0)
                    }
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Default config
        return {
            'enabled': True,
            'branch': 'main',
            'check_interval': 300,
            'auto_pull': False,
            'last_commit': None,
            'last_check': 0
        }
    
    def save_update_config(self, update_config: Dict):
        """Save update configuration to config.json"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                config = {}
        else:
            config = {}
        
        config['updates'] = update_config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save update config: {e}")
    
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None
    
    def get_latest_commit(self, branch: str = 'main') -> Optional[Tuple[str, str]]:
        """Get latest commit hash and message from GitHub API
        
        Returns:
            Tuple of (commit_hash, commit_message) or None if error
        """
        if not self.update_config['enabled']:
            return None
            
        try:
            url = f"{self.api_base}/commits/{branch}"
            request = Request(url)
            request.add_header('Accept', 'application/vnd.github.v3+json')
            request.add_header('User-Agent', 'WesWorld-FX-UpdateChecker/1.0')
            
            with urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())
                commit_hash = data.get('sha', '')[:7]  # Short hash
                commit_message = data.get('commit', {}).get('message', '').split('\n')[0]
                return (commit_hash, commit_message)
        except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as e:
            self.logger.error(f"Update check failed: {e}", exception=str(e))
            print(f"Update check failed: {e}")
            return None
    
    def check_for_updates(self, force: bool = False) -> Optional[Dict]:
        """Check if updates are available
        
        Args:
            force: Force check even if within check_interval
            
        Returns:
            Dict with update info if available, None otherwise
        """
        if not self.update_config['enabled']:
            return None
        
        current_time = time.time()
        check_interval = self.update_config['check_interval']
        
        # Don't check if within interval (unless forced)
        if not force and (current_time - self.update_config['last_check']) < check_interval:
            return None
        
        branch = self.update_config['branch']
        latest = self.get_latest_commit(branch)
        
        if latest is None:
            return None
        
        latest_hash, latest_message = latest
        current_hash = self.get_current_commit()
        
        # Update last check time
        self.update_config['last_check'] = current_time
        self.save_update_config(self.update_config)
        
        # If we don't have a stored commit, store current one
        if self.update_config['last_commit'] is None:
            # Prefer current git hash, fallback to latest from GitHub
            if current_hash:
                self.update_config['last_commit'] = current_hash[:7]
            elif latest_hash:
                self.update_config['last_commit'] = latest_hash
            if self.update_config['last_commit']:
                self.save_update_config(self.update_config)
            return None
        
        # Check if update available
        stored_hash = self.update_config['last_commit']
        if latest_hash != stored_hash:
            return {
                'available': True,
                'current': stored_hash,
                'latest': latest_hash,
                'message': latest_message,
                'branch': branch
            }
        
        return {'available': False}
    
    def pull_updates(self) -> Tuple[bool, str]:
        """Pull latest updates from git
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            repo_dir = os.path.dirname(os.path.abspath(__file__))
            branch = self.update_config['branch']
            
            # Fetch latest
            fetch_result = subprocess.run(
                ['git', 'fetch', 'origin', branch],
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=30
            )
            
            if fetch_result.returncode != 0:
                self.logger.error("Git fetch failed", error=fetch_result.stderr)
                return (False, f"Git fetch failed: {fetch_result.stderr}")
            
            # Pull latest
            pull_result = subprocess.run(
                ['git', 'pull', 'origin', branch],
                capture_output=True,
                text=True,
                cwd=repo_dir,
                timeout=30
            )
            
            if pull_result.returncode != 0:
                self.logger.error("Git pull failed", error=pull_result.stderr)
                return (False, f"Git pull failed: {pull_result.stderr}")
            
            # Update stored commit
            latest = self.get_latest_commit(branch)
            if latest:
                self.update_config['last_commit'] = latest[0]
                self.save_update_config(self.update_config)
            
            self.logger.log_event("update_pulled", {
                "branch": branch,
                "commit": latest[0] if latest else None
            })
            return (True, "Updates pulled successfully")
            
        except subprocess.TimeoutExpired:
            return (False, "Update operation timed out")
        except Exception as e:
            return (False, f"Update error: {str(e)}")
    
    def should_reload(self) -> bool:
        """Check if application should reload after update"""
        if not self.update_config.get('auto_pull', False):
            return False
        
        update_info = self.check_for_updates(force=True)
        if update_info and update_info.get('available'):
            success, message = self.pull_updates()
            if success:
                return True
        
        return False


def main():
    """Test the update checker"""
    checker = UpdateChecker()
    print("Update Checker Test")
    print("=" * 50)
    print(f"Enabled: {checker.update_config['enabled']}")
    print(f"Branch: {checker.update_config['branch']}")
    print(f"Check Interval: {checker.update_config['check_interval']}s")
    print(f"Auto Pull: {checker.update_config['auto_pull']}")
    print()
    
    current = checker.get_current_commit()
    print(f"Current commit: {current[:7] if current else 'Unknown'}")
    
    print("\nChecking for updates...")
    update_info = checker.check_for_updates(force=True)
    
    if update_info:
        if update_info.get('available'):
            print(f"Update available!")
            print(f"  Current: {update_info['current']}")
            print(f"  Latest: {update_info['latest']}")
            print(f"  Message: {update_info['message']}")
        else:
            print("No updates available")
    else:
        print("Could not check for updates")


if __name__ == '__main__':
    main()

