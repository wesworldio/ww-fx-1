import cv2
import numpy as np
import sys
import json
import os
import time
import threading
from datetime import datetime
from face_filters import FaceFilter
from typing import Tuple, Optional, List, Dict
try:
    from update_checker import UpdateChecker
    UPDATE_CHECKER_AVAILABLE = True
except ImportError:
    UPDATE_CHECKER_AVAILABLE = False
    UpdateChecker = None
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
            def log_performance(self, *args, **kwargs): pass
        return DummyLogger()


class InteractiveFilterViewer:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.filter_app = None
        self.auto_advance = False
        self.last_advance_time = 0
        self.window_name = 'WesWorld FX'
        self.display_width = width
        self.display_height = height
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        self.camera_index = self.load_camera_index()
        self.advance_interval = self.load_advance_interval()
        self.show_ui = True
        self.frame_count = 0
        self.number_buffer = ''
        self.last_number_input_time = 0
        self.number_input_timeout = 1.0
        
        # Update checker
        self.update_checker = None
        self.update_available = False
        self.update_info = None
        self.last_update_check = 0
        self.update_check_interval = 300  # 5 minutes
        
        # Favorites/presets
        self.favorites = self.load_favorites()
        self.recording = False
        self.video_writer = None
        self.recording_path = None
        
        # Performance optimization
        self.frame_cache = None
        self.cache_valid = False
        
        # Camera selection
        self.available_cameras = []
        self.camera_names = {}
        
        # Theme support
        self.theme_name = self.load_config().get('theme', 'wesworld')
        self.available_themes = ['wesworld', 'dropout', 'default']
        self.theme = self.load_theme()
        
        # Filter categories (matching web UI)
        self.filter_categories = {
            'DROPOUT': [],  # Face masks are discovered dynamically
            'Distortion': [
                'bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror',
                'twirl', 'ripple', 'sphere', 'tunnel', 'water_ripple', 'radial_blur',
                'cylinder', 'barrel', 'pincushion', 'whirlpool', 'radial_zoom',
                'concave', 'convex', 'spiral', 'radial_stretch', 'radial_compress',
                'vertical_wave', 'horizontal_wave', 'skew_horizontal', 'skew_vertical',
                'rotate_zoom', 'radial_wave', 'zoom_in', 'zoom_out', 'fast_zoom_in',
                'fast_zoom_out', 'shake', 'pulse', 'spiral_zoom', 'extreme_closeup',
                'puzzle', 'rotate', 'rotate_45', 'rotate_90', 'flip_horizontal',
                'flip_vertical', 'flip_both', 'quad_mirror', 'tile', 'radial_tile',
                'zoom_blur', 'melt', 'kaleidoscope', 'glitch', 'double_vision'
            ],
            'Color & Style': [
                'black_white', 'sepia', 'vintage', 'neon_glow', 'pixelate', 'blur',
                'sharpen', 'emboss', 'red_tint', 'blue_tint', 'green_tint', 'rainbow',
                'negative', 'posterize', 'sketch', 'cartoon', 'thermal', 'ice', 'ocean',
                'plasma', 'jet', 'turbo', 'inferno', 'magma', 'viridis', 'cool', 'hot',
                'spring', 'summer', 'autumn', 'winter', 'rainbow_shift', 'acid_trip',
                'vhs', 'retro', 'cyberpunk', 'anime', 'glow', 'solarize', 'edge_detect', 'halftone'
            ]
        }
        
        # Build flat filter list from categories
        self.filter_list = [(None, 'None (Original)')]
        for category, filters in self.filter_categories.items():
            for filter_type in filters:
                # Find display name
                display_name = filter_type.replace('_', ' ').title()
                self.filter_list.append((filter_type, display_name))
        
        # Initialize current filter (None/Original is first)
        self.current_filter_index = 0
        self.current_filter = self.filter_list[0][0]
        self.current_filter_name = self.filter_list[0][1]
        
        # Setup quick access filters
        self.filters = {
            '0': self.filter_list[0],  # None (Original)
        }
        # Add number keys for first few filters after None
        for i in range(1, 8):
            if i < len(self.filter_list):
                self.filters[str(i)] = self.filter_list[i]
        
        # Search functionality (works continuously like web UI)
        self.search_query = ''
        self.search_buffer = ''  # For building search query
        self.search_active = False  # Whether we're currently typing search
        self.last_search_key_time = 0
        self.search_timeout = 2.0  # Clear search after 2 seconds of no input
        
        # Logger
        self.logger = get_logger("interactive", "logs")
    
    def display_number_to_index(self, display_num: int) -> Optional[int]:
        if display_num == 0:
            return 1
        elif display_num >= 1 and display_num < len(self.filter_list) - 1:
            return display_num + 1
        return None
    
    def index_to_display_number(self, index: int) -> Optional[str]:
        if index == 0:
            return 'S'
        elif index == 1:
            return '0'
        elif index <= 8:
            return str(index - 1)
        else:
            return str(index - 1)
    
    def load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                pass
        return {}
    
    def save_config(self, config: dict):
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def load_camera_index(self) -> Optional[int]:
        config = self.load_config()
        camera_index = config.get('camera_index')
        if camera_index is not None:
            return int(camera_index)
        return None
    
    def save_camera_index(self, camera_index: int):
        config = self.load_config()
        config['camera_index'] = camera_index
        self.save_config(config)
    
    def load_advance_interval(self) -> float:
        config = self.load_config()
        advance_interval = config.get('advance_interval')
        if advance_interval is not None:
            interval = float(advance_interval)
            if interval > 0:
                return interval
        return 0.3
    
    def load_favorites(self) -> List[str]:
        """Load favorite filter names from config"""
        config = self.load_config()
        favorites = config.get('favorites', [])
        return [f for f in favorites if isinstance(f, str)]
    
    def save_favorites(self):
        """Save favorite filters to config"""
        config = self.load_config()
        config['favorites'] = self.favorites
        self.save_config(config)
    
    def toggle_favorite(self, filter_type: Optional[str]):
        """Toggle favorite status for a filter"""
        if filter_type is None:
            return
        if filter_type in self.favorites:
            self.favorites.remove(filter_type)
            self.logger.log_event("favorite_removed", {"filter": filter_type})
            print(f"Removed {filter_type} from favorites")
        else:
            self.favorites.append(filter_type)
            self.logger.log_event("favorite_added", {"filter": filter_type})
            print(f"Added {filter_type} to favorites")
        self.save_favorites()
    
    def load_theme(self) -> Dict:
        """Load theme configuration matching HTML themes"""
        theme_path = os.path.join(os.path.dirname(__file__), 'themes', f'{self.theme_name}.json')
        
        def hex_to_rgb(hex_str):
            hex_str = str(hex_str).lstrip('#')
            if len(hex_str) == 6:
                return [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]
            return [128, 128, 128]
        
        if os.path.exists(theme_path):
            try:
                with open(theme_path, 'r') as f:
                    theme_data = json.load(f)
                    if 'colors' in theme_data:
                        colors = theme_data['colors']
                        return {
                            'background': hex_to_rgb(colors.get('background', '#000000')),
                            'surface': hex_to_rgb(colors.get('surface', '#1a1a1a')),
                            'surfaceHover': hex_to_rgb(colors.get('surfaceHover', '#2a2a2a')),
                            'text': hex_to_rgb(colors.get('text', '#ffffff')),
                            'textSecondary': hex_to_rgb(colors.get('textSecondary', '#cccccc')),
                            'accent': hex_to_rgb(colors.get('accent', '#5250ef')),
                            'accentHover': hex_to_rgb(colors.get('accentHover', '#6260ff')),
                            'border': hex_to_rgb(colors.get('border', '#333333')),
                            'borderHover': hex_to_rgb(colors.get('borderHover', '#5250ef')),
                            'button': hex_to_rgb(colors.get('button', '#5250ef')),
                            'buttonHover': hex_to_rgb(colors.get('buttonHover', '#6260ff')),
                            'statusConnected': hex_to_rgb(colors.get('statusConnected', '#4caf50')),
                            'statusError': hex_to_rgb(colors.get('statusError', '#f44336')),
                            'groupTitle': hex_to_rgb(colors.get('groupTitle', '#5250ef')),
                            'selectedText': hex_to_rgb(colors.get('selectedText', '#ffffff')),
                            'bg_alpha': 0.95,
                            'recording_color': [0, 255, 0],
                            'update_color': [255, 200, 0]
                        }
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Default themes (matching HTML)
        if self.theme_name == 'dropout':
            return {
                'background': [10, 10, 10],
                'surface': [26, 26, 26],
                'surfaceHover': [42, 42, 42],
                'text': [255, 255, 255],
                'textSecondary': [224, 224, 224],
                'accent': [254, 234, 59],
                'accentHover': [255, 240, 74],
                'border': [51, 51, 51],
                'borderHover': [254, 234, 59],
                'button': [254, 234, 59],
                'buttonHover': [255, 240, 74],
                'statusConnected': [76, 175, 80],
                'statusError': [255, 68, 68],
                'groupTitle': [254, 234, 59],
                'selectedText': [0, 0, 0],
                'bg_alpha': 0.95,
                'recording_color': [0, 255, 0],
                'update_color': [255, 200, 0]
            }
        else:  # wesworld or default
            return {
                'background': [0, 0, 0],
                'surface': [26, 26, 26],
                'surfaceHover': [42, 42, 42],
                'text': [255, 255, 255],
                'textSecondary': [204, 204, 204],
                'accent': [82, 80, 239],
                'accentHover': [98, 96, 255],
                'border': [51, 51, 51],
                'borderHover': [82, 80, 239],
                'button': [82, 80, 239],
                'buttonHover': [90, 174, 255],
                'statusConnected': [76, 175, 80],
                'statusError': [244, 67, 54],
                'groupTitle': [82, 80, 239],
                'selectedText': [255, 255, 255],
                'bg_alpha': 0.95,
                'recording_color': [0, 255, 0],
                'update_color': [255, 200, 0]
            }
    
    def switch_theme(self, theme_name: str):
        """Switch theme"""
        if theme_name in self.available_themes:
            self.theme_name = theme_name
            self.theme = self.load_theme()
            config = self.load_config()
            config['theme'] = theme_name
            self.save_config(config)
            self.logger.log_event("theme_changed", {"theme": theme_name})
            print(f"Theme switched to: {theme_name}")
    
    def get_filters_by_category(self) -> Dict[str, List[Tuple[Optional[str], str]]]:
        """Get filters organized by category"""
        categorized = {}
        for category, filter_types in self.filter_categories.items():
            categorized[category] = []
            for filter_type in filter_types:
                # Find in filter_list
                for ft, name in self.filter_list:
                    if ft == filter_type:
                        categorized[category].append((ft, name))
                        break
        return categorized
    
    def get_filtered_filters(self) -> Optional[List[Tuple[Optional[str], str]]]:
        """Get filtered filters based on search query (matches web UI behavior)"""
        if not self.search_query:
            return None
        
        search_lower = self.search_query.lower().strip()
        if not search_lower:
            return None
        
        filtered = []
        for filter_type, name in self.filter_list:
            # Skip if pinned (pinned items shown separately in web UI)
            if filter_type and filter_type in self.favorites:
                continue
            
            # Match against name, filter type, or formatted name
            name_lower = name.lower()
            type_lower = filter_type.lower() if filter_type else ''
            
            if (search_lower in name_lower or 
                search_lower in type_lower or
                search_lower in name_lower.replace(' ', '_')):
                filtered.append((filter_type, name))
        
        return filtered if filtered else None
    
    def get_available_cameras(self) -> List[Tuple[int, str]]:
        """Get list of available cameras"""
        cameras = []
        for i in range(10):  # Check up to 10 cameras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to get camera name (may not work on all systems)
                name = f"Camera {i}"
                try:
                    # Some systems support getting camera name
                    backend = cap.getBackendName()
                    name = f"Camera {i} ({backend})"
                except:
                    pass
                cameras.append((i, name))
                cap.release()
        return cameras
    
    def start_recording(self, output_path: Optional[str] = None):
        """Start recording video"""
        if self.recording:
            return
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(os.path.dirname(__file__), 'recordings', f'recording_{timestamp}.mp4')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if self.video_writer.isOpened():
            self.recording = True
            self.recording_path = output_path
            self.logger.log_event("recording_started", {"path": output_path})
            print(f"Recording started: {output_path}")
        else:
            self.logger.error("Failed to start recording")
            print("Failed to start recording")
            self.video_writer = None
    
    def stop_recording(self):
        """Stop recording video"""
        if not self.recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.recording = False
        if self.recording_path:
            self.logger.log_event("recording_stopped", {"path": self.recording_path})
            print(f"Recording saved: {self.recording_path}")
            self.recording_path = None
    
    def check_for_updates(self, force: bool = False):
        """Check for updates in background"""
        if not UPDATE_CHECKER_AVAILABLE or not self.update_checker:
            return
        
        current_time = time.time()
        if not force and (current_time - self.last_update_check) < self.update_check_interval:
            return
        
        try:
            update_info = self.update_checker.check_for_updates(force=force)
            if update_info and update_info.get('available'):
                self.update_available = True
                self.update_info = update_info
                self.logger.log_event("update_available", update_info)
                print(f"\n{'='*50}")
                print("UPDATE AVAILABLE!")
                print(f"Current: {update_info.get('current', 'Unknown')}")
                print(f"Latest: {update_info.get('latest', 'Unknown')}")
                print(f"Message: {update_info.get('message', '')}")
                print(f"Press 'U' to pull updates")
                print(f"{'='*50}\n")
            else:
                self.update_available = False
            self.last_update_check = current_time
        except Exception as e:
            self.logger.error(f"Update check error: {e}", exception=str(e))
            print(f"Update check error: {e}")
        
    def __enter__(self):
        self.logger.info("Initializing interactive filter viewer")
        self.logger.log_event("viewer_start", {
            "width": self.width,
            "height": self.height,
            "fps": self.fps
        })
        
        # Initialize update checker
        if UPDATE_CHECKER_AVAILABLE:
            try:
                self.update_checker = UpdateChecker(self.config_path)
                # Check for updates on startup
                threading.Thread(target=self.check_for_updates, args=(True,), daemon=True).start()
                self.logger.info("Update checker initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize update checker: {e}")
        
        self.logger.info("Opening camera...")
        
        # Get available cameras
        self.available_cameras = self.get_available_cameras()
        if self.available_cameras:
            print(f"Found {len(self.available_cameras)} camera(s)")
            for idx, name in self.available_cameras:
                print(f"  [{idx}] {name}")
        
        camera_indices_to_try = []
        if self.camera_index is not None:
            camera_indices_to_try.append(self.camera_index)
            print(f"Trying saved camera index {self.camera_index} first...")
        camera_indices_to_try.extend([0, 1, 2])
        camera_indices_to_try = list(dict.fromkeys(camera_indices_to_try))
        
        self.cap = None
        
        for idx in camera_indices_to_try:
            if idx != self.camera_index or self.camera_index is None:
                print(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                import time
                time.sleep(0.5)
                
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cap = cap
                    if idx != self.camera_index:
                        self.save_camera_index(idx)
                        self.logger.info(f"Camera {idx} initialized successfully! (saved to config)")
                        print(f"Camera {idx} initialized successfully! (saved to config)")
                    else:
                        self.logger.info(f"Camera {idx} initialized successfully! (using saved config)")
                        print(f"Camera {idx} initialized successfully! (using saved config)")
                    resolution = f"{int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                    print(f"Resolution: {resolution}")
                    self.logger.log_event("camera_initialized", {
                        "camera_index": idx,
                        "resolution": resolution
                    })
                    break
                else:
                    cap.release()
                    if idx == self.camera_index:
                        print(f"Saved camera {idx} no longer works, trying others...")
                        self.camera_index = None
                    else:
                        print(f"Camera {idx} opened but could not read frames")
            else:
                if idx == self.camera_index:
                    print(f"Saved camera {idx} no longer available, trying others...")
                    self.camera_index = None
                else:
                    print(f"Could not open camera {idx}")
        
        if self.cap is None:
            print("\n" + "="*50)
            print("ERROR: Could not access camera")
            print("="*50)
            print("\nTroubleshooting steps:")
            print("  1. Check camera permissions:")
            print("     System Settings → Privacy & Security → Camera")
            print("     Make sure Terminal/Python has camera access")
            print("  2. Close other applications using the camera:")
            print("     - Photo Booth")
            print("     - Zoom, Teams, etc.")
            print("     - Other video apps")
            print("  3. Try restarting Terminal/Python")
            print("  4. Check if camera hardware is working:")
            print("     Open Photo Booth to test")
            print("="*50)
            raise RuntimeError("Could not access camera")
        
        self.filter_app = FaceFilter(width=self.width, height=self.height, fps=self.fps)
        self.filter_app.__enter__()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info("Shutting down interactive filter viewer")
        self.stop_recording()
        if self.filter_app:
            self.filter_app.__exit__(exc_type, exc_val, exc_tb)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.log_event("viewer_stopped", {
            "frame_count": self.frame_count
        })
    
    def apply_filter(self, frame: np.ndarray, filter_type: Optional[str]) -> np.ndarray:
        if filter_type is None:
            return frame
        
        animated_filters = {
            'extreme_closeup', 'puzzle', 'fast_zoom_in', 'fast_zoom_out', 'shake', 'pulse', 'spiral_zoom'
        }
        
        full_image_filters = {
            'bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror',
            'twirl', 'ripple', 'sphere', 'tunnel', 'water_ripple', 'radial_blur',
            'cylinder', 'barrel', 'pincushion', 'whirlpool', 'radial_zoom',
            'concave', 'convex', 'spiral', 'radial_stretch', 'radial_compress',
            'vertical_wave', 'horizontal_wave', 'skew_horizontal', 'skew_vertical',
            'rotate_zoom', 'radial_wave', 'zoom_in', 'zoom_out', 'rotate',
            'rotate_45', 'rotate_90', 'flip_horizontal', 'flip_vertical',
            'flip_both', 'quad_mirror', 'tile', 'radial_tile',
            'zoom_blur', 'melt', 'kaleidoscope', 'glitch', 'double_vision',
            'black_white', 'sepia', 'vintage', 'negative', 'posterize', 'sketch',
            'cartoon', 'anime', 'thermal', 'ice', 'ocean', 'plasma', 'jet',
            'turbo', 'inferno', 'magma', 'viridis', 'cool', 'hot', 'spring',
            'summer', 'autumn', 'winter', 'rainbow', 'rainbow_shift', 'acid_trip',
            'vhs', 'retro', 'cyberpunk', 'glow', 'solarize', 'edge_detect',
            'halftone', 'red_tint', 'blue_tint', 'green_tint', 'neon_glow',
            'pixelate', 'blur', 'sharpen', 'emboss'
        }
        
        # Face mask filters are handled dynamically via apply_face_mask_from_asset
        # Check if this is a face mask filter by checking if it contains 'face_mask'
        try:
            if 'face_mask' in filter_type:
                # Parse filter name to extract folder and mask name
                parts = filter_type.split('_')
                if len(parts) >= 3:
                    folder = parts[0]
                    mask_name = parts[-1]
                    if folder == 'dropout':
                        asset_dir = 'assets/dropout/face_mask'
                    elif folder == 'assets':
                        asset_dir = 'assets/face_mask'
                    else:
                        asset_dir = f'assets/{folder}/face_mask'
                    faces = self.filter_app.detect_all_faces(frame)
                    if faces:
                        for face in faces:
                            frame = self.filter_app.apply_face_mask_from_asset(frame.copy(), face, mask_name, asset_dir=asset_dir)
                return frame
            elif filter_type in animated_filters:
                dummy_face = (0, 0, frame.shape[1], frame.shape[0])
                filter_method = getattr(self.filter_app, f'apply_{filter_type}', None)
                if filter_method and callable(filter_method):
                    return filter_method(frame.copy(), dummy_face, self.frame_count)
            elif filter_type in full_image_filters:
                dummy_face = (0, 0, frame.shape[1], frame.shape[0])
                filter_method = getattr(self.filter_app, f'apply_{filter_type}', None)
                if filter_method and callable(filter_method):
                    return filter_method(frame.copy(), dummy_face)
            else:
                face = self.filter_app.detect_face(frame)
                if face:
                    filter_method = getattr(self.filter_app, f'apply_{filter_type}', None)
                    if filter_method and callable(filter_method):
                        return filter_method(frame.copy(), face)
        except Exception as e:
            self.logger.error(f"Error applying filter {filter_type}: {e}", exception=str(e))
            print(f"Error applying filter {filter_type}: {e}")
        
        return frame
    
    def draw_rounded_rect(self, img, pt1, pt2, color, thickness, radius):
        """Draw rounded rectangle border"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Ensure valid coordinates
        if x1 >= x2 or y1 >= y2:
            return
        
        # Adjust radius if too large
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        if radius <= 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            return
        
        # Draw main rectangle parts
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw rounded corners (arcs)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw modern UI overlay matching standalone.html design"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Scale factor for responsive design
        base_width = 1280
        base_height = 720
        scale_factor = min(w / base_width, h / base_height)
        
        # Theme colors (matching HTML)
        surface = tuple(self.theme['surface'])
        surface_alpha = self.theme['bg_alpha']
        text = tuple(self.theme['text'])
        text_secondary = tuple(self.theme['textSecondary'])
        accent = tuple(self.theme['accent'])
        border = tuple(self.theme['border'])
        status_connected = tuple(self.theme['statusConnected'])
        status_error = tuple(self.theme['statusError'])
        recording_color = tuple(self.theme['recording_color'])
        update_color = tuple(self.theme['update_color'])
        group_title = tuple(self.theme['groupTitle'])
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = max(0.3, 0.45 * scale_factor)
        font_medium = max(0.35, 0.5 * scale_factor)
        font_large = max(0.4, 0.6 * scale_factor)
        font_title = max(0.35, 0.5 * scale_factor)
        font_group = max(0.28, 0.4 * scale_factor)
        
        # Controls panel (top-left, matching HTML)
        panel_x = int(10 * scale_factor)
        panel_y = int(10 * scale_factor)
        panel_width = int(220 * scale_factor)
        panel_padding = int(12 * scale_factor)
        corner_radius = int(8 * scale_factor)
        
        # Calculate panel height based on content
        line_height = int(18 * scale_factor)
        section_spacing = int(8 * scale_factor)
        
        # Estimate content height (theme + camera + search + pinned + current + categories)
        num_categories = len(self.filter_categories)
        estimated_lines = 15 + (num_categories * 3) + min(20, len(self.filter_list))
        panel_height = int((estimated_lines * line_height) + (panel_padding * 2) + (section_spacing * 5))
        panel_height = min(panel_height, int(h * 0.85))  # Max 85% of screen height
        
        # Draw controls panel background with rounded corners
        panel_roi = overlay[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width]
        if panel_roi.size > 0:
            bg_overlay = np.full(panel_roi.shape, surface, dtype=np.uint8)
            overlay[panel_y:panel_y+panel_height, panel_x:panel_x+panel_width] = cv2.addWeighted(
                panel_roi, 1 - surface_alpha, bg_overlay, surface_alpha, 0
            )
        
        # Draw rounded border
        self.draw_rounded_rect(overlay, (panel_x, panel_y), 
                             (panel_x + panel_width, panel_y + panel_height),
                             border, 1, corner_radius)
        
        # Content position
        text_x = panel_x + panel_padding
        y_pos = panel_y + panel_padding + int(line_height * 0.8)
        
        # Theme selector (matching HTML exactly)
        cv2.putText(overlay, 'Theme:', (text_x, y_pos), font, font_small, 
                   text_secondary, max(1, int(scale_factor)))
        y_pos += int(line_height * 0.9)
        
        # Theme dropdown display (showing current with checkmark)
        current_theme_display = self.theme_name.title()
        for theme in self.available_themes:
            if theme == self.theme_name:
                current_theme_display = f'✓ {theme.title()}'
                break
        cv2.putText(overlay, current_theme_display, (text_x, y_pos), font, font_small, 
                   accent, max(1, int(scale_factor)))
        y_pos += int(line_height * 1.2)
        
        # Divider
        cv2.line(overlay, (text_x, y_pos), (panel_x + panel_width - panel_padding, y_pos), 
                border, 1)
        y_pos += int(section_spacing)
        
        # Search FX (matching HTML - input field style)
        cv2.putText(overlay, 'Search FX:', (text_x, y_pos), font, font_small, 
                   text_secondary, max(1, int(scale_factor)))
        y_pos += int(line_height * 0.9)
        
        # Draw search input box (like HTML input field)
        search_box_y1 = y_pos - int(line_height * 0.7)
        search_box_y2 = y_pos + int(line_height * 0.3)
        search_box_bg = overlay[search_box_y1:search_box_y2, text_x:panel_x + panel_width - panel_padding]
        if search_box_bg.size > 0:
            input_bg = np.full(search_box_bg.shape, surface, dtype=np.uint8)
            overlay[search_box_y1:search_box_y2, text_x:panel_x + panel_width - panel_padding] = cv2.addWeighted(
                search_box_bg, 0.3, input_bg, 0.7, 0
            )
        # Draw border around search box
        self.draw_rounded_rect(overlay, (text_x, search_box_y1), 
                             (panel_x + panel_width - panel_padding, search_box_y2),
                             accent if self.search_active else border, 1, int(4 * scale_factor))
        
        # Search text
        search_text = self.search_query if self.search_query else 'Search FX...'
        search_color = text_secondary if not self.search_query else text
        cv2.putText(overlay, search_text[:20], (text_x + int(4 * scale_factor), y_pos), font, font_small, 
                   search_color, max(1, int(scale_factor)))
        y_pos += int(line_height * 1.2)
        
        # Divider
        cv2.line(overlay, (text_x, y_pos), (panel_x + panel_width - panel_padding, y_pos), 
                border, 1)
        y_pos += int(section_spacing)
        
        # Pinned section (matching HTML - yellow buttons with X)
        if self.favorites:
            cv2.putText(overlay, 'Pinned:', (text_x, y_pos), font, font_small, 
                       text_secondary, max(1, int(scale_factor)))
            y_pos += int(line_height * 0.9)
            
            for fav_filter in self.favorites[:5]:  # Show max 5 pinned
                if y_pos + int(line_height * 1.2) > panel_y + panel_height - panel_padding:
                    break
                
                # Find display name
                fav_name = fav_filter.replace('_', ' ').title()
                for ft, name in self.filter_list:
                    if ft == fav_filter:
                        fav_name = name
                        break
                
                # Check if matches search
                if self.search_query:
                    search_lower = self.search_query.lower()
                    if search_lower not in fav_name.lower() and search_lower not in fav_filter.lower():
                        continue  # Skip if doesn't match search
                
                # Draw pinned button (yellow/accent colored like HTML)
                button_y1 = y_pos - int(line_height * 0.6)
                button_y2 = y_pos + int(line_height * 0.4)
                button_width = panel_width - (panel_padding * 2)
                
                # Draw button background with accent color
                button_bg_roi = overlay[button_y1:button_y2, text_x:text_x + button_width]
                if button_bg_roi.size > 0:
                    button_bg = np.full(button_bg_roi.shape, accent, dtype=np.uint8)
                    overlay[button_y1:button_y2, text_x:text_x + button_width] = cv2.addWeighted(
                        button_bg_roi, 0.2, button_bg, 0.8, 0
                    )
                
                # Draw button border
                self.draw_rounded_rect(overlay, (text_x, button_y1), 
                                     (text_x + button_width, button_y2),
                                     accent, 1, int(4 * scale_factor))
                
                # Button text with X
                button_text = f'{fav_name} X'
                text_color = tuple(self.theme['selectedText']) if self.theme_name == 'dropout' else text
                cv2.putText(overlay, button_text, (text_x + int(4 * scale_factor), y_pos), font, font_small, 
                           text_color, max(1, int(scale_factor)))
                y_pos += int(line_height * 1.1)
        
        # Divider
        if self.favorites:
            cv2.line(overlay, (text_x, y_pos), (panel_x + panel_width - panel_padding, y_pos), 
                    border, 1)
            y_pos += int(section_spacing)
        
        # Current FX (matching HTML - dropdown button style)
        cv2.putText(overlay, 'Current FX:', (text_x, y_pos), font, font_small, 
                   text_secondary, max(1, int(scale_factor)))
        y_pos += int(line_height * 0.9)
        
        # Current FX button (large, accent colored, like HTML dropdown button)
        button_y1 = y_pos - int(line_height * 0.6)
        button_y2 = y_pos + int(line_height * 0.5)
        button_width = panel_width - (panel_padding * 2)
        
        # Draw button background
        button_bg_roi = overlay[button_y1:button_y2, text_x:text_x + button_width]
        if button_bg_roi.size > 0:
            button_bg = np.full(button_bg_roi.shape, accent, dtype=np.uint8)
            overlay[button_y1:button_y2, text_x:text_x + button_width] = cv2.addWeighted(
                button_bg_roi, 0.1, button_bg, 0.9, 0
            )
        
        # Draw button border
        self.draw_rounded_rect(overlay, (text_x, button_y1), 
                             (text_x + button_width, button_y2),
                             accent, 1, int(4 * scale_factor))
        
        # Button text with pin icon
        pin_icon = '📌' if self.current_filter and self.current_filter in self.favorites else ''
        current_text = f'{self.current_filter_name} {pin_icon}'
        text_color = tuple(self.theme['selectedText']) if self.theme_name == 'dropout' else text
        cv2.putText(overlay, current_text, (text_x + int(4 * scale_factor), y_pos), font, font_medium, 
                   text_color, max(1, int(1.2 * scale_factor)))
        y_pos += int(line_height * 1.3)
        
        # Divider
        cv2.line(overlay, (text_x, y_pos), (panel_x + panel_width - panel_padding, y_pos), 
                border, 1)
        y_pos += int(section_spacing)
        
        # Filter categories (matching HTML structure)
        filtered = self.get_filtered_filters()
        categorized = self.get_filters_by_category()
        
        # Show DROPOUT first
        if 'DROPOUT' in categorized and categorized['DROPOUT']:
            if y_pos + line_height <= panel_y + panel_height - panel_padding:
                # Category title (group header style)
                cat_y1 = y_pos - int(line_height * 0.5)
                cat_y2 = y_pos + int(line_height * 0.5)
                cat_bg = overlay[cat_y1:cat_y2, text_x:panel_x + panel_width - panel_padding]
                if cat_bg.size > 0:
                    hover_bg = np.full(cat_bg.shape, tuple(self.theme['surfaceHover']), dtype=np.uint8)
                    overlay[cat_y1:cat_y2, text_x:panel_x + panel_width - panel_padding] = cv2.addWeighted(
                        cat_bg, 0.2, hover_bg, 0.8, 0
                    )
                cv2.putText(overlay, 'DROPOUT', (text_x, y_pos), font, font_group, 
                           group_title, max(1, int(1.1 * scale_factor)))
                y_pos += int(line_height * 1.0)
                
                # DROPOUT filters (skip pinned ones - they're in pinned section)
                for filter_type, name in categorized['DROPOUT']:
                    if y_pos + line_height > panel_y + panel_height - panel_padding:
                        break
                    # Skip if pinned
                    if filter_type and filter_type in self.favorites:
                        continue
                    # Check search filter (skip if search active and doesn't match)
                    if self.search_query:
                        search_lower = self.search_query.lower()
                        name_lower = name.lower()
                        type_lower = filter_type.lower() if filter_type else ''
                        if (search_lower not in name_lower and 
                            search_lower not in type_lower and
                            search_lower not in name_lower.replace(' ', '_')):
                            continue
                    is_active = filter_type == self.current_filter
                    
                    if is_active:
                        item_y1 = y_pos - int(line_height * 0.6)
                        item_y2 = y_pos + int(line_height * 0.4)
                        item_bg = overlay[item_y1:item_y2, text_x:panel_x + panel_width - panel_padding]
                        if item_bg.size > 0:
                            accent_bg = np.full(item_bg.shape, accent, dtype=np.uint8)
                            overlay[item_y1:item_y2, text_x:panel_x + panel_width - panel_padding] = cv2.addWeighted(
                                item_bg, 0.3, accent_bg, 0.7, 0
                            )
                    color = tuple(self.theme['selectedText']) if is_active else text
                    cv2.putText(overlay, name, (text_x, y_pos), font, font_small, 
                               color, max(1, int(1.1 * scale_factor) if is_active else int(scale_factor)))
                    y_pos += int(line_height * 0.9)
        
        # Show other categories
        for category in ['Distortion', 'Color & Style']:
            if category not in categorized or not categorized[category]:
                continue
            if y_pos + line_height > panel_y + panel_height - panel_padding:
                break
            
            # Category title
            cat_y1 = y_pos - int(line_height * 0.5)
            cat_y2 = y_pos + int(line_height * 0.5)
            cat_bg = overlay[cat_y1:cat_y2, text_x:panel_x + panel_width - panel_padding]
            if cat_bg.size > 0:
                hover_bg = np.full(cat_bg.shape, tuple(self.theme['surfaceHover']), dtype=np.uint8)
                overlay[cat_y1:cat_y2, text_x:panel_x + panel_width - panel_padding] = cv2.addWeighted(
                    cat_bg, 0.2, hover_bg, 0.8, 0
                )
            cv2.putText(overlay, category.upper(), (text_x, y_pos), font, font_group, 
                       group_title, max(1, int(1.1 * scale_factor)))
            y_pos += int(line_height * 1.0)
            
            # Category filters (skip pinned, apply search filter)
            for filter_type, name in categorized[category]:
                if y_pos + line_height > panel_y + panel_height - panel_padding:
                    break
                # Skip if pinned
                if filter_type and filter_type in self.favorites:
                    continue
                # Check search filter (skip if search active and doesn't match)
                if self.search_query:
                    search_lower = self.search_query.lower()
                    name_lower = name.lower()
                    type_lower = filter_type.lower() if filter_type else ''
                    if (search_lower not in name_lower and 
                        search_lower not in type_lower and
                        search_lower not in name_lower.replace(' ', '_')):
                        continue
                is_active = filter_type == self.current_filter
                
                if is_active:
                    item_y1 = y_pos - int(line_height * 0.6)
                    item_y2 = y_pos + int(line_height * 0.4)
                    item_bg = overlay[item_y1:item_y2, text_x:panel_x + panel_width - panel_padding]
                    if item_bg.size > 0:
                        accent_bg = np.full(item_bg.shape, accent, dtype=np.uint8)
                        overlay[item_y1:item_y2, text_x:panel_x + panel_width - panel_padding] = cv2.addWeighted(
                            item_bg, 0.3, accent_bg, 0.7, 0
                        )
                
                color = tuple(self.theme['selectedText']) if is_active else text
                cv2.putText(overlay, name, (text_x, y_pos), font, font_small, 
                           color, max(1, int(1.1 * scale_factor) if is_active else int(scale_factor)))
                y_pos += int(line_height * 0.9)
        
        # Status indicator (top-right, matching HTML)
        status_text = 'READY'
        status_bg_color = status_connected
        if self.recording:
            status_text = 'REC'
            status_bg_color = recording_color
        elif self.update_available:
            status_text = 'UPDATE'
            status_bg_color = update_color
        
        status_x = w - int(120 * scale_factor)
        status_y_pos = int(10 * scale_factor)
        status_width = int(100 * scale_factor)
        status_height = int(30 * scale_factor)
        
        # Draw status background
        status_roi = overlay[status_y_pos:status_y_pos+status_height, status_x:status_x+status_width]
        if status_roi.size > 0:
            status_bg = np.full(status_roi.shape, surface, dtype=np.uint8)
            overlay[status_y_pos:status_y_pos+status_height, status_x:status_x+status_width] = cv2.addWeighted(
                status_roi, 1 - surface_alpha, status_bg, surface_alpha, 0
            )
        
        # Draw status border with accent color
        self.draw_rounded_rect(overlay, (status_x, status_y_pos), 
                             (status_x + status_width, status_y_pos + status_height),
                             status_bg_color, 1, int(8 * scale_factor))
        
        # Status text
        (text_w, text_h), _ = cv2.getTextSize(status_text, font, font_small, 1)
        status_text_x = status_x + (status_width - text_w) // 2
        status_text_y = status_y_pos + (status_height + text_h) // 2
        cv2.putText(overlay, status_text, (status_text_x, status_text_y), font, font_small, 
                   text, max(1, int(scale_factor)))
        
        return overlay
    
    def run(self):
        print("WesWorld FX - Interactive Filter Viewer")
        print("=" * 50)
        print("Controls:")
        print(f"  SPACEBAR: Toggle auto-advance (cycles every {self.advance_interval:.1f} seconds)")
        print("  Arrow Left/Right: Cycle through filters manually")
        print("  Press H: Toggle UI overlay (hide/show)")
        print("  Press F: Toggle favorite for current filter")
        print("  Press R: Start/Stop recording")
        print("  Press U: Check/Pull updates")
        print("  Press T: Switch theme (WesWorld/Dropout/Default)")
        print("  Press /: Start search (then type to filter, Enter to select, Esc to clear)")
        print("  Press 0: None (Original)")
        for key in ['1', '2', '3', '4', '5', '6', '7']:
            if key in self.filters:
                _, name = self.filters[key]
                print(f"  Press {key}: {name}")
        print("  Press Q: Quit")
        if self.favorites:
            print(f"\nFavorites ({len(self.favorites)}): {', '.join(self.favorites[:5])}")
        print("=" * 50)
        print(f"\nStarting with: {self.current_filter_name}")
        print("\nCamera ready! Starting viewer...")
        print("Make sure the window has focus to use keyboard controls.")
        print("You can resize and move the window - it will maintain your preferences.\n")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        frame_time = max(1, 1000 // self.fps)
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print("\nError: Could not read frames from camera")
                    print("Camera may have been disconnected or is being used by another application.")
                    break
                import time
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            
            import time
            current_time = time.time()
            self.frame_count += 1
            
            frame = cv2.flip(frame, 1)
            
            if self.auto_advance:
                if current_time - self.last_advance_time >= self.advance_interval:
                    self.current_filter_index = (self.current_filter_index + 1) % len(self.filter_list)
                    self.current_filter, self.current_filter_name = self.filter_list[self.current_filter_index]
                    self.last_advance_time = current_time
                    print(f"Auto-advance: {self.current_filter_name}")
            
            filtered_frame = self.apply_filter(frame, self.current_filter)
            
            # Write to video if recording
            if self.recording and self.video_writer:
                self.video_writer.write(filtered_frame)
            
            if self.show_ui:
                display_frame = self.draw_overlay(filtered_frame)
            else:
                display_frame = filtered_frame
            
            # Periodic update check (every 5 minutes)
            if UPDATE_CHECKER_AVAILABLE and self.update_checker:
                if current_time - self.last_update_check > self.update_check_interval:
                    threading.Thread(target=self.check_for_updates, daemon=True).start()
            
            try:
                window_size = cv2.getWindowImageRect(self.window_name)
                if window_size[2] > 0 and window_size[3] > 0:
                    self.display_width = window_size[2]
                    self.display_height = window_size[3]
                    display_frame = cv2.resize(display_frame, (self.display_width, self.display_height), interpolation=cv2.INTER_LINEAR)
            except:
                pass
            
            window_title = 'WesWorld FX'
            if self.auto_advance:
                window_title += ' [AUTO-ADVANCE]'
            if self.recording:
                window_title += ' [RECORDING]'
            if self.update_available:
                window_title += ' [UPDATE AVAILABLE]'
            cv2.setWindowTitle(self.window_name, window_title)
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(frame_time)
            
            if key == -1:
                if self.number_buffer and current_time - self.last_number_input_time > self.number_input_timeout:
                    try:
                        display_num = int(self.number_buffer)
                        filter_index = self.display_number_to_index(display_num)
                        if filter_index is not None:
                            self.auto_advance = False
                            self.current_filter_index = filter_index
                            self.current_filter, self.current_filter_name = self.filter_list[filter_index]
                            print(f"Switched to filter {display_num}: {self.current_filter_name}")
                        else:
                            print(f"Filter number {display_num} out of range")
                    except ValueError:
                        pass
                    self.number_buffer = ''
                continue
                
            key_code = key & 0xFF
            
            if key_code == ord('q') or key_code == ord('Q'):
                break
            elif key_code == ord('h') or key_code == ord('H'):
                self.show_ui = not self.show_ui
                print(f"UI {'hidden' if not self.show_ui else 'shown'}")
            elif key_code == ord(' ') or key_code == 32:
                self.auto_advance = not self.auto_advance
                self.last_advance_time = current_time
                status = "ON" if self.auto_advance else "OFF"
                print(f"Auto-advance: {status}")
            elif key_code == 13 or key_code == 10:
                if self.number_buffer:
                    try:
                        display_num = int(self.number_buffer)
                        filter_index = self.display_number_to_index(display_num)
                        if filter_index is not None:
                            self.auto_advance = False
                            self.current_filter_index = filter_index
                            self.current_filter, self.current_filter_name = self.filter_list[filter_index]
                            print(f"Switched to filter {display_num}: {self.current_filter_name}")
                        else:
                            print(f"Filter number {display_num} out of range")
                    except ValueError:
                        pass
                    self.number_buffer = ''
            elif key == 81 or key == 65361 or key == 2:
                self.auto_advance = False
                self.number_buffer = ''
                self.current_filter_index = (self.current_filter_index - 1) % len(self.filter_list)
                self.current_filter, self.current_filter_name = self.filter_list[self.current_filter_index]
                print(f"Switched to: {self.current_filter_name}")
            elif key == 83 or key == 65363 or key == 3:
                self.auto_advance = False
                self.number_buffer = ''
                self.current_filter_index = (self.current_filter_index + 1) % len(self.filter_list)
                self.current_filter, self.current_filter_name = self.filter_list[self.current_filter_index]
                print(f"Switched to: {self.current_filter_name}")
            elif key_code >= ord('0') and key_code <= ord('9'):
                self.number_buffer += chr(key_code)
                self.last_number_input_time = current_time
                try:
                    display_num = int(self.number_buffer)
                    filter_index = self.display_number_to_index(display_num)
                    if filter_index is not None:
                        filter_name = self.filter_list[filter_index][1]
                        print(f"Entering filter number: {self.number_buffer} -> {filter_name} (press Enter or wait 1s)")
                    else:
                        print(f"Filter number {display_num} out of range")
                except ValueError:
                    pass
            elif chr(key_code) in self.filters:
                self.auto_advance = False
                self.number_buffer = ''
                filter_type, filter_name = self.filters[chr(key_code)]
                try:
                    self.current_filter_index = next(i for i, (f, _) in enumerate(self.filter_list) if f == filter_type)
                except StopIteration:
                    continue
                self.current_filter = filter_type
                self.current_filter_name = filter_name
                print(f"Switched to: {filter_name}")
            elif key_code == ord('f') or key_code == ord('F'):
                self.toggle_favorite(self.current_filter)
            elif key_code == ord('r') or key_code == ord('R'):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key_code == ord('u') or key_code == ord('U'):
                if self.update_available and self.update_checker:
                    self.logger.info("Pulling updates...")
                    print("Pulling updates...")
                    success, message = self.update_checker.pull_updates()
                    if success:
                        self.logger.log_event("update_pulled", {"success": True})
                        print(f"✅ {message}")
                        print("Please restart the application to apply updates.")
                        self.update_available = False
                    else:
                        self.logger.log_event("update_pull_failed", {"message": message})
                        print(f"❌ {message}")
                else:
                    self.logger.info("Checking for updates manually...")
                    print("Checking for updates...")
                    self.check_for_updates(force=True)
            elif key_code == ord('t') or key_code == ord('T'):
                # Cycle themes
                current_idx = self.available_themes.index(self.theme_name)
                next_idx = (current_idx + 1) % len(self.available_themes)
                self.switch_theme(self.available_themes[next_idx])
            elif key_code == ord('/') or key_code == ord('?'):
                # Start/activate search (if not already active)
                if not self.search_active:
                    self.search_active = True
                    self.search_buffer = ''
                    self.search_query = ''
                    print("Search active: Type to search, Enter to select first result, Esc to clear")
                else:
                    # If already active, treat as character input
                    self.search_buffer += '/'
                    self.search_query = self.search_buffer
                    self.last_search_key_time = current_time
                    filtered = self.get_filtered_filters()
                    count = len(filtered) if filtered else 0
                    print(f"Search: '{self.search_query}' ({count} results)")
            elif self.search_active or self.search_query:
                if key_code == 27:  # Escape - clear search
                    self.search_query = ''
                    self.search_buffer = ''
                    self.search_active = False
                    print("Search cleared")
                elif key_code == 8:  # Backspace
                    if self.search_buffer:
                        self.search_buffer = self.search_buffer[:-1]
                        self.search_query = self.search_buffer
                        print(f"Search: {self.search_query}")
                elif key_code == 13 or key_code == 10:  # Enter - select first result
                    filtered = self.get_filtered_filters()
                    if filtered and len(filtered) > 0:
                        # Try pinned first
                        pinned_matches = [f for f in filtered if f[0] and f[0] in self.favorites]
                        if pinned_matches:
                            filter_type, name = pinned_matches[0]
                        else:
                            filter_type, name = filtered[0]
                        
                        # Find and select
                        for i, (ft, n) in enumerate(self.filter_list):
                            if ft == filter_type:
                                self.current_filter_index = i
                                self.current_filter = filter_type
                                self.current_filter_name = name
                                self.search_query = ''
                                self.search_buffer = ''
                                self.search_active = False
                                print(f"Selected: {name}")
                                break
                elif (key_code >= ord('a') and key_code <= ord('z')) or \
                     (key_code >= ord('A') and key_code <= ord('Z')) or \
                     (key_code >= ord('0') and key_code <= ord('9')) or \
                     key_code == ord(' ') or key_code == ord('_') or key_code == ord('-'):
                    # Add character to search
                    if not self.search_active:
                        self.search_active = True
                    char = chr(key_code).lower() if key_code != ord(' ') else ' '
                    self.search_buffer += char
                    self.search_query = self.search_buffer
                    self.last_search_key_time = current_time
                    filtered = self.get_filtered_filters()
                    count = len(filtered) if filtered else 0
                    print(f"Search: '{self.search_query}' ({count} results)")
            
            # Check if search should timeout (clear after inactivity)
            if self.search_query and not self.search_active:
                if current_time - self.last_search_key_time > self.search_timeout:
                    self.search_query = ''
                    self.search_buffer = ''
                    print("Search cleared (timeout)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WesWorld FX - Interactive face filter viewer')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    
    args = parser.parse_args()
    
    try:
        with InteractiveFilterViewer(width=args.width, height=args.height, fps=args.fps) as viewer:
            viewer.run()
    except KeyboardInterrupt:
        print("\nStopping viewer...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

