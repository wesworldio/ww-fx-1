import cv2
import numpy as np
import sys
import json
import os
from face_filters import FaceFilter
from typing import Tuple, Optional


class InteractiveFilterViewer:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        self.filter_list = [
            ('sam_reich', 'SAM REICH Tattoo'),
            ('sam_face_mask', 'Sam Face Mask'),
            ('extreme_closeup', 'Extreme Close-Up'),
            ('puzzle', 'Puzzle'),
            ('fast_zoom_in', 'Fast Zoom In'),
            ('fast_zoom_out', 'Fast Zoom Out'),
            ('shake', 'Shake'),
            ('pulse', 'Pulse'),
            ('spiral_zoom', 'Spiral Zoom'),
            (None, 'None (Original)'),
            ('bulge', 'Bulge'),
            ('stretch', 'Stretch'),
            ('swirl', 'Swirl'),
            ('fisheye', 'Fisheye'),
            ('pinch', 'Pinch'),
            ('wave', 'Wave'),
            ('mirror', 'Mirror'),
            ('twirl', 'Twirl'),
            ('ripple', 'Ripple'),
            ('sphere', 'Sphere'),
            ('tunnel', 'Tunnel'),
            ('water_ripple', 'Water Ripple'),
            ('radial_blur', 'Radial Blur'),
            ('cylinder', 'Cylinder'),
            ('barrel', 'Barrel'),
            ('pincushion', 'Pincushion'),
            ('whirlpool', 'Whirlpool'),
            ('radial_zoom', 'Radial Zoom'),
            ('concave', 'Concave'),
            ('convex', 'Convex'),
            ('spiral', 'Spiral'),
            ('radial_stretch', 'Radial Stretch'),
            ('radial_compress', 'Radial Compress'),
            ('vertical_wave', 'Vertical Wave'),
            ('horizontal_wave', 'Horizontal Wave'),
            ('skew_horizontal', 'Skew Horizontal'),
            ('skew_vertical', 'Skew Vertical'),
            ('rotate_zoom', 'Rotate Zoom'),
            ('radial_wave', 'Radial Wave'),
            ('zoom_in', 'Zoom In'),
            ('zoom_out', 'Zoom Out'),
            ('rotate', 'Rotate'),
            ('rotate_45', 'Rotate 45°'),
            ('rotate_90', 'Rotate 90°'),
            ('flip_horizontal', 'Flip Horizontal'),
            ('flip_vertical', 'Flip Vertical'),
            ('flip_both', 'Flip Both'),
            ('quad_mirror', 'Quad Mirror'),
            ('tile', 'Tile'),
            ('radial_tile', 'Radial Tile'),
            ('zoom_blur', 'Zoom Blur'),
            ('melt', 'Melt'),
            ('kaleidoscope', 'Kaleidoscope'),
            ('glitch', 'Glitch'),
            ('double_vision', 'Double Vision'),
            ('black_white', 'Black & White'),
            ('sepia', 'Sepia'),
            ('vintage', 'Vintage'),
            ('neon_glow', 'Neon Glow'),
            ('pixelate', 'Pixelate'),
            ('blur', 'Blur'),
            ('sharpen', 'Sharpen'),
            ('emboss', 'Emboss'),
            ('red_tint', 'Red Tint'),
            ('blue_tint', 'Blue Tint'),
            ('green_tint', 'Green Tint'),
            ('rainbow', 'Rainbow'),
            ('negative', 'Negative'),
            ('posterize', 'Posterize'),
            ('sketch', 'Sketch'),
            ('cartoon', 'Cartoon'),
            ('thermal', 'Thermal'),
            ('ice', 'Ice'),
            ('ocean', 'Ocean'),
            ('plasma', 'Plasma'),
            ('jet', 'Jet'),
            ('turbo', 'Turbo'),
            ('inferno', 'Inferno'),
            ('magma', 'Magma'),
            ('viridis', 'Viridis'),
            ('cool', 'Cool'),
            ('hot', 'Hot'),
            ('spring', 'Spring'),
            ('summer', 'Summer'),
            ('autumn', 'Autumn'),
            ('winter', 'Winter'),
            ('rainbow_shift', 'Rainbow Shift'),
            ('acid_trip', 'Acid Trip'),
            ('double_vision', 'Double Vision'),
            ('zoom_blur', 'Zoom Blur'),
            ('melt', 'Melt'),
            ('kaleidoscope', 'Kaleidoscope'),
            ('glitch', 'Glitch'),
            ('vhs', 'VHS'),
            ('retro', 'Retro'),
            ('cyberpunk', 'Cyberpunk'),
            ('anime', 'Anime'),
            ('glow', 'Glow'),
            ('solarize', 'Solarize'),
            ('edge_detect', 'Edge Detect'),
            ('halftone', 'Halftone'),
        ]
        
        self.filters = {
            's': self.filter_list[0],
            '1': self.filter_list[2],
            '2': self.filter_list[3],
            '3': self.filter_list[4],
            '4': self.filter_list[5],
            '5': self.filter_list[6],
            '6': self.filter_list[7],
            '7': self.filter_list[8],
            '0': self.filter_list[1],
        }
        
        sam_reich_idx = next((i for i, (ft, _) in enumerate(self.filter_list) if ft == 'sam_reich'), 0)
        self.current_filter_index = sam_reich_idx
        self.current_filter = self.filter_list[sam_reich_idx][0]
        self.current_filter_name = self.filter_list[sam_reich_idx][1]
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
        
    def __enter__(self):
        print("Opening camera...")
        
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
                        print(f"Camera {idx} initialized successfully! (saved to config)")
                    else:
                        print(f"Camera {idx} initialized successfully! (using saved config)")
                    print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
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
        if self.filter_app:
            self.filter_app.__exit__(exc_type, exc_val, exc_tb)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
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
        
        face_tracking_filters = {'sam_reich', 'sam_face_mask'}
        
        try:
            if filter_type in face_tracking_filters:
                if filter_type == 'sam_reich':
                    face = self.filter_app.detect_face(frame)
                    if face:
                        return self.filter_app.apply_sam_reich_tattoo(frame.copy(), face)
                elif filter_type == 'sam_face_mask':
                    faces = self.filter_app.detect_all_faces(frame)
                    if faces:
                        return self.filter_app.apply_sam_face_mask(frame.copy(), faces)
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
            print(f"Error applying filter {filter_type}: {e}")
        
        return frame
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        scale_factor = min(w / 1280.0, h / 720.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        bg_color = (20, 20, 20)
        bg_alpha = 0.85
        text_color = (240, 240, 240)
        highlight_color = (239, 80, 82)
        border_color = (239, 80, 82)
        
        box_width = int(320 * scale_factor)
        box_height = int(min(450 * scale_factor, h - 40))
        padding = int(12 * scale_factor)
        text_x = int(18 * scale_factor)
        y_start = int(15 * scale_factor)
        
        bg_roi = overlay[padding:padding+box_height, padding:padding+box_width]
        bg_overlay = np.full(bg_roi.shape, bg_color, dtype=np.uint8)
        overlay[padding:padding+box_height, padding:padding+box_width] = cv2.addWeighted(
            bg_roi, 1 - bg_alpha, bg_overlay, bg_alpha, 0
        )
        
        cv2.rectangle(overlay, (padding, padding), (padding + box_width, padding + box_height), 
                     border_color, max(2, int(3 * scale_factor)))
        
        small_font_scale = max(0.4, 0.6 * scale_factor)
        medium_font_scale = max(0.45, 0.55 * scale_factor)
        large_font_scale = max(0.6, 0.7 * scale_factor)
        title_font_scale = max(0.5, 0.65 * scale_factor)
        
        line_height = int(22 * scale_factor)
        y_offset = y_start + int(line_height * 0.8)
        
        title_text = 'WesWorld FX'
        title_thickness = max(2, int(2 * scale_factor))
        (text_width, text_height), _ = cv2.getTextSize(title_text, font, title_font_scale, title_thickness)
        if text_x + text_width > box_width - padding:
            title_font_scale = max(0.4, title_font_scale * 0.9)
            (text_width, text_height), _ = cv2.getTextSize(title_text, font, title_font_scale, title_thickness)
        
        cv2.putText(overlay, title_text, (text_x, y_offset), font, title_font_scale, 
                   highlight_color, title_thickness)
        y_offset += text_height + int(line_height * 0.5)
        
        cv2.putText(overlay, f'Filter: {self.current_filter_name}', (text_x, y_offset), 
                   font, large_font_scale, highlight_color, max(2, int(2 * scale_factor)))
        y_offset += int(line_height * 1.5)
        
        cv2.putText(overlay, 'Controls:', (text_x, y_offset), font, small_font_scale, 
                   text_color, max(1, int(1.5 * scale_factor)))
        y_offset += int(line_height * 1.2)
        
        controls = [
            ('H', 'Toggle UI'),
            ('SPACE', 'Auto-advance'),
            ('<- ->', 'Navigate'),
            ('Q', 'Quit')
        ]
        
        for key, desc in controls:
            cv2.putText(overlay, f'  {key}: {desc}', (text_x, y_offset), 
                       font, medium_font_scale, text_color, max(1, int(scale_factor)))
            y_offset += int(line_height * 0.9)
        
        y_offset += int(line_height * 0.5)
        
        max_visible = int((box_height - y_offset + padding) / (line_height * 0.85))
        visible_range = max_visible // 2
        
        start_idx = max(0, self.current_filter_index - visible_range)
        end_idx = min(len(self.filter_list), start_idx + max_visible)
        
        if end_idx - start_idx < max_visible:
            start_idx = max(0, end_idx - max_visible)
        
        if start_idx > 0:
            cv2.putText(overlay, '  ...', (text_x, y_offset), font, medium_font_scale, 
                       (150, 150, 150), max(1, int(scale_factor)))
            y_offset += int(line_height * 0.85)
        
        for i in range(start_idx, end_idx):
            filter_type, name = self.filter_list[i]
            is_active = i == self.current_filter_index
            color = highlight_color if is_active else text_color
            thickness = max(2, int(1.5 * scale_factor)) if is_active else max(1, int(scale_factor))
            
            marker = ' [ACTIVE]' if is_active else ''
            
            key_label = self.index_to_display_number(i)
            if key_label is None:
                key_label = str(i)
            
            display_text = f'  {key_label}: {name}{marker}'
            cv2.putText(overlay, display_text, (text_x, y_offset), 
                       font, medium_font_scale, color, thickness)
            y_offset += int(line_height * 0.85)
        
        if end_idx < len(self.filter_list):
            cv2.putText(overlay, '  ...', (text_x, y_offset), font, medium_font_scale, 
                       (150, 150, 150), max(1, int(scale_factor)))
            y_offset += int(line_height * 0.85)
        
        if self.auto_advance:
            y_offset += int(line_height * 0.5)
            cv2.putText(overlay, f'AUTO-ADVANCE: ON ({self.advance_interval:.1f}s)', (text_x, y_offset), 
                       font, medium_font_scale, (0, 255, 0), max(2, int(2 * scale_factor)))
        
        return overlay
    
    def run(self):
        print("WesWorld FX - Interactive Filter Viewer")
        print("=" * 50)
        print("Controls:")
        print(f"  SPACEBAR: Toggle auto-advance (cycles every {self.advance_interval:.1f} seconds)")
        print("  Arrow Left/Right: Cycle through filters manually")
        print("  Press H: Toggle UI overlay (hide/show)")
        print("  Press S: SAM REICH Tattoo (default)")
        print("  Press 0: None (Original)")
        for key in ['1', '2', '3', '4', '5', '6', '7']:
            if key in self.filters:
                _, name = self.filters[key]
                print(f"  Press {key}: {name}")
        print("  Press Q: Quit")
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
            if self.show_ui:
                display_frame = self.draw_overlay(filtered_frame)
            else:
                display_frame = filtered_frame
            
            try:
                window_size = cv2.getWindowImageRect(self.window_name)
                if window_size[2] > 0 and window_size[3] > 0:
                    self.display_width = window_size[2]
                    self.display_height = window_size[3]
                    display_frame = cv2.resize(display_frame, (self.display_width, self.display_height), interpolation=cv2.INTER_LINEAR)
            except:
                pass
            
            window_title = 'WesWorld FX - Press SPACE to auto-advance, Q to quit'
            if self.auto_advance:
                window_title += ' [AUTO-ADVANCE ON]'
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

