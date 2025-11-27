import cv2
import numpy as np
import pyvirtualcam
from typing import Tuple, Optional, List
import argparse
import sys
import os
import time
import json


class FaceFilter:
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        self.camera_index = self.load_camera_index()
        self.sam_drops = []
        self.last_spawn_time = 0
    
    def load_camera_index(self) -> Optional[int]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    camera_index = config.get('camera_index')
                    if camera_index is not None:
                        return int(camera_index)
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        return None
    
    def save_camera_index(self, camera_index: int):
        config = {'camera_index': camera_index}
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save camera index to config: {e}")
        
    def __enter__(self):
        camera_indices_to_try = []
        if self.camera_index is not None:
            camera_indices_to_try.append(self.camera_index)
        camera_indices_to_try.extend([0, 1, 2])
        camera_indices_to_try = list(dict.fromkeys(camera_indices_to_try))
        
        for idx in camera_indices_to_try:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                time.sleep(0.5)
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    if idx != self.camera_index:
                        self.save_camera_index(idx)
                    return self
                else:
                    self.cap.release()
                    self.cap = None
                    if idx == self.camera_index:
                        self.camera_index = None
        
        raise RuntimeError("Could not open camera")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (x, y, w, h)
        return None
    
    def detect_all_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
        
    def apply_bulge(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist_sq = dx * dx + dy * dy
        max_dist_sq = radius * radius
        
        mask = dist_sq < max_dist_sq
        dist = np.sqrt(np.maximum(dist_sq, 1))
        
        strength = 0.5
        factor = 1.0 - (dist / radius) * strength
        factor = np.clip(factor, 0, 1)
        
        new_x = center_x + dx * factor
        new_y = center_y + dy * factor
        
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
    def apply_stretch(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = face
        center_x = x + w // 2
        center_y = y + h // 2
        
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        
        dx = x_coords - center_x
        dy = y_coords - center_y
        
        stretch_x = 1.5
        stretch_y = 0.7
        
        new_x = center_x + dx * stretch_x
        new_y = center_y + dy * stretch_y
        
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def apply_swirl(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        
        y_coords, x_coords = np.ogrid[:h_frame, :w_frame]
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        angle = np.arctan2(dy, dx)
        swirl_strength = 2.0
        max_angle = swirl_strength * (1.0 - np.clip(dist / radius, 0, 1))
        
        new_angle = angle + max_angle
        new_x = center_x + dist * np.cos(new_angle)
        new_y = center_y + dist * np.sin(new_angle)
        
        new_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        new_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        mask = dist < radius
        map_x, map_y = np.meshgrid(np.arange(w_frame), np.arange(h_frame))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        map_x[mask] = new_x[mask]
        map_y[mask] = new_y[mask]
        
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def apply_fisheye(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        max_dist = radius
        normalized_dist = np.clip(dist / max_dist, 0, 1)
        
        fisheye_strength = 0.8
        new_dist = normalized_dist * (1.0 - fisheye_strength * normalized_dist * normalized_dist)
        new_dist = new_dist * max_dist
        
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def apply_pinch(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        
        y_coords, x_coords = np.ogrid[:h_frame, :w_frame]
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        max_dist = radius
        normalized_dist = np.clip(dist / max_dist, 0, 1)
        
        pinch_strength = 0.6
        new_dist = normalized_dist * (1.0 + pinch_strength * (1.0 - normalized_dist))
        new_dist = new_dist * max_dist
        
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        
        new_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        new_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        mask = dist < radius
        map_x, map_y = np.meshgrid(np.arange(w_frame), np.arange(h_frame))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        map_x[mask] = new_x[mask]
        map_y[mask] = new_y[mask]
        
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return result
        
    def apply_wave(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_y = h_frame // 2
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        
        wave_amplitude = 30.0
        wave_frequency = 0.05
        wave_phase = np.sin((y_coords - center_y) * wave_frequency) * wave_amplitude
        
        new_x = x_coords + wave_phase
        new_y = y_coords
        
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
    def apply_mirror_split(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        
        left_half = result[:, :center_x]
        right_half = cv2.flip(left_half, 1)
        
        if center_x + right_half.shape[1] <= w_frame:
            result[:, center_x:center_x + right_half.shape[1]] = right_half
        
        return result
    
    def apply_sam_reich_tattoo(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = face
        result = frame.copy()
        
        forehead_y = y + int(h * 0.1)
        text_scale = max(0.5, w / 800.0)
        thickness = max(2, int(text_scale * 2))
        
        text = "SAM REICH"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, text_scale, thickness)
        text_x = x + (w - text_width) // 2
        text_y = forehead_y + text_height
        
        cv2.putText(result, text, (text_x, text_y), font, text_scale, (0, 0, 0), thickness)
        
        return result
    
    def apply_sam_face_mask(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        sam_path = os.path.join(os.path.dirname(__file__), 'assets', 'sam.png')
        
        if not os.path.exists(sam_path):
            return result
        
        sam_img = cv2.imread(sam_path, cv2.IMREAD_UNCHANGED)
        if sam_img is None:
            return result
        
        faces = self.detect_all_faces(frame)
        if not faces:
            return result
        
        for x, y, w, h in faces:
            sam_resized = cv2.resize(sam_img, (w, h))
            
            if sam_resized.shape[2] == 4:
                alpha = sam_resized[:, :, 3:4] / 255.0
                sam_bgr = sam_resized[:, :, :3]
                roi = result[y:y+h, x:x+w]
                blended = (alpha * sam_bgr + (1 - alpha) * roi).astype(np.uint8)
                result[y:y+h, x:x+w] = blended
            else:
                result[y:y+h, x:x+w] = sam_resized
        
        return result
    
    def apply_black_white(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def apply_sepia(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy().astype(np.float32)
        sepia_matrix = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        result = cv2.transform(result, sepia_matrix)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def apply_vintage(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = self.apply_sepia(frame, face)
        noise = np.random.randint(0, 50, result.shape, dtype=np.uint8)
        return cv2.add(result, noise)
    
    def apply_neon_glow(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (21, 21), 0)
        neon = cv2.addWeighted(roi, 1.5, blurred, -0.5, 0)
        result[y:y+h, x:x+w] = neon
        return result
    
    def apply_pixelate(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        small = cv2.resize(roi, (w//10, h//10), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result[y:y+h, x:x+w] = pixelated
        return result
    
    def apply_blur(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (51, 51), 0)
        result[y:y+h, x:x+w] = blurred
        return result
    
    def apply_sharpen(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(roi, -1, kernel)
        result[y:y+h, x:x+w] = sharpened
        return result
    
    def apply_emboss(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
        embossed = cv2.filter2D(roi, -1, kernel)
        result[y:y+h, x:x+w] = embossed
        return result
    
    def apply_red_tint(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        roi[:,:,2] = np.clip(roi[:,:,2] * 1.5, 0, 255)
        result[y:y+h, x:x+w] = roi
        return result
    
    def apply_blue_tint(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        roi[:,:,0] = np.clip(roi[:,:,0] * 1.5, 0, 255)
        result[y:y+h, x:x+w] = roi
        return result
    
    def apply_green_tint(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        roi[:,:,1] = np.clip(roi[:,:,1] * 1.5, 0, 255)
        result[y:y+h, x:x+w] = roi
        return result
    
    def apply_rainbow(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 30) % 180
        rainbow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result[y:y+h, x:x+w] = rainbow
        return result
    
    def apply_negative(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        negative = 255 - roi
        result[y:y+h, x:x+w] = negative
        return result
    
    def apply_posterize(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        roi = (roi // 32) * 32
        result[y:y+h, x:x+w] = roi
        return result
    
    def apply_sketch(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        result[y:y+h, x:x+w] = sketch_bgr
        return result
    
    def apply_cartoon(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        result[y:y+h, x:x+w] = cartoon
        return result
    
    def apply_oil_painting(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        oil = cv2.xphoto.oilPainting(roi, 7, 1) if hasattr(cv2, 'xphoto') else roi
        result[y:y+h, x:x+w] = oil
        return result
    
    def apply_watercolor(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        x, y, w, h = face
        roi = result[y:y+h, x:x+w]
        blurred = cv2.bilateralFilter(roi, 9, 75, 75)
        watercolor = cv2.stylization(blurred, sigma_s=60, sigma_r=0.6) if hasattr(cv2, 'stylization') else blurred
        result[y:y+h, x:x+w] = watercolor
        return result
    
    def apply_thermal(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    
    def apply_ice(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)
    
    def apply_ocean(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
    
    def apply_plasma(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_PLASMA)
    
    def apply_jet(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    def apply_turbo(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    
    def apply_inferno(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    
    def apply_magma(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    
    def apply_viridis(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
    
    def apply_cool(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
    
    def apply_hot(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    
    def apply_spring(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)
    
    def apply_summer(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_SUMMER)
    
    def apply_autumn(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_AUTUMN)
    
    def apply_winter(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)
    
    def apply_rainbow_shift(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 60) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_acid_trip(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = self.apply_swirl(frame, face)
        return self.apply_rainbow(result, face)
    
    def apply_double_vision(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        shifted = np.roll(frame, 10, axis=1)
        return cv2.addWeighted(frame, 0.5, shifted, 0.5, 0)
    
    def apply_zoom_blur(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        
        zoom_factor = 1.0 + (dist / max_dist) * 0.3
        
        new_x = center_x + dx * zoom_factor
        new_y = center_y + dy * zoom_factor
        
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        blurred = cv2.GaussianBlur(result, (15, 15), 0)
        return cv2.remap(blurred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_melt(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        
        melt_strength = 30.0
        new_y = y_coords + np.sin(x_coords * 0.05) * melt_strength
        
        map_x = x_coords.astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_kaleidoscope(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        small_h, small_w = h // 2, w // 2
        roi = cv2.resize(frame, (small_w, small_h))
        flipped_h = cv2.flip(roi, 1)
        flipped_v = cv2.flip(roi, 0)
        flipped_both = cv2.flip(roi, -1)
        combined = np.hstack([np.vstack([roi, flipped_v]), np.vstack([flipped_h, flipped_both])])
        return cv2.resize(combined, (w, h))
    
    def apply_glitch(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.copy()
        h, w = frame.shape[:2]
        for i in range(0, h, 20):
            offset = np.random.randint(-10, 10)
            if 0 <= i+offset < h:
                result[i:i+10] = frame[i+offset:i+offset+10]
        return result
    
    def apply_vhs(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        vhs = cv2.addWeighted(frame, 0.8, noise, 0.2, 0)
        scanlines = np.zeros_like(frame)
        scanlines[::3] = [0, 255, 0]
        return cv2.addWeighted(vhs, 0.9, scanlines, 0.1, 0)
    
    def apply_retro(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = self.apply_sepia(frame, face)
        return self.apply_vhs(result, face)
    
    def apply_cyberpunk(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 120) % 180
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.3, 0, 255)
        cyber = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        edges = cv2.Canny(frame, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(cyber, 0.8, edges_bgr, 0.2, 0)
    
    def apply_anime(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        smoothed = cv2.bilateralFilter(frame, 9, 75, 75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(smoothed, 0.7, edges_bgr, 0.3, 0)
    
    def apply_glow(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        glow = cv2.addWeighted(frame, 1.2, blurred, 0.3, 0)
        return np.clip(glow, 0, 255)
    
    def apply_solarize(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        result = frame.astype(np.float32)
        threshold = 128
        mask = result > threshold
        solarized = result.copy()
        solarized[mask] = 255 - result[mask]
        return np.clip(solarized, 0, 255).astype(np.uint8)
    
    def apply_edge_detect(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def apply_halftone(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small = cv2.resize(gray, (w//4, h//4))
        halftone = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        halftone = (halftone // 64) * 64
        return cv2.cvtColor(halftone, cv2.COLOR_GRAY2BGR)
    
    def apply_twirl(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        twirl_strength = 3.0
        max_angle = twirl_strength * (1.0 - np.clip(dist / radius, 0, 1))
        new_angle = angle + max_angle
        new_x = center_x + dist * np.cos(new_angle)
        new_y = center_y + dist * np.sin(new_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_ripple(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        ripple_frequency = 0.1
        ripple_amplitude = 20.0
        ripple = np.sin(dist * ripple_frequency) * ripple_amplitude
        angle = np.arctan2(dy, dx)
        new_x = x_coords + ripple * np.cos(angle)
        new_y = y_coords + ripple * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_sphere(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        normalized_dist = np.clip(dist / radius, 0, 1)
        sphere_strength = 0.5
        new_dist = normalized_dist * (1.0 - sphere_strength * normalized_dist)
        new_dist = new_dist * radius
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_tunnel(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        normalized_dist = np.clip(dist / max_dist, 0, 1)
        tunnel_strength = 0.8
        new_dist = normalized_dist * (1.0 + tunnel_strength * normalized_dist)
        new_dist = new_dist * max_dist
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_water_ripple(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        ripple_frequency = 0.05
        ripple_amplitude = 15.0
        ripple = np.sin(dist * ripple_frequency) * ripple_amplitude
        angle = np.arctan2(dy, dx)
        new_x = x_coords + ripple * np.cos(angle)
        new_y = y_coords + ripple * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_radial_blur(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        blur_strength = 5.0
        offset = blur_strength * (dist / max_dist)
        angle = np.arctan2(dy, dx)
        new_x = x_coords + offset * np.cos(angle)
        new_y = y_coords + offset * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        return cv2.remap(blurred, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_cylinder(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        cylinder_strength = 0.3
        new_x = center_x + dx * (1.0 - cylinder_strength * (dx / (w_frame // 2))**2)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = y_coords.astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_barrel(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        normalized_dist = np.clip(dist / radius, 0, 1)
        barrel_strength = 0.3
        new_dist = normalized_dist * (1.0 + barrel_strength * normalized_dist * normalized_dist)
        new_dist = new_dist * radius
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_pincushion(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        normalized_dist = np.clip(dist / radius, 0, 1)
        pincushion_strength = 0.4
        new_dist = normalized_dist * (1.0 - pincushion_strength * normalized_dist)
        new_dist = new_dist * radius
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_whirlpool(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        whirlpool_strength = 4.0
        max_angle = whirlpool_strength * (1.0 - np.clip(dist / radius, 0, 1))
        new_angle = angle + max_angle
        new_dist = dist * 0.9
        new_x = center_x + new_dist * np.cos(new_angle)
        new_y = center_y + new_dist * np.sin(new_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_radial_zoom(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        zoom_strength = 0.5
        zoom_factor = 1.0 + zoom_strength * (1.0 - dist / max_dist)
        new_x = center_x + dx * zoom_factor
        new_y = center_y + dy * zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_concave(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        normalized_dist = np.clip(dist / radius, 0, 1)
        concave_strength = 0.6
        new_dist = normalized_dist * (1.0 - concave_strength * normalized_dist)
        new_dist = new_dist * radius
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_convex(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        normalized_dist = np.clip(dist / radius, 0, 1)
        convex_strength = 0.5
        new_dist = normalized_dist * (1.0 + convex_strength * normalized_dist)
        new_dist = new_dist * radius
        angle = np.arctan2(dy, dx)
        new_x = center_x + new_dist * np.cos(angle)
        new_y = center_y + new_dist * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_spiral(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        radius = min(w_frame, h_frame) // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        spiral_turns = 2.0
        spiral_angle = angle + spiral_turns * np.pi * (dist / radius)
        new_x = center_x + dist * np.cos(spiral_angle)
        new_y = center_y + dist * np.sin(spiral_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_radial_stretch(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        stretch_factor = 1.0 + 0.5 * (dist / max_dist)
        angle = np.arctan2(dy, dx)
        new_x = center_x + dist * stretch_factor * np.cos(angle)
        new_y = center_y + dist * stretch_factor * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_radial_compress(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        compress_factor = 1.0 - 0.3 * (dist / max_dist)
        angle = np.arctan2(dy, dx)
        new_x = center_x + dist * compress_factor * np.cos(angle)
        new_y = center_y + dist * compress_factor * np.sin(angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_vertical_wave(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        wave_amplitude = 25.0
        wave_frequency = 0.05
        wave_phase = np.sin((x_coords - w_frame//2) * wave_frequency) * wave_amplitude
        new_x = x_coords
        new_y = y_coords + wave_phase
        map_x = new_x.astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_horizontal_wave(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        wave_amplitude = 25.0
        wave_frequency = 0.05
        wave_phase = np.sin((y_coords - h_frame//2) * wave_frequency) * wave_amplitude
        new_x = x_coords + wave_phase
        new_y = y_coords
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = new_y.astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_skew_horizontal(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        skew_strength = 0.3
        offset = (y_coords - center_y) * skew_strength
        new_x = x_coords + offset
        new_y = y_coords
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = new_y.astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_skew_vertical(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        skew_strength = 0.3
        offset = (x_coords - center_x) * skew_strength
        new_x = x_coords
        new_y = y_coords + offset
        map_x = new_x.astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_rotate_zoom(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        max_dist = min(w_frame, h_frame) // 2
        angle = np.arctan2(dy, dx)
        rotation = 2.0 * np.pi * (dist / max_dist)
        zoom_factor = 1.0 + 0.3 * (dist / max_dist)
        new_angle = angle + rotation
        new_dist = dist * zoom_factor
        new_x = center_x + new_dist * np.cos(new_angle)
        new_y = center_y + new_dist * np.sin(new_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_radial_wave(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        wave_frequency = 0.1
        wave_amplitude = 15.0
        radial_wave = np.sin(dist * wave_frequency) * wave_amplitude
        new_angle = angle + radial_wave / np.maximum(dist, 1)
        new_x = center_x + dist * np.cos(new_angle)
        new_y = center_y + dist * np.sin(new_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_zoom_in(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        zoom_factor = 1.3
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_zoom_out(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        zoom_factor = 0.8
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_fast_zoom_in(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        fps = 30.0
        animation_speed = 2.0
        zoom_factor = 1.0 + (frame_count / fps * animation_speed) % 2.0
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_fast_zoom_out(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        fps = 30.0
        animation_speed = 2.0
        zoom_factor = 1.5 - (frame_count / fps * animation_speed) % 1.0
        zoom_factor = max(0.5, zoom_factor)
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_shake(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        fps = 30.0
        animation_speed = 20.0
        shake_amount = 15.0
        offset_x = shake_amount * np.sin(frame_count / fps * animation_speed * 2 * np.pi)
        offset_y = shake_amount * np.cos(frame_count / fps * animation_speed * 2 * np.pi)
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        return cv2.warpAffine(frame, M, (w_frame, h_frame), borderMode=cv2.BORDER_REPLICATE)
    
    def apply_pulse(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        fps = 30.0
        animation_speed = 3.0
        animation_cycle = (frame_count / fps * animation_speed * 2 * np.pi) % (2 * np.pi)
        zoom_factor = 1.0 + 0.15 * np.sin(animation_cycle)
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_spiral_zoom(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        fps = 30.0
        animation_speed = 2.0
        animation_cycle = (frame_count / fps * animation_speed * 2 * np.pi) % (2 * np.pi)
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        angle = np.arctan2(dy, dx)
        max_dist = np.sqrt(center_x*center_x + center_y*center_y)
        zoom_factor = 1.0 + 0.3 * np.sin(dist / max_dist * 4 * np.pi + animation_cycle)
        new_angle = angle + animation_cycle * 0.5
        new_dist = dist / zoom_factor
        new_x = center_x + new_dist * np.cos(new_angle)
        new_y = center_y + new_dist * np.sin(new_angle)
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_puzzle(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        result = np.zeros_like(frame)
        
        puzzle_cols = 10
        puzzle_rows = 8
        piece_w = w_frame // puzzle_cols
        piece_h = h_frame // puzzle_rows
        
        np.random.seed(42)
        piece_map = list(range(puzzle_rows * puzzle_cols))
        np.random.shuffle(piece_map)
        
        rotations = []
        offsets = []
        for i in range(puzzle_rows * puzzle_cols):
            rotation = np.random.choice([0, 90, 180, 270])
            offset_x = np.random.randint(-piece_w // 3, piece_w // 3)
            offset_y = np.random.randint(-piece_h // 3, piece_h // 3)
            rotations.append(rotation)
            offsets.append((offset_x, offset_y))
        
        piece_idx = 0
        for row in range(puzzle_rows):
            for col in range(puzzle_cols):
                original_idx = piece_map[piece_idx]
                orig_row = original_idx // puzzle_cols
                orig_col = original_idx % puzzle_cols
                
                x_start = orig_col * piece_w
                y_start = orig_row * piece_h
                x_end = min(x_start + piece_w, w_frame)
                y_end = min(y_start + piece_h, h_frame)
                
                piece = frame[y_start:y_end, x_start:x_end].copy()
                original_h, original_w = piece.shape[:2]
                
                rotation = rotations[piece_idx]
                if rotation == 90:
                    piece = cv2.rotate(piece, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    piece = cv2.rotate(piece, cv2.ROTATE_180)
                elif rotation == 270:
                    piece = cv2.rotate(piece, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                offset_x, offset_y = offsets[piece_idx]
                
                target_x = col * piece_w + offset_x
                target_y = row * piece_h + offset_y
                
                piece_h_rot, piece_w_rot = piece.shape[:2]
                
                target_x_end = target_x + piece_w_rot
                target_y_end = target_y + piece_h_rot
                
                crop_x_start = 0
                crop_y_start = 0
                crop_x_end = piece_w_rot
                crop_y_end = piece_h_rot
                
                if target_x < 0:
                    crop_x_start = -target_x
                    target_x = 0
                if target_y < 0:
                    crop_y_start = -target_y
                    target_y = 0
                if target_x_end > w_frame:
                    crop_x_end = piece_w_rot - (target_x_end - w_frame)
                    target_x_end = w_frame
                if target_y_end > h_frame:
                    crop_y_end = piece_h_rot - (target_y_end - h_frame)
                    target_y_end = h_frame
                
                if crop_x_end > crop_x_start and crop_y_end > crop_y_start:
                    piece_crop = piece[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    if piece_crop.shape[0] > 0 and piece_crop.shape[1] > 0:
                        result_h = target_y_end - target_y
                        result_w = target_x_end - target_x
                        if piece_crop.shape[0] == result_h and piece_crop.shape[1] == result_w:
                            result[target_y:target_y_end, target_x:target_x_end] = piece_crop
                
                piece_idx += 1
        
        return result
    
    def apply_extreme_closeup(self, frame: np.ndarray, face: Tuple[int, int, int, int], frame_count: int = 0) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        
        fps = 30.0
        animation_speed = 1.5
        animation_cycle = (frame_count / fps * animation_speed * 2 * np.pi) % (2 * np.pi)
        
        zoom_factor = 1.0 + 2.5 * (0.5 + 0.5 * np.sin(animation_cycle))
        zoom_factor = max(1.0, min(zoom_factor, 4.0))
        
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        new_x = center_x + dx / zoom_factor
        new_y = center_y + dy / zoom_factor
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "EXTREME CLOSE-UP"
        scale_factor = min(w_frame / 1280.0, h_frame / 720.0)
        font_scale = max(1.0, 1.5 * scale_factor)
        thickness = max(2, int(3 * scale_factor))
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w_frame - text_width) // 2
        text_y = h_frame - int(50 * scale_factor)
        
        cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return result
    
    def apply_rotate(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        rotation_angle = np.pi / 4
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        new_x = center_x + dx * cos_a - dy * sin_a
        new_y = center_y + dx * sin_a + dy * cos_a
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_rotate_45(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        return self.apply_rotate(frame, face)
    
    def apply_rotate_90(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h_frame, w_frame = frame.shape[:2]
        center_x = w_frame // 2
        center_y = h_frame // 2
        y_coords, x_coords = np.meshgrid(np.arange(h_frame), np.arange(w_frame), indexing='ij')
        dx = x_coords - center_x
        dy = y_coords - center_y
        rotation_angle = np.pi / 2
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        new_x = center_x + dx * cos_a - dy * sin_a
        new_y = center_y + dx * sin_a + dy * cos_a
        map_x = np.clip(new_x, 0, w_frame - 1).astype(np.float32)
        map_y = np.clip(new_y, 0, h_frame - 1).astype(np.float32)
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    def apply_flip_horizontal(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        return cv2.flip(frame, 1)
    
    def apply_flip_vertical(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        return cv2.flip(frame, 0)
    
    def apply_flip_both(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        return cv2.flip(frame, -1)
    
    def apply_quad_mirror(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        top_left = frame[:h//2, :w//2]
        top_right = cv2.flip(top_left, 1)
        bottom_left = cv2.flip(top_left, 0)
        bottom_right = cv2.flip(top_left, -1)
        top = np.hstack([top_left, top_right])
        bottom = np.hstack([bottom_left, bottom_right])
        return np.vstack([top, bottom])
    
    def apply_tile(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//4, h//4))
        tiled = np.tile(small, (4, 4, 1))
        return tiled[:h, :w]
    
    def apply_radial_tile(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w//3, h//3))
        result = np.zeros_like(frame)
        for i in range(3):
            for j in range(3):
                y_start = i * (h // 3)
                y_end = min((i + 1) * (h // 3), h)
                x_start = j * (w // 3)
                x_end = min((j + 1) * (w // 3), w)
                result[y_start:y_end, x_start:x_end] = small[:y_end-y_start, :x_end-x_start]
        return result
        
    def run(self, filter_type: str, show_preview: bool = True, backend: Optional[str] = None, preview_only: bool = False):
        filter_funcs = {
            'bulge': self.apply_bulge,
            'stretch': self.apply_stretch,
            'swirl': self.apply_swirl,
            'fisheye': self.apply_fisheye,
            'pinch': self.apply_pinch,
            'wave': self.apply_wave,
            'mirror': self.apply_mirror_split,
            'sam_face_mask': self.apply_sam_face_mask,
            'twirl': self.apply_twirl,
            'ripple': self.apply_ripple,
            'sphere': self.apply_sphere,
            'tunnel': self.apply_tunnel,
            'water_ripple': self.apply_water_ripple,
            'radial_blur': self.apply_radial_blur,
            'cylinder': self.apply_cylinder,
            'barrel': self.apply_barrel,
            'pincushion': self.apply_pincushion,
            'whirlpool': self.apply_whirlpool,
            'radial_zoom': self.apply_radial_zoom,
            'concave': self.apply_concave,
            'convex': self.apply_convex,
            'spiral': self.apply_spiral,
            'radial_stretch': self.apply_radial_stretch,
            'radial_compress': self.apply_radial_compress,
            'vertical_wave': self.apply_vertical_wave,
            'horizontal_wave': self.apply_horizontal_wave,
            'skew_horizontal': self.apply_skew_horizontal,
            'skew_vertical': self.apply_skew_vertical,
            'rotate_zoom': self.apply_rotate_zoom,
            'radial_wave': self.apply_radial_wave,
            'zoom_in': self.apply_zoom_in,
            'zoom_out': self.apply_zoom_out,
            'fast_zoom_in': self.apply_fast_zoom_in,
            'fast_zoom_out': self.apply_fast_zoom_out,
            'shake': self.apply_shake,
            'pulse': self.apply_pulse,
            'spiral_zoom': self.apply_spiral_zoom,
            'extreme_closeup': self.apply_extreme_closeup,
            'puzzle': self.apply_puzzle,
            'rotate': self.apply_rotate,
            'rotate_45': self.apply_rotate_45,
            'rotate_90': self.apply_rotate_90,
            'flip_horizontal': self.apply_flip_horizontal,
            'flip_vertical': self.apply_flip_vertical,
            'flip_both': self.apply_flip_both,
            'quad_mirror': self.apply_quad_mirror,
            'tile': self.apply_tile,
            'radial_tile': self.apply_radial_tile,
            'zoom_blur': self.apply_zoom_blur,
            'melt': self.apply_melt,
            'kaleidoscope': self.apply_kaleidoscope,
            'glitch': self.apply_glitch,
            'double_vision': self.apply_double_vision,
            'sam_reich': self.apply_sam_reich_tattoo,
            'sam_face_mask': self.apply_sam_face_mask,
            'black_white': self.apply_black_white,
            'sepia': self.apply_sepia,
            'vintage': self.apply_vintage,
            'neon_glow': self.apply_neon_glow,
            'pixelate': self.apply_pixelate,
            'blur': self.apply_blur,
            'sharpen': self.apply_sharpen,
            'emboss': self.apply_emboss,
            'red_tint': self.apply_red_tint,
            'blue_tint': self.apply_blue_tint,
            'green_tint': self.apply_green_tint,
            'rainbow': self.apply_rainbow,
            'negative': self.apply_negative,
            'posterize': self.apply_posterize,
            'sketch': self.apply_sketch,
            'cartoon': self.apply_cartoon,
            'thermal': self.apply_thermal,
            'ice': self.apply_ice,
            'ocean': self.apply_ocean,
            'plasma': self.apply_plasma,
            'jet': self.apply_jet,
            'turbo': self.apply_turbo,
            'inferno': self.apply_inferno,
            'magma': self.apply_magma,
            'viridis': self.apply_viridis,
            'cool': self.apply_cool,
            'hot': self.apply_hot,
            'spring': self.apply_spring,
            'summer': self.apply_summer,
            'autumn': self.apply_autumn,
            'winter': self.apply_winter,
            'rainbow_shift': self.apply_rainbow_shift,
            'acid_trip': self.apply_acid_trip,
            'vhs': self.apply_vhs,
            'retro': self.apply_retro,
            'cyberpunk': self.apply_cyberpunk,
            'anime': self.apply_anime,
            'glow': self.apply_glow,
            'solarize': self.apply_solarize,
            'edge_detect': self.apply_edge_detect,
            'halftone': self.apply_halftone,
        }
        
        if filter_type not in filter_funcs:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        filter_func = filter_funcs[filter_type]
        
        cam = None
        virtual_cam_available = False
        
        if not preview_only:
            backends_to_try = []
            if backend:
                backends_to_try = [backend]
            else:
                backends_to_try = ['obs', 'unity', 'v4l2loopback']
            
            for backend_name in backends_to_try:
                try:
                    cam = pyvirtualcam.Camera(
                        width=self.width, 
                        height=self.height, 
                        fps=self.fps,
                        backend=backend_name
                    )
                    cam.__enter__()
                    print(f"Virtual camera started with '{backend_name}' backend: {cam.device}")
                    virtual_cam_available = True
                    break
                except Exception as e:
                    if backend:
                        raise RuntimeError(f"Failed to start virtual camera with '{backend_name}' backend: {e}")
                    continue
            
            if not virtual_cam_available:
                print("Warning: No virtual camera backend available.")
                print("Running in preview-only mode. You can capture this window in OBS using Window Capture.")
                print("To enable virtual camera:")
                print("  - macOS: Install OBS 30.0+ and enable Virtual Camera in OBS")
                print("  - Or use: --preview-only flag for testing without virtual camera")
                show_preview = True
        else:
            print("Running in preview-only mode (no virtual camera).")
            show_preview = True
        
        try:
            print(f"Running {filter_type} filter. Press 'q' to quit.")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                face = self.detect_face(frame)
                if face:
                    frame = filter_func(frame, face)
                
                if virtual_cam_available and cam:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.send(frame_rgb)
                    cam.sleep_until_next_frame()
                
                if show_preview:
                    cv2.imshow('WesWorld FX - Preview', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping filter...")
        finally:
            if cam and virtual_cam_available:
                cam.__exit__(None, None, None)
            if show_preview:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='WesWorld FX - Face distortion filters for OBS')
    parser.add_argument('filter', choices=['bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror'],
                       help='Filter type to apply')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--fps', type=int, default=30, help='FPS (default: 30)')
    parser.add_argument('--preview', action='store_true', help='Show preview window (default: True)')
    parser.add_argument('--no-preview', action='store_true', help='Hide preview window')
    parser.add_argument('--preview-only', action='store_true', help='Preview window only, no virtual camera')
    parser.add_argument('--backend', type=str, choices=['obs', 'unity', 'v4l2loopback'], 
                       help='Virtual camera backend (default: auto-detect)')
    
    args = parser.parse_args()
    
    show_preview = not args.no_preview
    
    try:
        with FaceFilter(width=args.width, height=args.height, fps=args.fps) as filter_app:
            filter_app.run(args.filter, show_preview=show_preview, backend=args.backend, preview_only=args.preview_only)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

