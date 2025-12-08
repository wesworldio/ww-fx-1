#!/usr/bin/env python3
"""
Validation script to test filter processing in the web application.
This script validates that filters can be applied to test frames.
"""

import sys
import os
import cv2
import numpy as np
import requests
import json
import base64
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_filters import FaceFilter


def create_test_frame(width=640, height=480) -> np.ndarray:
    """Create a test frame with a pattern for testing."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(height):
        frame[y, :] = [y * 255 // height, 128, 255 - y * 255 // height]
    
    # Add a circle (simulating a face)
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    cv2.circle(frame, center, radius, (255, 255, 255), -1)
    cv2.circle(frame, center, radius // 2, (0, 0, 0), -1)
    
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert OpenCV frame to base64 data URL."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{frame_base64}'


def test_filter_application(filter_name: str, test_frame: np.ndarray, filter_app: FaceFilter) -> Tuple[bool, str]:
    """Test that a filter can be applied to a frame."""
    try:
        # Determine filter category
        animated_filters = {
            'extreme_closeup', 'puzzle', 'fast_zoom_in', 'fast_zoom_out', 
            'shake', 'pulse', 'spiral_zoom'
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
        
        # Face mask filters are handled dynamically
        frame = test_frame.copy()
        frame_count = 0
        
        if 'face_mask' in filter_name:
            # Parse filter name to extract folder and mask name
            parts = filter_name.split('_')
            if len(parts) >= 3:
                folder = parts[0]
                mask_name = parts[-1]
                if folder == 'dropout':
                    asset_dir = 'assets/dropout/face_mask'
                elif folder == 'assets':
                    asset_dir = 'assets/face_mask'
                else:
                    asset_dir = f'assets/{folder}/face_mask'
                faces = filter_app.detect_all_faces(frame)
                if faces:
                    for face in faces:
                        frame = filter_app.apply_face_mask_from_asset(frame.copy(), face, mask_name, asset_dir=asset_dir)
        elif filter_name in animated_filters:
            dummy_face = (0, 0, frame.shape[1], frame.shape[0])
            filter_method = getattr(filter_app, f'apply_{filter_name}', None)
            if filter_method and callable(filter_method):
                frame = filter_method(frame, dummy_face, frame_count)
            else:
                return False, f"Filter method not found: apply_{filter_name}"
        elif filter_name in full_image_filters:
            dummy_face = (0, 0, frame.shape[1], frame.shape[0])
            filter_method = getattr(filter_app, f'apply_{filter_name}', None)
            if filter_method and callable(filter_method):
                frame = filter_method(frame, dummy_face)
            else:
                return False, f"Filter method not found: apply_{filter_name}"
        else:
            face = filter_app.detect_face(frame)
            if face:
                filter_method = getattr(filter_app, f'apply_{filter_name}', None)
                if filter_method and callable(filter_method):
                    frame = filter_method(frame, face)
                else:
                    return False, f"Filter method not found: apply_{filter_name}"
        
        # Verify frame was processed (should be different or same shape)
        if frame is None:
            return False, "Filter returned None"
        
        if frame.shape != test_frame.shape:
            # Some filters might change shape, but should still be valid
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                return False, f"Filter produced invalid frame shape: {frame.shape}"
        
        return True, "Success"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def validate_filters_api(base_url: str = "http://localhost:9000") -> Tuple[bool, List[str]]:
    """Validate that the filters API endpoint works."""
    try:
        response = requests.get(f"{base_url}/api/filters", timeout=5)
        if response.status_code != 200:
            return False, [f"API returned status {response.status_code}"]
        
        data = response.json()
        if "filters" not in data:
            return False, ["API response missing 'filters' key"]
        
        filters = data["filters"]
        if not isinstance(filters, list):
            return False, ["Filters is not a list"]
        
        if len(filters) == 0:
            return False, ["No filters returned"]
        
        return True, filters
        
    except requests.exceptions.ConnectionError:
        return False, ["Cannot connect to server. Is it running?"]
    except Exception as e:
        return False, [f"Error: {str(e)}"]


def main():
    """Main validation function."""
    print("=" * 60)
    print("WesWorld FX Filter Validation")
    print("=" * 60)
    print()
    
    base_url = "http://localhost:9000"
    
    # Test 1: API endpoint
    print("1. Testing API endpoint...")
    api_ok, result = validate_filters_api(base_url)
    if not api_ok:
        print(f"   ❌ FAILED: {result[0]}")
        print("\n   Please start the web server first:")
        print("   make web")
        return 1
    
    filters = result
    print(f"   ✅ SUCCESS: Found {len(filters)} filters")
    print()
    
    # Test 2: Filter processing
    print("2. Testing filter processing...")
    filter_app = FaceFilter()
    test_frame = create_test_frame(320, 240)
    
    # Test a sample of filters from different categories
    test_filters = [
        'bulge', 'swirl', 'fisheye',  # Distortion
        'black_white', 'sepia', 'negative',  # Color
        'thermal', 'plasma', 'jet',  # Color maps
        'blur', 'sharpen', 'pixelate',  # Effects
    ]
    
    passed = 0
    failed = 0
    failed_filters = []
    
    for filter_name in test_filters:
        if filter_name not in filters:
            print(f"   ⚠️  SKIP: {filter_name} (not in API)")
            continue
        
        success, message = test_filter_application(filter_name, test_frame, filter_app)
        if success:
            print(f"   ✅ {filter_name}: {message}")
            passed += 1
        else:
            print(f"   ❌ {filter_name}: {message}")
            failed += 1
            failed_filters.append(filter_name)
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        print(f"\nFailed filters: {', '.join(failed_filters)}")
        return 1
    
    print("✅ All filter tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

