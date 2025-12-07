import cv2
import numpy as np
import sys
import os
from face_filters import FaceFilter


def create_test_face_image():
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img.fill(50)
    
    center_x, center_y = 640, 360
    face_width, face_height = 400, 500
    
    cv2.ellipse(img, (center_x, center_y), (face_width//2, face_height//2), 0, 0, 360, (220, 200, 180), -1)
    
    eye_y = center_y - 80
    left_eye_x = center_x - 80
    right_eye_x = center_x + 80
    
    cv2.circle(img, (left_eye_x, eye_y), 30, (50, 50, 50), -1)
    cv2.circle(img, (right_eye_x, eye_y), 30, (50, 50, 50), -1)
    cv2.circle(img, (left_eye_x, eye_y), 15, (255, 255, 255), -1)
    cv2.circle(img, (right_eye_x, eye_y), 15, (255, 255, 255), -1)
    
    nose_y = center_y + 20
    cv2.ellipse(img, (center_x, nose_y), (20, 40), 0, 0, 360, (200, 180, 160), -1)
    
    mouth_y = center_y + 120
    cv2.ellipse(img, (center_x, mouth_y), (60, 30), 0, 0, 180, (150, 100, 100), -1)
    
    return img


def capture_frame():
    print("Attempting to capture from camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available, using test image instead.")
        cap.release()
        return create_test_face_image()
    
    print("Capturing frame in 2 seconds... Look at the camera!")
    import time
    time.sleep(2)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("Could not capture frame, using test image instead.")
        return create_test_face_image()
    
    print("Frame captured successfully!")
    return frame


def create_comparison(filter_type: str, original_frame: np.ndarray, output_path: str):
    filter_app = FaceFilter()
    filter_app.__enter__()
    
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
                face = filter_app.detect_face(original_frame)
                if face:
                    filtered_frame = filter_app.apply_sam_reich_tattoo(original_frame.copy(), face)
                else:
                    print("Warning: No face detected for sam_reich filter.")
                    filtered_frame = original_frame.copy()
            elif filter_type == 'sam_face_mask':
                faces = filter_app.detect_all_faces(original_frame)
                if faces:
                    filtered_frame = filter_app.apply_sam_face_mask(original_frame.copy(), faces)
                else:
                    print("Warning: No face detected for sam_face_mask filter.")
                    filtered_frame = original_frame.copy()
            else:
                filtered_frame = original_frame.copy()
        elif filter_type in animated_filters:
            dummy_face = (0, 0, original_frame.shape[1], original_frame.shape[0])
            filter_method = getattr(filter_app, f'apply_{filter_type}', None)
            if filter_method and callable(filter_method):
                filtered_frame = filter_method(original_frame.copy(), dummy_face, 30)
            else:
                print(f"Error: Filter method not found for {filter_type}")
                filter_app.__exit__(None, None, None)
                return False
        elif filter_type in full_image_filters:
            dummy_face = (0, 0, original_frame.shape[1], original_frame.shape[0])
            if filter_type == 'mirror':
                filter_method = filter_app.apply_mirror_split
            else:
                filter_method = getattr(filter_app, f'apply_{filter_type}', None)
            if filter_method and callable(filter_method):
                filtered_frame = filter_method(original_frame.copy(), dummy_face)
            else:
                print(f"Error: Filter method not found for {filter_type}")
                filter_app.__exit__(None, None, None)
                return False
        else:
            face = filter_app.detect_face(original_frame)
            if not face:
                print("Warning: No face detected. Applying filter to center region.")
                h, w = original_frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                face_size = min(w, h) // 3
                face = (center_x - face_size//2, center_y - face_size//2, face_size, face_size)
            
            filter_method = getattr(filter_app, f'apply_{filter_type}', None)
            if filter_method and callable(filter_method):
                filtered_frame = filter_method(original_frame.copy(), face)
            else:
                print(f"Error: Filter method not found for {filter_type}")
                filter_app.__exit__(None, None, None)
                return False
    except Exception as e:
        print(f"Error applying filter {filter_type}: {e}")
        filter_app.__exit__(None, None, None)
        return False
    
    filter_app.__exit__(None, None, None)
    
    h, w = original_frame.shape[:2]
    
    comparison = np.hstack([original_frame, filtered_frame])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)
    
    text_y = 50
    
    cv2.putText(comparison, 'BEFORE', (30, text_y), font, font_scale, shadow_color, thickness + 2)
    cv2.putText(comparison, 'BEFORE', (30, text_y), font, font_scale, color, thickness)
    
    cv2.putText(comparison, 'AFTER', (w + 30, text_y), font, font_scale, shadow_color, thickness + 2)
    cv2.putText(comparison, 'AFTER', (w + 30, text_y), font, font_scale, color, thickness)
    
    filter_text = f'{filter_type.upper()} FILTER'
    text_size = cv2.getTextSize(filter_text, font, 1.2, 2)[0]
    text_x = (comparison.shape[1] - text_size[0]) // 2
    cv2.putText(comparison, filter_text, (text_x, h - 30), font, 1.2, shadow_color, 3)
    cv2.putText(comparison, filter_text, (text_x, h - 30), font, 1.2, (0, 255, 255), 2)
    
    divider_x = w
    cv2.line(comparison, (divider_x, 0), (divider_x, h), (255, 255, 255), 4)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison image saved to: {output_path}")
    return True


def get_all_filters():
    return [
        'bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror',
        'sam_face_mask', 'twirl', 'ripple', 'sphere', 'tunnel', 'water_ripple',
        'radial_blur', 'cylinder', 'barrel', 'pincushion', 'whirlpool',
        'radial_zoom', 'concave', 'convex', 'spiral', 'radial_stretch',
        'radial_compress', 'vertical_wave', 'horizontal_wave', 'skew_horizontal',
        'skew_vertical', 'rotate_zoom', 'radial_wave', 'zoom_in', 'zoom_out',
        'fast_zoom_in', 'fast_zoom_out', 'shake', 'pulse', 'spiral_zoom',
        'extreme_closeup', 'puzzle', 'rotate', 'rotate_45', 'rotate_90',
        'flip_horizontal', 'flip_vertical', 'flip_both', 'quad_mirror', 'tile',
        'radial_tile', 'zoom_blur', 'melt', 'kaleidoscope', 'glitch',
        'double_vision', 'sam_reich', 'black_white', 'sepia', 'vintage',
        'neon_glow', 'pixelate', 'blur', 'sharpen', 'emboss', 'red_tint',
        'blue_tint', 'green_tint', 'rainbow', 'negative', 'posterize', 'sketch',
        'cartoon', 'thermal', 'ice', 'ocean', 'plasma', 'jet', 'turbo',
        'inferno', 'magma', 'viridis', 'cool', 'hot', 'spring', 'summer',
        'autumn', 'winter', 'rainbow_shift', 'acid_trip', 'vhs', 'retro',
        'cyberpunk', 'anime', 'glow', 'solarize', 'edge_detect', 'halftone',
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3.11 generate_comparison.py <filter-type> [output-path]")
        print("Or: python3.11 generate_comparison.py --all")
        print("Available filters: (use --all to see full list)")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        all_filters = get_all_filters()
        print(f"Generating comparison images for all {len(all_filters)} filters...")
        frame = capture_frame()
        if frame is None:
            sys.exit(1)
        
        output_dir = "docs"
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for filter_type in all_filters:
            output_path = os.path.join(output_dir, f"comparison_{filter_type}.jpg")
            print(f"\nProcessing {filter_type}...")
            if create_comparison(filter_type, frame, output_path):
                success_count += 1
        
        print(f"\n\nSuccessfully generated {success_count}/{len(all_filters)} comparison images in {output_dir}/")
        return
    
    filter_type = sys.argv[1].lower()
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"comparison_{filter_type}.jpg"
    
    print(f"Generating comparison for filter: {filter_type}")
    
    frame = capture_frame()
    if frame is None:
        sys.exit(1)
    
    success = create_comparison(filter_type, frame, output_path)
    if not success:
        sys.exit(1)
    
    print(f"\nSuccess! View the comparison at: {output_path}")


if __name__ == '__main__':
    main()
