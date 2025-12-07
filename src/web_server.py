import asyncio
import base64
import cv2
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from typing import Optional
from face_filters import FaceFilter
import time
from collections import deque

app = FastAPI()

# Store active connections and their filter state
active_connections = {}

# Get all available filters organized by category
def get_all_filters():
    return [
        # DROPOUT
        'dropout_logo', 'dropout_face_mask_sam', 'dropout_face_mask_ariel',
        # Distortion
        'bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror',
        'twirl', 'ripple', 'sphere', 'tunnel', 'water_ripple',
        'radial_blur', 'cylinder', 'barrel', 'pincushion', 'whirlpool', 'radial_zoom',
        'concave', 'convex', 'spiral', 'radial_stretch', 'radial_compress',
        'vertical_wave', 'horizontal_wave', 'skew_horizontal', 'skew_vertical',
        'rotate_zoom', 'radial_wave', 'zoom_in', 'zoom_out', 'fast_zoom_in',
        'fast_zoom_out', 'shake', 'pulse', 'spiral_zoom', 'extreme_closeup',
        'puzzle', 'rotate', 'rotate_45', 'rotate_90', 'flip_horizontal',
        'flip_vertical', 'flip_both', 'quad_mirror', 'tile', 'radial_tile',
        'zoom_blur', 'melt', 'kaleidoscope', 'glitch', 'double_vision',
        # Color & Style
        'black_white', 'sepia', 'vintage', 'neon_glow',
        'pixelate', 'blur', 'sharpen', 'emboss', 'red_tint', 'blue_tint',
        'green_tint', 'rainbow', 'negative', 'posterize', 'sketch', 'cartoon',
        'thermal', 'ice', 'ocean', 'plasma', 'jet', 'turbo', 'inferno',
        'magma', 'viridis', 'cool', 'hot', 'spring', 'summer', 'autumn',
        'winter', 'rainbow_shift', 'acid_trip', 'vhs', 'retro', 'cyberpunk',
        'anime', 'glow', 'solarize', 'edge_detect', 'halftone'
    ]

# Get filters organized by category for UI grouping
def get_filters_by_category():
    return {
        'DROPOUT': ['dropout_logo', 'dropout_face_mask_sam', 'dropout_face_mask_ariel'],
        'Distortion': ['bulge', 'stretch', 'swirl', 'fisheye', 'pinch', 'wave', 'mirror',
                      'twirl', 'ripple', 'sphere', 'tunnel', 'water_ripple',
                      'radial_blur', 'cylinder', 'barrel', 'pincushion', 'whirlpool', 'radial_zoom',
                      'concave', 'convex', 'spiral', 'radial_stretch', 'radial_compress',
                      'vertical_wave', 'horizontal_wave', 'skew_horizontal', 'skew_vertical',
                      'rotate_zoom', 'radial_wave', 'zoom_in', 'zoom_out', 'fast_zoom_in',
                      'fast_zoom_out', 'shake', 'pulse', 'spiral_zoom', 'extreme_closeup',
                      'puzzle', 'rotate', 'rotate_45', 'rotate_90', 'flip_horizontal',
                      'flip_vertical', 'flip_both', 'quad_mirror', 'tile', 'radial_tile',
                      'zoom_blur', 'melt', 'kaleidoscope', 'glitch', 'double_vision'],
        'Color & Style': ['black_white', 'sepia', 'vintage', 'neon_glow',
                         'pixelate', 'blur', 'sharpen', 'emboss', 'red_tint', 'blue_tint',
                         'green_tint', 'rainbow', 'negative', 'posterize', 'sketch', 'cartoon',
                         'thermal', 'ice', 'ocean', 'plasma', 'jet', 'turbo', 'inferno',
                         'magma', 'viridis', 'cool', 'hot', 'spring', 'summer', 'autumn',
                         'winter', 'rainbow_shift', 'acid_trip', 'vhs', 'retro', 'cyberpunk',
                         'anime', 'glow', 'solarize', 'edge_detect', 'halftone']
    }

# Serve static files if static directory exists
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve themes directory
themes_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'themes')
if os.path.exists(themes_dir):
    app.mount("/themes", StaticFiles(directory=themes_dir), name="themes")

# Serve assets directory
assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

@app.get("/api/themes")
async def get_themes():
    """Get list of available themes"""
    themes = []
    if os.path.exists(themes_dir):
        for file in os.listdir(themes_dir):
            if file.endswith('.json'):
                theme_name = file[:-5]  # Remove .json extension
                themes.append(theme_name)
    return {"themes": themes}

@app.get("/api/themes/{theme_name}")
async def get_theme(theme_name: str):
    """Get theme configuration"""
    theme_path = os.path.join(themes_dir, f"{theme_name}.json")
    if os.path.exists(theme_path):
        with open(theme_path, 'r') as f:
            return json.load(f)
    return {"error": "Theme not found"}

@app.get("/")
async def get_index():
    """Serve the main HTML page"""
    html_file = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    if os.path.exists(html_file):
        return FileResponse(html_file)
    else:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>WesWorld FX - Web</title>
            <style>
                body { margin: 0; padding: 0; background: #000; overflow: hidden; }
                #videoContainer { width: 100vw; height: 100vh; display: flex; align-items: center; justify-content: center; }
                #videoFeed { max-width: 100%; max-height: 100%; object-fit: contain; }
                #filterSelect { position: fixed; top: 10px; left: 10px; z-index: 1000; padding: 10px; background: rgba(0,0,0,0.7); color: white; border: 1px solid #333; }
                #status { position: fixed; top: 10px; right: 10px; z-index: 1000; padding: 10px; background: rgba(0,0,0,0.7); color: white; }
            </style>
        </head>
        <body>
            <div id="filterSelect">
                <label>Filter: <select id="filter"></select></label>
            </div>
            <div id="status">Connecting...</div>
            <div id="videoContainer">
                <video id="videoFeed" autoplay playsinline></video>
            </div>
            <script>
                // This will be replaced by the actual static file
                console.log("Please create static/index.html");
            </script>
        </body>
        </html>
        """)

@app.get("/api/filters")
async def get_filters():
    """Get list of all available filters"""
    filters = get_all_filters()
    return {"filters": filters}

@app.get("/api/filters/categories")
async def get_filters_categories():
    """Get filters organized by category"""
    categories = get_filters_by_category()
    return {"categories": categories}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize connection state
    connection_id = id(websocket)
    active_connections[connection_id] = {
        'websocket': websocket,
        'filter': None,
        'filter_app': None,  # Will be created when needed
        'frame_count': 0,
        'last_frame_time': 0,
        'frame_queue': deque(maxlen=2),  # Limit queue to prevent delay buildup
    }
    
    # Initialize filter app (without camera, just for processing)
    try:
        filter_app = FaceFilter()
        # Don't call __enter__ since we're not using the camera
        active_connections[connection_id]['filter_app'] = filter_app
    except Exception as e:
        print(f"Error initializing filter app: {e}")
        await websocket.close()
        del active_connections[connection_id]
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get('type')
            
            if msg_type == 'filter':
                # Update filter selection immediately
                filter_name = message.get('filter')
                print(f"Received filter change request: {filter_name}")
                if filter_name in get_all_filters() or filter_name is None:
                    active_connections[connection_id]['filter'] = filter_name
                    # Reset frame count for animated filters
                    active_connections[connection_id]['frame_count'] = 0
                    # Clear frame queue to apply filter immediately
                    active_connections[connection_id]['frame_queue'].clear()
                    print(f"Filter set to: {filter_name or 'None'}")
                    await websocket.send_json({'type': 'status', 'message': f'Filter set to: {filter_name or "None"}'})
                else:
                    print(f"Warning: Filter '{filter_name}' not in allowed filters list")
            
            elif msg_type == 'frame':
                # Skip if queue is full (prevents delay buildup)
                conn = active_connections[connection_id]
                if len(conn['frame_queue']) >= 2:
                    # Drop oldest frame, process newest
                    try:
                        conn['frame_queue'].popleft()
                    except:
                        pass
                
                # Process video frame
                frame_data = message.get('data')
                if not frame_data:
                    continue
                
                # Add to queue for async processing
                conn['frame_queue'].append(frame_data)
                
                # Process most recent frame
                if len(conn['frame_queue']) > 0:
                    latest_frame_data = conn['frame_queue'][-1]
                    
                    try:
                        # Decode base64 image (optimized)
                        if ',' in latest_frame_data:
                            image_bytes = base64.b64decode(latest_frame_data.split(',')[1])
                        else:
                            image_bytes = base64.b64decode(latest_frame_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            continue
                        
                        # Skip processing if frame is too small (likely corrupted)
                        if frame.shape[0] < 10 or frame.shape[1] < 10:
                            continue
                        
                        # Apply filter if selected
                        filter_name = conn.get('filter')
                        filter_app = conn['filter_app']
                        frame_count = conn['frame_count']
                        
                        if filter_name:
                            # Debug logging
                            print(f"Processing frame with filter: {filter_name} (connection: {connection_id})")
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
                            
                            face_tracking_filters = {'dropout_logo', 'dropout_face_mask_sam', 'dropout_face_mask_ariel'}
                            
                            # Parse hierarchical filter names: <assets_folder>_<fx_type>_<fx_option>
                            # Example: dropout_face_mask_ariel -> folder: dropout, type: face_mask, option: ariel
                            def parse_filter_name(filter_name):
                                """Parse hierarchical filter name into components"""
                                parts = filter_name.split('_')
                                if len(parts) >= 3:
                                    # Format: <folder>_<type>_<option>
                                    folder = parts[0]
                                    fx_type = '_'.join(parts[1:-1])  # Handle multi-word types
                                    option = parts[-1]
                                    return folder, fx_type, option
                                return None, None, None
                            
                            # Face mask filters that use assets
                            face_mask_assets = {}
                            for fx_filter_name in face_tracking_filters:
                                if 'face_mask' in fx_filter_name:
                                    folder, fx_type, option = parse_filter_name(fx_filter_name)
                                    if folder and fx_type == 'face_mask' and option:
                                        # Map to asset directory based on hierarchical structure
                                        # Format: <folder>_<fx_type>_<option> -> assets/<folder>/<fx_type>/<option>.png
                                        if folder == 'dropout':
                                            asset_dir = 'assets/dropout/face_mask'
                                        elif folder == 'assets':
                                            asset_dir = 'assets/face_mask'
                                        else:
                                            asset_dir = f'assets/{folder}/face_mask'
                                        face_mask_assets[fx_filter_name] = (option, asset_dir)
                                        print(f"[BUILD] Mapped filter '{fx_filter_name}' -> asset '{option}' from '{asset_dir}'")
                            
                            try:
                                if filter_name in face_tracking_filters:
                                    if filter_name == 'dropout_logo':
                                        face = filter_app.detect_face(frame)
                                        if face:
                                            frame = filter_app.apply_dropout_logo(frame.copy(), face)
                                    elif filter_name in face_mask_assets:
                                        # Use dynamic asset loading - same processing for all face masks
                                        asset_name, asset_dir = face_mask_assets[filter_name]
                                        print(f"[FILTER DEBUG] Applying face mask: filter='{filter_name}' -> asset='{asset_name}', dir='{asset_dir}'")
                                        print(f"[FILTER DEBUG] face_mask_assets dict: {face_mask_assets}")
                                        faces = filter_app.detect_all_faces(frame)
                                        if faces and len(faces) > 0:
                                            print(f"[FILTER DEBUG] Found {len(faces)} face(s), applying mask '{asset_name}' from '{asset_dir}'...")
                                            # Apply mask to all detected faces
                                            for face in faces:
                                                frame = filter_app.apply_face_mask_from_asset(frame.copy(), face, asset_name, asset_dir=asset_dir)
                                            print(f"[FILTER DEBUG] Face mask '{asset_name}' applied successfully for filter: {filter_name}")
                                        else:
                                            print(f"[FILTER DEBUG] No faces detected for filter: {filter_name}")
                                    else:
                                        print(f"[FILTER DEBUG] Filter '{filter_name}' not in face_mask_assets. Available: {list(face_mask_assets.keys())}")
                                elif filter_name in animated_filters:
                                    dummy_face = (0, 0, frame.shape[1], frame.shape[0])
                                    filter_method = getattr(filter_app, f'apply_{filter_name}', None)
                                    if filter_method and callable(filter_method):
                                        frame = filter_method(frame.copy(), dummy_face, frame_count)
                                elif filter_name in full_image_filters:
                                    dummy_face = (0, 0, frame.shape[1], frame.shape[0])
                                    filter_method = getattr(filter_app, f'apply_{filter_name}', None)
                                    if filter_method and callable(filter_method):
                                        frame = filter_method(frame.copy(), dummy_face)
                                else:
                                    # Try to find filter method by name
                                    filter_method = getattr(filter_app, f'apply_{filter_name}', None)
                                    if filter_method and callable(filter_method):
                                        face = filter_app.detect_face(frame)
                                        if face:
                                            frame = filter_method(frame.copy(), face)
                                    else:
                                        print(f"Warning: Filter method 'apply_{filter_name}' not found")
                            except Exception as e:
                                import traceback
                                print(f"Error applying filter {filter_name}: {e}")
                                traceback.print_exc()
                            
                            # Encode processed frame with lower quality for speed
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            # Send processed frame back immediately
                            await websocket.send_json({
                                'type': 'frame',
                                'data': f'data:image/jpeg;base64,{frame_base64}'
                            })
                            
                            conn['frame_count'] += 1
                            conn['last_frame_time'] = time.time()
                            
                            # Clear queue after successful processing
                            conn['frame_queue'].clear()
                        else:
                            # No filter - send original frame quickly
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            
                            await websocket.send_json({
                                'type': 'frame',
                                'data': f'data:image/jpeg;base64,{frame_base64}'
                            })
                            
                            conn['frame_count'] += 1
                            conn['last_frame_time'] = time.time()
                            conn['frame_queue'].clear()
                        
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        # Remove failed frame from queue
                        if len(conn['frame_queue']) > 0:
                            conn['frame_queue'].pop()
                        continue
                    
    except WebSocketDisconnect:
        print(f"Client disconnected: {connection_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if connection_id in active_connections:
            # No need to call __exit__ since we didn't use __enter__
            del active_connections[connection_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)

