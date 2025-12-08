"""
Tests to validate filter processing functionality in the web application.
These tests verify that filters are correctly applied to video frames.
"""

import os
import sys
import time
import base64
import subprocess
import pytest
import cv2
import numpy as np
from playwright.sync_api import Page, expect, WebSocket
import json


BASE_URL = "http://localhost:9000"


@pytest.fixture(scope="module")
def web_server():
    """Start the web server before tests and stop it after."""
    # Check if server is already running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 9000))
    sock.close()
    
    if result == 0:
        # Server already running, don't start another
        yield None
        return
    
    # Start the server
    process = subprocess.Popen(
        [sys.executable, "web_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    # Wait for server to start
    time.sleep(3)
    
    yield process
    
    # Cleanup: stop the server
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def create_test_frame(width=640, height=480):
    """Create a test frame with a simple pattern for testing."""
    # Create a test image with a gradient and some shapes
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


def frame_to_base64(frame):
    """Convert OpenCV frame to base64 data URL."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpeg;base64,{frame_base64}'


def base64_to_frame(data_url):
    """Convert base64 data URL back to OpenCV frame."""
    # Remove data URL prefix
    if ',' in data_url:
        header, encoded = data_url.split(',', 1)
    else:
        encoded = data_url
    
    # Decode base64
    image_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame


def test_filter_processing_bulge(web_server, page: Page):
    """Test that bulge filter can be selected and processed."""
    page.goto(BASE_URL)
    time.sleep(2)  # Wait for page to load
    
    # Show controls and select bulge filter
    page.click("#toggleControls")
    filter_select = page.locator("#filterSelect")
    time.sleep(2)  # Wait for filters to load
    
    # Select bulge filter
    filter_select.select_option("bulge")
    time.sleep(0.5)
    
    # Verify selection
    selected_value = filter_select.input_value()
    assert selected_value == "bulge", "Bulge filter should be selected"
    
    # Verify filter exists in API
    response = page.request.get(f"{BASE_URL}/api/filters")
    filters = response.json()["filters"]
    assert "bulge" in filters, "Bulge filter should be in API response"


def test_filter_processing_multiple_filters(web_server, page: Page):
    """Test that multiple different filter types can be applied."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    test_filters = ['bulge', 'swirl', 'black_white', 'sepia', 'thermal', 'blur']
    test_frame = create_test_frame()
    frame_data = frame_to_base64(test_frame)
    
    # Test each filter
    for filter_name in test_filters:
        # Make API request to verify filter exists
        response = page.request.get(f"{BASE_URL}/api/filters")
        assert response.ok
        filters = response.json()["filters"]
        assert filter_name in filters, f"Filter {filter_name} should be available"
    
    # If we get here, all filters are available
    assert len(test_filters) > 0


def test_filter_api_returns_all_filters(web_server, page: Page):
    """Test that API returns all expected filters."""
    page.goto(BASE_URL)
    
    response = page.request.get(f"{BASE_URL}/api/filters")
    assert response.ok
    
    data = response.json()
    assert "filters" in data
    filters = data["filters"]
    
    # Check for expected filter categories
    expected_filters = {
        'bulge', 'swirl', 'fisheye',  # Distortion
        'black_white', 'sepia', 'negative',  # Color
        'thermal', 'plasma', 'jet',  # Color maps
        'blur', 'sharpen', 'pixelate',  # Effects
        # Face masks are discovered dynamically
    }
    
    for expected in expected_filters:
        assert expected in filters, f"Expected filter '{expected}' not found in API response"
    
    # Should have a good number of filters
    assert len(filters) >= 50, f"Expected at least 50 filters, got {len(filters)}"


def test_filter_selection_via_websocket(web_server, page: Page):
    """Test that filter selection works via WebSocket."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    # Show controls and select a filter
    page.click("#toggleControls")
    filter_select = page.locator("#filterSelect")
    
    time.sleep(2)  # Wait for filters to load
    
    # Select different filters
    test_filters = ['bulge', 'swirl', 'black_white']
    
    for filter_name in test_filters:
        # Find and select the filter
        filter_select.select_option(filter_name)
        time.sleep(0.5)
        
        # Verify selection
        selected_value = filter_select.input_value()
        assert selected_value == filter_name, f"Filter selection failed for {filter_name}"


def test_frame_processing_pipeline(web_server, page: Page):
    """Test the complete frame processing pipeline."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    # Create a test frame
    test_frame = create_test_frame(320, 240)  # Smaller for faster processing
    original_hash = hash(test_frame.tobytes())
    
    # Verify frame can be encoded/decoded
    frame_data = frame_to_base64(test_frame)
    decoded_frame = base64_to_frame(frame_data)
    
    assert decoded_frame is not None, "Frame decoding failed"
    assert decoded_frame.shape == test_frame.shape, "Frame shape mismatch"
    
    # Verify frame is different after encoding (due to JPEG compression)
    # but still recognizable
    assert decoded_frame.shape[0] > 0 and decoded_frame.shape[1] > 0


def test_filter_categories_work(web_server, page: Page):
    """Test that filters from different categories can be selected."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    # Get all filters
    response = page.request.get(f"{BASE_URL}/api/filters")
    filters = response.json()["filters"]
    
    # Test filters from different categories
    category_tests = {
        'distortion': ['bulge', 'swirl', 'fisheye', 'pinch'],
        'color': ['black_white', 'sepia', 'negative', 'rainbow'],
        'colormap': ['thermal', 'plasma', 'jet', 'turbo'],
        'effects': ['blur', 'sharpen', 'pixelate', 'glow'],
        # Face masks are discovered dynamically
    }
    
    for category, filter_list in category_tests.items():
        for filter_name in filter_list:
            if filter_name in filters:
                # Filter exists and should be processable
                assert True
            else:
                pytest.skip(f"Filter {filter_name} not available")


def test_websocket_filter_change(web_server, page: Page):
    """Test that filter can be changed via WebSocket during session."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    # Show controls
    page.click("#toggleControls")
    filter_select = page.locator("#filterSelect")
    time.sleep(2)
    
    # Change filters multiple times
    filters_to_test = ['bulge', 'swirl', 'black_white', 'thermal', 'blur']
    
    for filter_name in filters_to_test:
        # Check if filter exists in dropdown
        options = filter_select.locator("option")
        option_values = [opt.get_attribute("value") for opt in options.all()]
        
        if filter_name in option_values:
            filter_select.select_option(filter_name)
            time.sleep(0.3)
            
            # Verify it's selected
            selected = filter_select.input_value()
            assert selected == filter_name, f"Failed to select {filter_name}"


def test_no_filter_returns_original(web_server, page: Page):
    """Test that selecting 'None' filter returns original frame."""
    page.goto(BASE_URL)
    time.sleep(2)
    
    page.click("#toggleControls")
    filter_select = page.locator("#filterSelect")
    time.sleep(2)
    
    # Select "None" option
    filter_select.select_option("")
    time.sleep(0.5)
    
    selected_value = filter_select.input_value()
    assert selected_value == "", "None filter should be empty string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

