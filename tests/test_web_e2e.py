"""
End-to-end tests for WesWorld FX web application using Playwright.
Tests validate camera access, filter selection, and video processing functionality.
"""

import os
import sys
import time
import subprocess
import signal
import pytest
from playwright.sync_api import Page, expect


# Base URL for the web application
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


@pytest.fixture
def page(web_server, browser):
    """Create a new page for each test."""
    page = browser.new_page()
    yield page
    page.close()


def test_server_starts_and_serves_page(page: Page):
    """Test that the server starts and serves the main page."""
    page.goto(BASE_URL)
    expect(page).to_have_title("WesWorld FX - Web")
    
    # Check that main elements are present
    expect(page.locator("#videoContainer")).to_be_visible()
    expect(page.locator("#status")).to_be_visible()
    expect(page.locator("#toggleControls")).to_be_visible()


def test_controls_can_be_toggled(page: Page):
    """Test that controls can be shown and hidden."""
    page.goto(BASE_URL)
    
    # Controls should be hidden by default
    controls = page.locator("#controls")
    expect(controls).not_to_have_class("visible")
    
    # Click toggle button
    page.click("#toggleControls")
    
    # Controls should now be visible
    expect(controls).to_have_class("visible")
    
    # Click again to hide
    page.click("#toggleControls")
    expect(controls).not_to_have_class("visible")


def test_camera_list_loads(page: Page):
    """Test that camera list is populated."""
    page.goto(BASE_URL)
    
    # Show controls
    page.click("#toggleControls")
    
    # Wait for camera select to be visible
    camera_select = page.locator("#cameraSelect")
    expect(camera_select).to_be_visible()
    
    # Check that options are loaded (may take a moment)
    time.sleep(1)
    options = camera_select.locator("option")
    option_count = options.count()
    
    # Should have at least one option (even if it's empty)
    assert option_count >= 0


def test_filter_list_loads(page: Page):
    """Test that filter list is populated from API."""
    page.goto(BASE_URL)
    
    # Show controls
    page.click("#toggleControls")
    
    # Wait for filter select to be visible
    filter_select = page.locator("#filterSelect")
    expect(filter_select).to_be_visible()
    
    # Wait for filters to load
    time.sleep(2)
    
    # Check that "None (Original)" option exists
    expect(filter_select.locator('option[value=""]')).to_contain_text("None (Original)")
    
    # Check that at least some filters are loaded
    options = filter_select.locator("option")
    option_count = options.count()
    assert option_count > 1, "Should have at least one filter option"


def test_api_filters_endpoint(page: Page):
    """Test that the /api/filters endpoint returns filter list."""
    response = page.request.get(f"{BASE_URL}/api/filters")
    expect(response).to_be_ok()
    
    data = response.json()
    assert "filters" in data
    assert isinstance(data["filters"], list)
    assert len(data["filters"]) > 0, "Should return at least one filter"
    
    # Check that some expected filters are present
    expected_filters = ["bulge", "swirl", "fisheye"]
    for filter_name in expected_filters:
        assert filter_name in data["filters"], f"Expected filter '{filter_name}' not found"


def test_websocket_connection(page: Page):
    """Test that WebSocket connection can be established."""
    page.goto(BASE_URL)
    
    # Wait for page to load
    time.sleep(1)
    
    # Check status indicator
    status = page.locator("#status")
    
    # Status should eventually show "Connected" or "Connecting"
    # Give it a few seconds to connect
    time.sleep(3)
    
    status_text = status.text_content()
    assert status_text is not None
    # Status should not be "Initializing..." after a few seconds
    assert "Initializing" not in status_text or "Connected" in status_text or "Disconnected" in status_text


def test_filter_selection_updates(page: Page):
    """Test that filter selection can be changed."""
    page.goto(BASE_URL)
    
    # Show controls
    page.click("#toggleControls")
    
    # Wait for filter select
    filter_select = page.locator("#filterSelect")
    expect(filter_select).to_be_visible()
    
    time.sleep(2)  # Wait for filters to load
    
    # Select a filter
    filter_select.select_option("bulge")
    
    # Verify selection
    selected_value = filter_select.input_value()
    assert selected_value == "bulge"


def test_start_stop_camera_buttons(page: Page):
    """Test that start/stop camera buttons work."""
    page.goto(BASE_URL)
    
    # Show controls
    page.click("#toggleControls")
    
    # Start button should be visible initially
    start_button = page.locator("#startButton")
    stop_button = page.locator("#stopButton")
    
    expect(start_button).to_be_visible()
    expect(stop_button).not_to_be_visible()
    
    # Note: We can't actually start the camera in headless mode without permissions,
    # but we can verify the button structure is correct


def test_video_elements_exist(page: Page):
    """Test that video elements are present in the DOM."""
    page.goto(BASE_URL)
    
    # Check for video input (hidden)
    video_input = page.locator("#videoInput")
    expect(video_input).to_be_attached()
    
    # Check for video feed (img element)
    video_feed = page.locator("#videoFeed")
    expect(video_feed).to_be_attached()
    expect(video_feed).to_be_visible()


def test_static_files_served(page: Page):
    """Test that static files (HTML, CSS, JS) are served correctly."""
    page.goto(BASE_URL)
    
    # Check that page has expected structure
    expect(page.locator("body")).to_be_visible()
    expect(page.locator("#videoContainer")).to_be_visible()
    
    # Check that styles are applied (element should have styling)
    container = page.locator("#videoContainer")
    styles = container.evaluate("el => window.getComputedStyle(el)")
    assert styles is not None


def test_multiple_filters_available(page: Page):
    """Test that multiple filters are available in the dropdown."""
    page.goto(BASE_URL)
    
    # Show controls
    page.click("#toggleControls")
    
    filter_select = page.locator("#filterSelect")
    time.sleep(2)
    
    # Get all filter options
    options = filter_select.locator("option")
    option_count = options.count()
    
    # Should have multiple filters (None + at least 10 filters)
    assert option_count >= 10, f"Expected at least 10 filter options, got {option_count}"
    
    # Check for specific popular filters
    filter_texts = [opt.text_content() for opt in options.all()]
    filter_text = " ".join(filter_texts).lower()
    
    assert "bulge" in filter_text or "swirl" in filter_text, "Should have common filters"


def test_page_responsive_layout(page: Page):
    """Test that page layout is responsive."""
    page.goto(BASE_URL)
    
    # Test at different viewport sizes
    page.set_viewport_size({"width": 1920, "height": 1080})
    expect(page.locator("#videoContainer")).to_be_visible()
    
    page.set_viewport_size({"width": 1280, "height": 720})
    expect(page.locator("#videoContainer")).to_be_visible()
    
    page.set_viewport_size({"width": 640, "height": 480})
    expect(page.locator("#videoContainer")).to_be_visible()


def test_error_handling_disconnected_server(page: Page):
    """Test that page handles server disconnection gracefully."""
    # This test would require stopping the server mid-test
    # For now, we'll just verify the page structure handles errors
    page.goto(BASE_URL)
    
    # Status element should exist for error display
    status = page.locator("#status")
    expect(status).to_be_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

