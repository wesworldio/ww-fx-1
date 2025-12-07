.PHONY: help install install-test run-bulge run-stretch run-swirl run-fisheye run-pinch run-wave run-mirror preview-bulge preview-stretch preview-swirl preview-fisheye preview-pinch preview-wave preview-mirror interactive interactive-daemon interactive-daemon-stop interactive-daemon-status interactive-daemon-restart interactive-daemon-logs interactive-daemon-logs-json test clean comparison web test-e2e test-install

FILTERS = bulge stretch swirl fisheye pinch wave mirror
WIDTH = 1280
HEIGHT = 720
FPS = 30

help:
	@echo "WesWorld FX Commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make run-bulge        - Run bulge distortion filter"
	@echo "  make run-stretch      - Run stretch distortion filter"
	@echo "  make run-swirl        - Run swirl distortion filter"
	@echo "  make run-fisheye      - Run fisheye distortion filter"
	@echo "  make run-pinch        - Run pinch distortion filter"
	@echo "  make run-wave         - Run wave distortion filter"
	@echo "  make run-mirror       - Run mirror split filter"
	@echo "  make preview-bulge    - Preview only (no virtual camera)"
	@echo "  make preview-<filter> - Preview any filter (replace <filter> with filter name)"
	@echo "  make interactive       - Interactive mode: switch filters with 1-7 keys"
	@echo "  make interactive-daemon - Start interactive mode as daemon with auto-reload"
	@echo "  make interactive-daemon-stop - Stop interactive daemon"
	@echo "  make interactive-daemon-status - Check daemon status"
	@echo "  make interactive-daemon-logs - View daemon logs (tail -f)"
	@echo "  make interactive-daemon-logs-json - View JSON logs from logs/ directory"
	@echo "  make web               - Start web server with hot reload (foreground)"
	@echo "  make web-daemon        - Start web server as daemon with hot reload"
	@echo "  make web-stop          - Stop web server daemon"
	@echo "  make web-logs          - View web server logs (tail -f)"
	@echo "  make web-status        - Check web server status"
	@echo "  make test             - Test camera and face detection"
	@echo "  make test-e2e         - Run end-to-end web tests (requires test-install)"
	@echo "  make test-filters     - Run filter processing validation tests"
	@echo "  make validate-filters - Validate filters work (requires web server running)"
	@echo "  make test-install     - Install test dependencies (Playwright)"
	@echo "  make comparison       - Generate before/after comparison images"
	@echo "  make clean            - Remove Python cache files"
	@echo ""
	@echo "Options:"
	@echo "  WIDTH=1920 HEIGHT=1080 FPS=60 make run-bulge  - Custom resolution/FPS"

PYTHON = python3.11
PIP = pip3

install:
	$(PIP) install -r requirements.txt

install-test:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	$(PYTHON) -m playwright install chromium

run-bulge:
	$(PYTHON) src/face_filters.py bulge --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-stretch:
	$(PYTHON) src/face_filters.py stretch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-swirl:
	$(PYTHON) src/face_filters.py swirl --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-fisheye:
	$(PYTHON) src/face_filters.py fisheye --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-pinch:
	$(PYTHON) src/face_filters.py pinch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-wave:
	$(PYTHON) src/face_filters.py wave --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-mirror:
	$(PYTHON) src/face_filters.py mirror --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

preview-bulge:
	$(PYTHON) src/face_filters.py bulge --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-stretch:
	$(PYTHON) src/face_filters.py stretch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-swirl:
	$(PYTHON) src/face_filters.py swirl --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-fisheye:
	$(PYTHON) src/face_filters.py fisheye --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-pinch:
	$(PYTHON) src/face_filters.py pinch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-wave:
	$(PYTHON) src/face_filters.py wave --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-mirror:
	$(PYTHON) src/face_filters.py mirror --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

interactive:
	$(PYTHON) src/interactive_filters.py --width $(WIDTH) --height $(HEIGHT) --fps $(FPS)

interactive-daemon:
	@echo "Starting interactive filters daemon with auto-reload..."
	@echo "To stop: make interactive-daemon-stop"
	@echo "To check status: make interactive-daemon-status"
	@echo "To view logs: make interactive-daemon-logs"
	@$(PYTHON) src/daemon_interactive.py start --width $(WIDTH) --height $(HEIGHT) --fps $(FPS)

interactive-daemon-stop:
	@echo "Stopping interactive filters daemon..."
	@$(PYTHON) src/daemon_interactive.py stop || echo "⚠️  Daemon not running"

interactive-daemon-status:
	@$(PYTHON) src/daemon_interactive.py status

interactive-daemon-restart:
	@echo "Restarting interactive filters daemon..."
	@$(PYTHON) src/daemon_interactive.py restart --width $(WIDTH) --height $(HEIGHT) --fps $(FPS)

interactive-daemon-logs:
	@if [ -f /tmp/ww_fx_interactive.log ]; then \
		tail -f /tmp/ww_fx_interactive.log; \
	else \
		echo "⚠️  No log file found. Start daemon with: make interactive-daemon"; \
	fi

interactive-daemon-logs-json:
	@echo "JSON logs in logs/ directory:"
	@ls -lh logs/*.jsonl 2>/dev/null || echo "No JSON logs found"
	@echo ""
	@echo "View latest logs:"
	@for file in logs/*.jsonl; do \
		if [ -f "$$file" ]; then \
			echo "=== $$file ==="; \
			tail -5 "$$file" | python3 -m json.tool 2>/dev/null || tail -5 "$$file"; \
			echo ""; \
		fi \
	done

web:
	@echo "Checking for existing server on port 9000..."
	@-lsof -ti:9000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "Starting web server with hot reload..."
	@echo "Open http://localhost:9000 in your browser"
	@echo "Server will auto-reload on file changes"
	@echo "To stop: make web-stop"
	@cd src && $(PYTHON) -m uvicorn web_server:app --host 0.0.0.0 --port 9000 --reload

web-daemon:
	@echo "Checking for existing server on port 9000..."
	@-lsof -ti:9000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "Starting web server as daemon with hot reload..."
	@echo "Open http://localhost:9000 in your browser"
	@echo "Server will auto-reload on file changes"
	@echo "Logs: tail -f /tmp/web_server.log"
	@echo "To stop: make web-stop"
	@cd src && nohup $(PYTHON) -m uvicorn web_server:app --host 0.0.0.0 --port 9000 --reload > /tmp/web_server.log 2>&1 & \
	echo $$! > /tmp/web_server.pid && \
	echo "✅ Server started (PID: $$(cat /tmp/web_server.pid))"

web-stop:
	@if [ -f /tmp/web_server.pid ]; then \
		PID=$$(cat /tmp/web_server.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			kill $$PID && echo "✅ Stopped server (PID: $$PID)"; \
		else \
			echo "⚠️  Process $$PID not found"; \
		fi; \
		rm -f /tmp/web_server.pid; \
	else \
		echo "Checking for server on port 9000..."; \
		lsof -ti:9000 | xargs kill -9 2>/dev/null && echo "✅ Stopped server on port 9000" || echo "⚠️  No server running"; \
	fi

web-logs:
	@if [ -f /tmp/web_server.log ]; then \
		tail -f /tmp/web_server.log; \
	else \
		echo "⚠️  No log file found. Start server with: make web-daemon"; \
	fi

web-status:
	@if [ -f /tmp/web_server.pid ]; then \
		PID=$$(cat /tmp/web_server.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ Server running (PID: $$PID)"; \
			curl -s http://localhost:9000/api/filters | $(PYTHON) -c "import sys, json; d=json.load(sys.stdin); print(f'   Filters: {len(d.get(\"filters\", []))}')" 2>/dev/null || echo "   ⚠️  Server not responding"; \
		else \
			echo "⚠️  Server not running (PID file exists but process not found)"; \
		fi; \
	else \
		if lsof -ti:9000 > /dev/null 2>&1; then \
			echo "⚠️  Server running on port 9000 but no PID file"; \
		else \
			echo "⚠️  Server not running"; \
		fi; \
	fi

test-install:
	@echo "Installing test dependencies..."
	$(PIP) install -r requirements-test.txt
	$(PYTHON) -m playwright install chromium
	@echo "Test dependencies installed!"

test-e2e:
	@echo "Running end-to-end web tests..."
	@echo "Make sure the web server is NOT running (tests will start their own instance)"
	@echo ""
	$(PYTHON) -m pytest tests/test_web_e2e.py tests/test_filter_processing.py -v --tb=short

test-e2e-headed:
	@echo "Running end-to-end web tests in headed mode (visible browser)..."
	@echo "Make sure the web server is NOT running (tests will start their own instance)"
	@echo ""
	PLAYWRIGHT_HEADLESS=false $(PYTHON) -m pytest tests/test_web_e2e.py tests/test_filter_processing.py -v --tb=short -s

test-filters:
	@echo "Running filter processing validation tests..."
	@echo "Make sure the web server is NOT running (tests will start their own instance)"
	@echo ""
	$(PYTHON) -m pytest tests/test_filter_processing.py -v --tb=short

validate-filters:
	@echo "Validating filter processing (requires web server to be running)"
	@echo "Start server in another terminal: make web"
	@echo ""
	$(PYTHON) scripts/validate_filters.py

test:
	$(PYTHON) -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"

comparison:
	@echo "Generating comparison images for all filters..."
	@$(PYTHON) src/generate_comparison.py --all
	@echo "Comparison images generated in docs/ directory"

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

