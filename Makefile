.PHONY: help install install-test run-bulge run-stretch run-swirl run-fisheye run-pinch run-wave run-mirror preview-bulge preview-stretch preview-swirl preview-fisheye preview-pinch preview-wave preview-mirror interactive interactive-daemon interactive-daemon-stop interactive-daemon-status interactive-daemon-restart interactive-daemon-logs interactive-daemon-logs-json test clean comparison web web-daemon web-stop web-logs web-status wasm-build wasm-clean wasm-watch wasm-daemon wasm-daemon-stop wasm-daemon-status wasm-daemon-logs dev dev-stop test-e2e test-install build-info setup-hooks

FILTERS = bulge stretch swirl fisheye pinch wave mirror
WIDTH = 1280
HEIGHT = 720
FPS = 30

help:
	@echo "WesWorld FX Commands:"
	@echo ""
	@echo "Installation:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make install-test     - Install test dependencies (Playwright)"
	@echo ""
	@echo "WASM Build (NEW):"
	@echo "  make wasm-build       - Build WASM module"
	@echo "  make wasm-clean       - Clean WASM build artifacts"
	@echo "  make wasm-watch       - Watch WASM files and rebuild on changes (foreground)"
	@echo "  make wasm-daemon      - Start WASM watcher as daemon with auto-rebuild"
	@echo "  make wasm-daemon-stop - Stop WASM watcher daemon"
	@echo "  make wasm-daemon-status - Check WASM watcher status"
	@echo "  make wasm-daemon-logs - View WASM watcher logs (tail -f)"
	@echo ""
	@echo "Development (NEW):"
	@echo "  make dev              - Start web server + WASM watcher (foreground)"
	@echo "  make dev-stop         - Stop all dev services (web + WASM)"
	@echo ""
	@echo "Filters:"
	@echo "  make run-bulge        - Run bulge distortion filter"
	@echo "  make run-stretch      - Run stretch distortion filter"
	@echo "  make run-swirl        - Run swirl distortion filter"
	@echo "  make run-fisheye      - Run fisheye distortion filter"
	@echo "  make run-pinch        - Run pinch distortion filter"
	@echo "  make run-wave         - Run wave distortion filter"
	@echo "  make run-mirror       - Run mirror split filter"
	@echo "  make preview-bulge    - Preview only (no virtual camera)"
	@echo "  make preview-<filter> - Preview any filter (replace <filter> with filter name)"
	@echo ""
	@echo "Interactive Mode:"
	@echo "  make interactive       - Interactive mode: switch filters with 1-7 keys"
	@echo "  make interactive-daemon - Start interactive mode as daemon with auto-reload"
	@echo "  make interactive-daemon-stop - Stop interactive daemon"
	@echo "  make interactive-daemon-status - Check daemon status"
	@echo "  make interactive-daemon-logs - View daemon logs (tail -f)"
	@echo "  make interactive-daemon-logs-json - View JSON logs from logs/ directory"
	@echo ""
	@echo "Web Server:"
	@echo "  make web               - Start web server with hot reload (foreground)"
	@echo "  make web-daemon        - Start web server as daemon with hot reload"
	@echo "  make web-stop          - Stop web server daemon"
	@echo "  make web-logs          - View web server logs (tail -f)"
	@echo "  make web-status        - Check web server status"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Test camera and face detection"
	@echo "  make test-e2e         - Run end-to-end web tests (requires test-install)"
	@echo "  make test-filters     - Run filter processing validation tests"
	@echo "  make validate-filters - Validate filters work (requires web server running)"
	@echo ""
	@echo "Utilities:"
	@echo "  make comparison       - Generate before/after comparison images"
	@echo "  make clean            - Remove Python cache files"
	@echo "  make build-info       - Generate build-info.json from git"
	@echo "  make setup-hooks      - Install git hooks to auto-update build-info.json"
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

build-info:
	@echo "Generating build-info.json from git..."
	@$(PYTHON) scripts/generate_build_info.py

setup-hooks:
	@echo "Setting up git hooks to auto-update build-info.json..."
	@bash scripts/setup_git_hooks.sh

web: build-info
	@echo "Checking for existing server on port 9000..."
	@-lsof -ti:9000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@echo "Starting web server with hot reload..."
	@echo "Open http://localhost:9000 in your browser"
	@echo "Server will auto-reload on file changes"
	@echo "To stop: make web-stop"
	@cd src && $(PYTHON) -m uvicorn web_server:app --host 0.0.0.0 --port 9000 --reload

web-daemon: build-info
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

# WASM Build Commands
wasm-build:
	@echo "Building WASM module..."
	@if [ ! -f wasm/build.sh ]; then \
		echo "❌ Error: wasm/build.sh not found"; \
		exit 1; \
	fi
	@chmod +x wasm/build.sh
	@cd wasm && ./build.sh

wasm-clean:
	@echo "Cleaning WASM build artifacts..."
	@rm -rf wasm/build 2>/dev/null || true
	@rm -f static/wasm/wwfx_module.js static/wasm/wwfx_module.wasm 2>/dev/null || true
	@echo "✅ WASM build artifacts cleaned"

wasm-watch:
	@echo "Starting WASM file watcher (foreground)..."
	@echo "Watching for changes in wasm/src/, wasm/include/, and wasm/CMakeLists.txt"
	@echo "Press Ctrl+C to stop"
	@$(PYTHON) scripts/watch_wasm.py

wasm-daemon:
	@echo "Starting WASM watcher as daemon..."
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "⚠️  WASM watcher already running (PID: $$PID)"; \
			echo "   Stop with: make wasm-daemon-stop"; \
			exit 1; \
		fi; \
	fi
	@echo "Building WASM module initially..."
	@$(MAKE) wasm-build || true
	@echo "Starting watcher daemon..."
	@nohup $(PYTHON) scripts/watch_wasm.py > /tmp/ww_fx_wasm.log 2>&1 & \
	echo $$! > /tmp/ww_fx_wasm.pid && \
	echo "✅ WASM watcher started (PID: $$(cat /tmp/ww_fx_wasm.pid))"
	@echo "To view logs: make wasm-daemon-logs"
	@echo "To stop: make wasm-daemon-stop"

wasm-daemon-stop:
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			kill $$PID && echo "✅ Stopped WASM watcher (PID: $$PID)"; \
		else \
			echo "⚠️  Process $$PID not found"; \
		fi; \
		rm -f /tmp/ww_fx_wasm.pid; \
	else \
		echo "⚠️  No WASM watcher PID file found"; \
	fi

wasm-daemon-status:
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ WASM watcher running (PID: $$PID)"; \
		else \
			echo "⚠️  WASM watcher not running (PID file exists but process not found)"; \
		fi; \
	else \
		echo "⚠️  WASM watcher not running"; \
	fi

wasm-daemon-logs:
	@if [ -f /tmp/ww_fx_wasm.log ]; then \
		tail -f /tmp/ww_fx_wasm.log; \
	else \
		echo "⚠️  No log file found. Start watcher with: make wasm-daemon"; \
	fi

# Development mode: web server + WASM watcher
dev: build-info
	@if [ -f /tmp/ww_fx_wasm.pid ] || [ -f /tmp/web_server.pid ]; then \
		echo "⚠️  Some services may already be running"; \
		echo "   Run 'make dev-stop' first to clean up"; \
		echo ""; \
	fi
	@$(PYTHON) scripts/dev_server.py

dev-stop:
	@echo "Stopping all development services..."
	@$(MAKE) wasm-daemon-stop
	@$(MAKE) web-stop
	@echo "✅ All services stopped"

