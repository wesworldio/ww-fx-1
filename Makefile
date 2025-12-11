.PHONY: help build clean watch daemon daemon-stop daemon-status daemon-logs build-info setup-hooks

PYTHON = python3

help:
	@echo "WesWorld FX - Commands:"
	@echo ""
	@echo "Build:"
	@echo "  make build            - Build WASM module"
	@echo "  make clean            - Clean build artifacts"
	@echo "  make watch            - Watch files and rebuild on changes (foreground)"
	@echo "  make daemon           - Start watcher as daemon with auto-rebuild"
	@echo "  make daemon-stop      - Stop watcher daemon"
	@echo "  make daemon-status    - Check watcher status"
	@echo "  make daemon-logs      - View watcher logs (tail -f)"
	@echo ""
	@echo "Utilities:"
	@echo "  make build-info      - Generate build-info.json from git"
	@echo "  make setup-hooks     - Install git hooks to auto-update build-info.json"
	@echo ""
	@echo "Note: This is the WASM-only version. Python backend code has been archived."
	@echo "      See archive/README.md for archived code."

build-info:
	@echo "Generating build-info.json from git..."
	@$(PYTHON) scripts/generate_build_info.py

setup-hooks:
	@echo "Setting up git hooks to auto-update build-info.json..."
	@bash scripts/setup_git_hooks.sh

# Build Commands
build:
	@echo "Building WASM module..."
	@if [ ! -f wasm/build.sh ]; then \
		echo "❌ Error: wasm/build.sh not found"; \
		exit 1; \
	fi
	@chmod +x wasm/build.sh
	@cd wasm && ./build.sh

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf wasm/build 2>/dev/null || true
	@rm -f static/wasm/wwfx_module.js static/wasm/wwfx_module.wasm 2>/dev/null || true
	@echo "✅ Build artifacts cleaned"

watch:
	@echo "Starting file watcher (foreground)..."
	@echo "Watching for changes in wasm/src/, wasm/include/, and wasm/CMakeLists.txt"
	@echo "Press Ctrl+C to stop"
	@$(PYTHON) scripts/watch_wasm.py

daemon:
	@echo "Starting watcher as daemon..."
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "⚠️  Watcher already running (PID: $$PID)"; \
			echo "   Stop with: make daemon-stop"; \
			exit 1; \
		fi; \
	fi
	@echo "Building WASM module initially..."
	@$(MAKE) build || true
	@echo "Starting watcher daemon..."
	@nohup $(PYTHON) scripts/watch_wasm.py > /tmp/ww_fx_wasm.log 2>&1 & \
	echo $$! > /tmp/ww_fx_wasm.pid && \
	echo "✅ Watcher started (PID: $$(cat /tmp/ww_fx_wasm.pid))"
	@echo "To view logs: make daemon-logs"
	@echo "To stop: make daemon-stop"

daemon-stop:
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			kill $$PID && echo "✅ Stopped watcher (PID: $$PID)"; \
		else \
			echo "⚠️  Process $$PID not found"; \
		fi; \
		rm -f /tmp/ww_fx_wasm.pid; \
	else \
		echo "⚠️  No watcher PID file found"; \
	fi

daemon-status:
	@if [ -f /tmp/ww_fx_wasm.pid ]; then \
		PID=$$(cat /tmp/ww_fx_wasm.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ Watcher running (PID: $$PID)"; \
		else \
			echo "⚠️  Watcher not running (PID file exists but process not found)"; \
		fi; \
	else \
		echo "⚠️  Watcher not running"; \
	fi

daemon-logs:
	@if [ -f /tmp/ww_fx_wasm.log ]; then \
		tail -f /tmp/ww_fx_wasm.log; \
	else \
		echo "⚠️  No log file found. Start watcher with: make daemon"; \
	fi
