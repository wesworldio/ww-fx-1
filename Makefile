.PHONY: help install run-bulge run-stretch run-swirl run-fisheye run-pinch run-wave run-mirror preview-bulge preview-stretch preview-swirl preview-fisheye preview-pinch preview-wave preview-mirror interactive test clean comparison

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
	@echo "  make test             - Test camera and face detection"
	@echo "  make comparison       - Generate before/after comparison images"
	@echo "  make clean            - Remove Python cache files"
	@echo ""
	@echo "Options:"
	@echo "  WIDTH=1920 HEIGHT=1080 FPS=60 make run-bulge  - Custom resolution/FPS"

PYTHON = python3.11
PIP = pip3

install:
	$(PIP) install -r requirements.txt

run-bulge:
	$(PYTHON) face_filters.py bulge --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-stretch:
	$(PYTHON) face_filters.py stretch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-swirl:
	$(PYTHON) face_filters.py swirl --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-fisheye:
	$(PYTHON) face_filters.py fisheye --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-pinch:
	$(PYTHON) face_filters.py pinch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-wave:
	$(PYTHON) face_filters.py wave --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

run-mirror:
	$(PYTHON) face_filters.py mirror --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview

preview-bulge:
	$(PYTHON) face_filters.py bulge --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-stretch:
	$(PYTHON) face_filters.py stretch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-swirl:
	$(PYTHON) face_filters.py swirl --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-fisheye:
	$(PYTHON) face_filters.py fisheye --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-pinch:
	$(PYTHON) face_filters.py pinch --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-wave:
	$(PYTHON) face_filters.py wave --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

preview-mirror:
	$(PYTHON) face_filters.py mirror --width $(WIDTH) --height $(HEIGHT) --fps $(FPS) --preview-only

interactive:
	$(PYTHON) interactive_filters.py --width $(WIDTH) --height $(HEIGHT) --fps $(FPS)

test:
	$(PYTHON) -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened()); cap.release()"

comparison:
	@echo "Generating comparison images for all filters..."
	@$(PYTHON) generate_comparison.py --all
	@echo "Comparison images generated in docs/ directory"

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

