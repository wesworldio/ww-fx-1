# WesWorld FX - Implementation Details

## Overview

This project implements real-time face distortion filters using OpenCV for face detection and pyvirtualcam for OBS integration. The system detects faces in the camera feed and applies various geometric distortion effects.

## Architecture

### Core Components

1. **FaceFilter Class**: Main application class handling camera capture and filter application
2. **Face Detection**: Uses OpenCV's Haar Cascade classifier for face detection
3. **Distortion Effects**: Multiple filter functions applying geometric transformations
4. **Virtual Camera**: pyvirtualcam integration for OBS compatibility

### Face Detection

Uses OpenCV's pre-trained Haar Cascade classifier (`haarcascade_frontalface_default.xml`) for real-time face detection. The detector identifies the largest face in the frame and returns bounding box coordinates.

### Distortion Techniques

All filters use OpenCV's `remap` function with custom mapping matrices to apply geometric distortions:

- **Bulge**: Radial expansion from center using distance-based scaling
- **Stretch**: Non-uniform scaling (horizontal expansion, vertical compression)
- **Swirl**: Rotational distortion based on distance from center
- **Fisheye**: Radial distortion simulating fisheye lens
- **Pinch**: Inward radial distortion
- **Wave**: Sinusoidal horizontal displacement
- **Mirror**: Horizontal mirroring of left half

### Virtual Camera Integration

pyvirtualcam creates a virtual camera device that OBS can capture. The filtered frames are converted from BGR to RGB and sent to the virtual camera at the specified frame rate.

## Technical Details

### Coordinate Mapping

Each filter generates mapping matrices (`map_x`, `map_y`) that define where each pixel in the output should sample from in the input image. This allows for smooth, real-time distortion effects.

### Performance Considerations

- Face detection runs on every frame but only processes the largest detected face
- Remapping operations use NumPy vectorization for efficiency
- Frame rate is maintained through pyvirtualcam's sleep mechanism

### Dependencies

- `opencv-python`: Computer vision and image processing
- `numpy`: Numerical operations and array manipulation
- `pyvirtualcam`: Virtual camera creation for OBS
- `mediapipe`: Alternative face detection (not currently used, but available)

## Filter Parameters

Each filter has tunable parameters that can be adjusted in the code:

- Bulge strength: `strength = 0.5`
- Stretch factors: `stretch_x = 1.5`, `stretch_y = 0.7`
- Swirl strength: `swirl_strength = 2.0`
- Fisheye strength: `fisheye_strength = 0.8`
- Pinch strength: `pinch_strength = 0.6`
- Wave amplitude/frequency: `wave_amplitude = 20.0`, `wave_frequency = 0.1`

## Extensibility

New filters can be added by:
1. Creating a new `apply_*` method in the FaceFilter class
2. Adding the filter name and function to the `filter_funcs` dictionary
3. Adding a Makefile target and updating help text

