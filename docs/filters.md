# WesWorld FX - Filter Reference

## Available Filters

### Bulge
Creates a bulging effect that expands the face outward from the center. The distortion is strongest at the center and fades toward the edges.

**Effect**: Face appears to bulge outward
**Use Case**: Comic exaggeration, fun distortion

### Stretch
Horizontally stretches the face while vertically compressing it, creating a wide, flattened appearance.

**Effect**: Face becomes wider and shorter
**Use Case**: Exaggerated width distortion

### Swirl
Applies a rotational swirl effect centered on the face. The rotation is strongest at the center and decreases toward the edges.

**Effect**: Face appears to be rotating/swirling
**Use Case**: Dynamic, disorienting effect

### Fisheye
Simulates a fisheye lens distortion, creating a barrel distortion effect that curves straight lines.

**Effect**: Face appears curved and distorted like through a fisheye lens
**Use Case**: Wide-angle lens simulation

### Pinch
Creates an inward pinching effect, pulling the edges of the face toward the center.

**Effect**: Face appears pinched inward from edges
**Use Case**: Inward distortion effect

### Wave
Applies a horizontal wave distortion that creates a rippling effect across the face.

**Effect**: Face appears wavy/rippled
**Use Case**: Water-like distortion

### Mirror
Creates a mirror split effect by mirroring the left half of the face to the right side.

**Effect**: Symmetrical face with mirrored left side
**Use Case**: Symmetry effects, artistic distortion

## Filter Selection Guide

- **Subtle Effects**: Bulge, Pinch
- **Strong Distortion**: Stretch, Swirl
- **Lens Effects**: Fisheye
- **Dynamic Effects**: Wave, Swirl
- **Artistic Effects**: Mirror

## Performance Notes

All filters run in real-time at 30 FPS (configurable). Face detection adds minimal overhead. Filters using radial calculations (bulge, swirl, fisheye, pinch) may have slightly higher CPU usage than linear transformations (stretch, wave, mirror).

