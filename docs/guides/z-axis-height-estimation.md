# Z-Axis Crane Hook Height Estimation

**Technical Feasibility Report — March 2026**

This document explores how to extract the real-world Z-position (height above ground) of a crane hook from live video, given that we already detect the hook via RF-DETR bounding boxes. It covers the minimum physical data required, the mathematical foundations, three candidate approaches with tradeoffs, accuracy expectations, and a design for an autonomous agent harness that can iterate on the implementation until convergence.

---

## Background & Problem Statement

### What We Have Today

Batman detects crane hooks in video frames using a fine-tuned RF-DETR model. The detection pipeline produces **bounding boxes** in two formats depending on where in the stack you look:

| Layer | Format | Example |
|---|---|---|
| Model output (`RFDETRInference`) | Pixel `xyxy` — `(x1, y1, x2, y2)` | `(412, 180, 530, 340)` |
| API output (`InferenceRunner`) | Normalized center `xywh` — `(cx, cy, w, h)` all in `[0, 1]` | `(0.49, 0.27, 0.06, 0.15)` |

These give us the **2D position** of the hook in the image plane — its pixel column (roughly X-axis) and pixel row (roughly Y-axis). But the image is a flat projection of a 3D scene. We have no information about the third axis: **how high the hook is above the ground**.

### What We Want

For every frame of the live video stream, we want to output:

```json
{
  "frame_idx": 142,
  "detections": [
    {
      "class_name": "crane_hook",
      "bbox": {"x": 0.49, "y": 0.27, "width": 0.06, "height": 0.15},
      "confidence": 0.94,
      "height_above_ground_m": 12.7,
      "height_confidence": 0.87,
      "height_method": "encoder_vision_fusion"
    }
  ]
}
```

### Why This Is Possible

Two properties make this tractable rather than an open research problem:

1. **Known object dimensions** — We can physically measure the crane hook. If you know the real-world size of an object and its apparent size in pixels, the pinhole camera model gives you the distance.
2. **Machine telemetry access** — We "have control of the machine controlling the crane hook," meaning we likely have access to hoist encoder data from the crane's PLC. Encoders directly measure rope payout, which directly gives hook height.

---

## Part I: Physical Data Requirements

### The Three Approaches

We identified three viable approaches, each requiring a different set of physical data:

| Approach | Expected Accuracy (at 20m) | Hardware Cost | Complexity | Best When... |
|---|---|---|---|---|
| **A. Sensor Fusion** (encoder + vision) | ±2–10 mm | Low (PLC already exists) | Medium | You have PLC access |
| **B. Vision-Only** (pinhole camera model) | ±10–40 cm | Low (camera only) | Low–Medium | No PLC access, or as starting point |
| **C. Two-Point Reference** (empirical calibration) | ±10–30 cm (raw), ±5–15 cm (filtered) | Low (camera only) | Very Low | Quick deployment, no calibration expertise |

!!! tip "Our Recommendation"
    **Start with Approach C** (two-point reference calibration) to get a working height estimate in under 10 minutes. It requires zero camera calibration — just two reference images at known heights. Upgrade to **Approach B** if you need higher accuracy or your camera has significant tilt (>15°). If you have PLC/encoder access, **Approach A** gives millimeter precision by fusing encoder data with vision.

---

### Approach A: Sensor Fusion (Encoder + Vision)

This is the recommended approach. It combines two independent measurements of the same quantity (hook height) using a Kalman filter to produce an estimate that is more accurate than either alone.

#### Why Two Sensors Are Better Than One

**Encoders** are excellent at measuring relative changes — they count rotations of the hoist drum with high precision and high frequency (100–1000 Hz). But they suffer from:

- **Rope slip** — Under heavy acceleration or with worn rope, the rope slides on the drum. The encoder thinks the hook moved, but it didn't (or not as much).
- **Rope stretch** — Under load, steel wire rope stretches. A 50m rope under heavy load might stretch 10–30mm. The encoder doesn't see this.
- **Cumulative drift** — Small errors accumulate over hours of operation. After a power cycle, the encoder loses its absolute reference entirely.

**Vision** is excellent at absolute position — if you can see the hook, you know roughly where it is, regardless of what happened before. But it suffers from:

- **Low precision** — Bounding boxes have ±5 pixel uncertainty, which translates to ±10–40 cm at typical distances.
- **Low frequency** — Camera runs at 10–30 Hz vs. encoder's 100–1000 Hz.
- **Environmental sensitivity** — Fog, rain, darkness, and occlusion degrade detection.

**Fusing both** gives you the encoder's precision and update rate with vision's absolute reference and drift correction. When the encoder says the hook is at 12.003m and vision says 12.1m, the Kalman filter trusts the encoder for frame-to-frame motion but periodically pulls it back to vision's absolute reference.

#### Required Data from the Crane PLC

| # | Data Point | What It Is | How to Get It | Why It's Needed |
|---|---|---|---|---|
| 1 | **Hoist encoder position** | Current pulse count or absolute rotation of the hoist drum motor | Read from PLC register via OPC-UA (modern cranes) or Modbus TCP (legacy cranes) | This is the primary Z measurement — drum rotation directly corresponds to rope payout |
| 2 | **Pulses per revolution** | How many encoder pulses equal one full rotation of the drum | From the encoder datasheet (e.g., 1024 ppr, 4096 ppr) | Needed to convert raw pulse counts into rotations |
| 3 | **Drum circumference** | The circumference of the hoist drum in millimeters | Measure the drum diameter with a tape measure, multiply by pi. E.g., diameter = 400mm, circumference = 1256.6mm | Each drum rotation pays out this much rope |
| 4 | **Sheave height** | The height of the top pulley (sheave) above ground level in meters | Survey measurement — measure once at installation | `hook_height = sheave_height - rope_length`. This is the reference point. |
| 5 | **Reeving factor** | The number of rope falls between the sheave and the hook block | Count the ropes, or check the crane specification sheet. Common values: 1, 2, 4, 6 | The hook moves less than the rope pays out. `actual_rope_travel = encoder_travel / reeving_factor` |

**The basic formula:**

```python
rope_payout_m = (encoder_pulses / pulses_per_rev) * drum_circumference_m
actual_descent_m = rope_payout_m / reeving_factor
hook_height_m = sheave_height_m - actual_descent_m
```

!!! note "What Is Reeving?"
    In most cranes, the rope wraps around pulleys multiple times between the top of the crane and the hook block. If there are 4 "falls" (4 parallel rope segments), then 4 meters of rope must be paid out for the hook to descend 1 meter. This is the reeving factor, and it also multiplies the lifting force (mechanical advantage).

#### Required Data from the Camera (Verification Layer)

| # | Data Point | What It Is | How to Get It | Why It's Needed |
|---|---|---|---|---|
| 6 | **Camera intrinsic matrix** | A 3x3 matrix containing `fx, fy` (focal lengths in pixels) and `cx, cy` (principal point) | OpenCV checkerboard calibration — print a checkerboard, take 20–30 photos from different angles | This is the mathematical description of how your specific camera turns 3D light rays into 2D pixel positions |
| 7 | **Distortion coefficients** | 5 numbers `(k1, k2, p1, p2, k3)` describing lens distortion | Same calibration as above — OpenCV computes both simultaneously | Real lenses bend light unevenly. Without correction, measurements at image edges can be 5–15% wrong |
| 8 | **Camera mounting height** | Height of the camera above ground in meters | Measure once with a tape measure or laser rangefinder | Needed to convert "distance from camera" into "height above ground" |
| 9 | **Camera tilt angle** | How many degrees below horizontal the camera is pointed | Use a digital inclinometer (phone app works), or compute from calibration | A tilted camera sees vertical distances as foreshortened. The tilt angle corrects for this. |
| 10 | **Hook real-world height** | The physical height of the crane hook body in millimeters | Measure with calipers or a tape measure. Measure the specific feature that the detector consistently boxes — usually the full hook body | The core of the pinhole model: `distance = (focal_length * real_height) / pixel_height` |

#### Optional Enhancements (Phase 2+)

| # | Data Point | When to Add | Impact |
|---|---|---|---|
| 11 | **Load cell weight** (from PLC) | When load varies significantly between operations | Enables rope stretch compensation. Steel rope stretches proportionally to load. |
| 12 | **Rope elasticity coefficient** | After load cell integration | `stretch_mm = (load_kg * rope_length_m * coefficient) / rope_cross_section_mm2` |
| 13 | **Temperature sensor** | For outdoor installations with >30C temperature swings | Camera focal length drifts ~0.01% per degree C. At 50m, a 30C swing causes ~15cm error. |
| 14 | **Wind speed sensor** | For safety features | Predicts hook sway, allows confidence reduction during high-wind conditions |

#### How OPC-UA / Modbus Communication Works

Modern cranes expose PLC data via **OPC-UA** (Open Platform Communications Unified Architecture), an industrial communication standard. Legacy cranes typically use **Modbus TCP/IP**.

```python
# Example: Reading hoist encoder via OPC-UA
from asyncua import Client

async def read_crane_data():
    async with Client("opc.tcp://crane-plc:4840") as client:
        # Read hoist encoder — the register address depends on PLC programming
        encoder_node = await client.get_node("ns=2;s=Hoist.Encoder.Position")
        encoder_value = await encoder_node.read_value()

        # Read load cell (optional, for stretch compensation)
        load_node = await client.get_node("ns=2;s=Load.Weight")
        load_value = await load_node.read_value()

        return encoder_value, load_value
```

!!! note "OPC Foundation Crane Standard"
    The OPC Foundation publishes **OPC 40020-1: Cranes & Hoists**, a standardized data model for crane telemetry. If the crane PLC supports this standard, the register addresses are predefined and documented. Ask the crane manufacturer.

#### Kalman Filter for Sensor Fusion

The Kalman filter is the standard algorithm for combining noisy sensor measurements. It has been used in aerospace navigation since the 1960s (Apollo program). The intuition: maintain a probabilistic estimate of the hook's position and velocity, then update it with each new measurement, weighting by how much you trust each sensor.

```python
from filterpy.kalman import KalmanFilter
import numpy as np

# State: [position, velocity]. We track where the hook is and how fast it's moving.
kf = KalmanFilter(dim_x=2, dim_z=2)

dt = 1/30  # Time step (30 Hz camera, or use actual frame timestamps)

# State transition: position changes by velocity * dt
kf.F = np.array([[1., dt],
                 [0., 1.]])

# Both sensors measure position directly
kf.H = np.array([[1., 0.],   # Encoder measures position
                 [1., 0.]])   # Vision measures position

# Process noise: how much do we trust our motion model?
# Small Q = trust the model (smooth output). Large Q = trust measurements more.
kf.Q = np.array([[0.001, 0.0],
                 [0.0,   0.01]])

# Measurement noise: how much do we trust each sensor?
encoder_variance = 0.0001    # Encoder is very precise: std dev ~1mm
vision_variance  = 0.04      # Vision is noisy: std dev ~20cm
kf.R = np.array([[encoder_variance, 0.],
                 [0., vision_variance]])

# Each frame:
kf.predict()
z_encoder = get_encoder_height()
z_vision = get_vision_height()
kf.update(np.array([z_encoder, z_vision]))

fused_height = kf.x[0]  # Best estimate, combining both sensors
```

**What the Kalman filter does for us:**

- At 100 Hz, the encoder drives smooth, precise tracking
- Every ~33ms (at 30 Hz), vision provides an absolute correction
- If the encoder drifts due to rope slip, vision pulls it back
- If vision is temporarily wrong (occlusion, lighting), the encoder carries through
- If either sensor fails entirely, the other continues operating (graceful degradation)

---

### Approach B: Vision-Only (Pinhole Camera Model)

If PLC integration is delayed or not feasible, the camera alone can estimate height using the **pinhole camera model** — the fundamental geometric relationship between a 3D scene and its 2D image.

#### The Pinhole Camera Model Explained

Imagine poking a tiny hole in a piece of cardboard and holding it between a candle and a wall. The candle projects an inverted image on the wall. A camera works the same way (with a lens instead of a pinhole, but the geometry is identical).

The key insight: **an object that is farther from the camera appears smaller in the image, in exact inverse proportion to its distance**. If an object is twice as far away, it appears half as tall in pixels.

```
                     +---------+
                     | Image   |
    Real object      | Plane   |      Camera
    (H_real tall)    |         |      Pinhole
                     |  h_px   |        *
    ==================|=========|========*
         Z distance   |         |    f (focal length)
                     +---------+
```

**The formula (triangle similarity):**

```
Z = (f * H_real) / h_pixels

Where:
  Z        = distance from camera to object (meters)
  f        = focal length in PIXELS (not millimeters!)
  H_real   = real-world height of the object (meters)
  h_pixels = height of the bounding box in pixels
```

!!! warning "Focal Length: Pixels, Not Millimeters"
    Camera specs list focal length in millimeters (e.g., "8mm lens"). But the formula needs focal length in **pixels**. The conversion is:

    ```
    f_pixels = (f_mm * image_width_pixels) / sensor_width_mm
    ```

    For example: an 8mm lens on a 1/2.3" sensor (6.17mm wide) at 1920px resolution:
    `f_pixels = (8 * 1920) / 6.17 = 2489 pixels`

    **Better yet**: camera calibration gives you `f_pixels` directly as `camera_matrix[0,0]` (horizontal) and `camera_matrix[1,1]` (vertical). No manual conversion needed.

#### The Full Height Estimation Pipeline

Getting from bounding box pixels to "height above ground" requires several steps:

**Step 1: Undistort the image** (correct for lens distortion)

```python
# Do this ONCE per camera setup — precompute the mapping
mapx, mapy = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1
)

# Then for each frame (fast — just a pixel lookup):
undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
```

**Step 2: Get bounding box height in pixels**

```python
# From Batman's normalized API output:
bbox_height_pixels = detection["box"]["height"] * image_height
bbox_center_y_pixels = detection["box"]["y"] * image_height
```

**Step 3: Compute distance from camera to hook**

```python
f_y = camera_matrix[1, 1]  # Vertical focal length in pixels
hook_real_height_m = 0.30   # 30cm hook, measured with calipers

Z_distance = (f_y * hook_real_height_m) / bbox_height_pixels
```

**Step 4: Convert distance to height above ground**

This is where the camera's tilt angle matters. If the camera is pointed straight at the horizon, the hook's position in the image directly corresponds to its height. But cameras are usually mounted high and tilted downward.

```python
import math

def estimate_hook_height(bbox_center_y_px, bbox_height_px, image_height,
                         camera_matrix, camera_height_m, camera_tilt_deg,
                         hook_real_height_m):
    """
    Estimate the hook's height above ground from a single bounding box.

    Args:
        bbox_center_y_px: Y-coordinate of bbox center in pixels (0 = top of image)
        bbox_height_px: Height of the bounding box in pixels
        image_height: Total image height in pixels
        camera_matrix: 3x3 intrinsic matrix from calibration
        camera_height_m: Camera mounting height above ground (meters)
        camera_tilt_deg: Camera tilt below horizontal (positive = looking down)
        hook_real_height_m: Physical height of the hook (meters)

    Returns:
        Estimated height of hook above ground (meters)
    """
    f_y = camera_matrix[1, 1]  # Vertical focal length (pixels)
    c_y = camera_matrix[1, 2]  # Principal point Y (pixels)

    # Distance from camera to hook (along the line of sight)
    Z = (f_y * hook_real_height_m) / bbox_height_px

    # Vertical angle from camera's optical axis to the hook
    pixel_offset = bbox_center_y_px - c_y
    angle_from_axis = math.atan2(pixel_offset, f_y)

    # Total angle below horizontal
    total_angle = math.radians(camera_tilt_deg) + angle_from_axis

    # Vertical drop from camera to hook
    vertical_drop = Z * math.sin(total_angle)

    # Hook height = camera height minus the drop
    hook_height = camera_height_m - vertical_drop

    return hook_height
```

#### Minimum Required Data (Vision-Only)

| # | Data Point | How to Get It | Effort |
|---|---|---|---|
| 1 | Camera intrinsic matrix (`fx, fy, cx, cy`) | OpenCV checkerboard calibration (30 min) | One-time |
| 2 | Distortion coefficients (`k1, k2, p1, p2, k3`) | Same calibration as above | One-time |
| 3 | Hook real-world height (mm) | Calipers or tape measure | One-time |
| 4 | Camera mounting height (m) | Tape measure or laser rangefinder | One-time |
| 5 | Camera tilt angle (degrees) | Digital inclinometer or phone app | One-time |

That's **5 measurements** — all one-time — to get a working vision-only system.

#### Camera Calibration Procedure

Camera calibration determines the intrinsic matrix and distortion coefficients. This is a well-established procedure using OpenCV:

**What you need:**

- A printed checkerboard pattern (9x6 inner corners recommended)
- The pattern glued to a rigid, flat surface (foam board or aluminum plate)
- The exact square size measured (e.g., 25mm squares)
- 20–30 photographs of the pattern through the installed camera, from different angles and positions

**The process:**

```python
import cv2
import numpy as np
import glob

# Define the checkerboard dimensions
CHECKERBOARD = (9, 6)        # Inner corners
SQUARE_SIZE = 25.0           # millimeters

# Prepare 3D object points (Z=0 since checkerboard is flat)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

# Process each calibration image
images = glob.glob("calibration_images/*.jpg")
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Refine corner locations to sub-pixel accuracy
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)

# Run calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# ret = reprojection error (should be < 0.5 pixels for good calibration)
# camera_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# dist_coeffs = [k1, k2, p1, p2, k3]

print(f"Reprojection error: {ret:.4f} pixels")
print(f"Camera matrix:\n{camera_matrix}")
print(f"Distortion coefficients: {dist_coeffs.ravel()}")
```

!!! tip "Calibration Quality"
    A **reprojection error below 0.5 pixels** indicates good calibration. If it's above 1.0, retake the calibration images — ensure the checkerboard is flat, well-lit, in focus, and captured from diverse angles (tilted, rotated, at different distances).

---

### Approach C: Two-Point Reference Calibration

This approach is a practical simplification of Approach B. Instead of measuring camera intrinsics, distortion coefficients, mounting height, and tilt angle, you calibrate the entire pixel-to-height mapping empirically using just **two reference images at known heights**.

#### Why This Works

The pinhole camera model (Approach B) tells us that the mapping from a hook's real-world height to its vertical pixel position in the image is governed by camera geometry. For a fixed camera watching a hook that moves purely vertically:

- **If the camera has no tilt** (optical axis horizontal): the mapping is **exactly linear**. The hook's pixel y-coordinate is a simple linear function of its real height.
- **If the camera is tilted**: the mapping follows a **Möbius transformation** (a type of rational function), but for typical industrial camera tilts (0–20°), it remains very close to linear.

The key insight: **you don't need to know the camera parameters individually**. Two reference measurements at known heights implicitly encode all the camera geometry — focal length, tilt, mounting height, and distance — into two simple coefficients.

#### The Math

Let `v` be the vertical pixel coordinate of the hook's bounding box centroid, and `h` be the real-world height in meters.

**For a horizontal camera** (θ = 0°), the pinhole model gives:

```
v = -(f_y / D) · h + (f_y · H_cam / D + c_y)
```

This is linear: **v = m·h + b**, where `m` and `b` implicitly encode all camera parameters (focal length `f_y`, horizontal distance `D`, camera height `H_cam`, and principal point `c_y`). You never need to measure any of them individually.

**For a tilted camera** (θ > 0°):

```
v = f_y · ((H_cam - h)·cos(θ) - D·sin(θ)) / ((H_cam - h)·sin(θ) + D·cos(θ)) + c_y
```

This is a **Möbius transformation** — a ratio of two linear functions of `h`. It's not perfectly linear, but the deviation from linearity is small when `sin(θ) · height_range / D` is small. In plain terms: the camera should be reasonably far from the crane relative to the hook's travel range, and not steeply angled. Both conditions hold for typical shipping yard camera installations.

**Two-point calibration solves the linear model exactly:**

Given:

- Reference 1: Hook at height `h₁` meters → bbox centroid at pixel row `v₁`
- Reference 2: Hook at height `h₂` meters → bbox centroid at pixel row `v₂`

Compute:

```
m = (v₁ - v₂) / (h₁ - h₂)
b = v₁ - m · h₁
```

Estimate height for any new observation:

```
h = (v - b) / m
```

That's it. No focal length, no distortion coefficients, no tilt measurement.

#### Calibration Procedure

1. **Move the hook to its maximum operating height** (e.g., 20m). Note the exact height — use a laser rangefinder, crane PLC readout, or tape measure against the boom.
2. **Capture a frame** through the installed camera. Run the RF-DETR detector to get the bounding box. Record the bbox **centroid y-coordinate** and **bbox height**, both in pixels.
3. **Move the hook to its minimum operating height** (e.g., 2m). Note the exact height.
4. **Capture another frame** and record the bbox centroid y-coordinate and bbox height.
5. **Compute `m` and `b`** using the formulas above. Store these along with both bbox heights as the calibration parameters.

Total time: **~10 minutes**. No checkerboard, no OpenCV, no protractor.

!!! tip "Choosing Reference Heights"
    Use the **full operating range** for the two reference points. Calibrating at 5m and 7m, then trying to estimate at 20m, is extrapolation and will be inaccurate. Always calibrate at the extremes of the range you care about.

#### Implementation

```python
from dataclasses import dataclass

@dataclass
class TwoPointCalibration:
    """Calibration from two reference height/pixel pairs."""
    h_high: float        # meters — hook height in reference image 1
    v_high: float        # pixels — centroid y in reference image 1
    bbox_h_high: float   # pixels — bbox height in reference image 1
    h_low: float         # meters — hook height in reference image 2
    v_low: float         # pixels — centroid y in reference image 2
    bbox_h_low: float    # pixels — bbox height in reference image 2

    @property
    def m(self) -> float:
        """Slope: pixels per meter of height change."""
        return (self.v_high - self.v_low) / (self.h_high - self.h_low)

    @property
    def b(self) -> float:
        """Intercept: pixel value at height = 0."""
        return self.v_high - self.m * self.h_high

    def estimate_height(self, bbox_center_y_px: float) -> float:
        """Estimate hook height from bbox centroid y-coordinate."""
        return (bbox_center_y_px - self.b) / self.m

    def pixel_for_height(self, height_m: float) -> float:
        """Predict where a hook at this height would appear (for validation)."""
        return self.m * height_m + self.b

    def estimate_with_confidence(
        self, bbox_center_y_px: float, bbox_height_px: float
    ) -> tuple[float, float]:
        """
        Estimate height and confidence.

        Confidence drops when the observed bbox height doesn't match
        the expected bbox height — indicating the hook's depth (distance
        from camera) has changed since calibration.

        Returns:
            (estimated_height_m, confidence) where confidence is 0.0–1.0
        """
        h_est = self.estimate_height(bbox_center_y_px)

        # Expected bbox height at this real height
        t = (h_est - self.h_low) / (self.h_high - self.h_low)
        t = max(0.0, min(1.0, t))
        expected_bbox_h = self.bbox_h_low + t * (self.bbox_h_high - self.bbox_h_low)

        if expected_bbox_h <= 0:
            return h_est, 0.0

        ratio = bbox_height_px / expected_bbox_h
        deviation = abs(ratio - 1.0)
        # 1.0 at perfect match, 0.0 at 30% bbox size deviation
        confidence = max(0.0, 1.0 - deviation / 0.3)

        return h_est, confidence
```

#### Accuracy Analysis

The accuracy depends on **camera tilt angle** (how far from horizontal the camera points) and the **ratio of height range to camera distance**.

**Numerical analysis** (camera 50m from crane, hook range 2–15m, f_y = 1000px):

| Camera Tilt | Max Interpolation Error | At Midpoint | Notes |
|---|---|---|---|
| 0–5° | < 7 cm | < 3 cm | Effectively exact — linear model matches reality |
| 5–15° | 7–25 cm | 3–12 cm | Good for most applications |
| 15–25° | 25–50 cm | 12–25 cm | Consider three-point calibration |
| 25°+ | > 50 cm | > 25 cm | Use Approach B (full calibration) instead |

These are worst-case **interpolation errors** from using a linear model instead of the true projective model. Actual accuracy also depends on:

- **Bounding box detection noise**: Typically ±3–5 pixels in centroid position, adding ±5–15 cm at typical distances
- **Detection consistency**: The detector must consistently box the same part of the hook across frames

**Total expected accuracy: ±10–30 cm raw**, improving with temporal smoothing (Kalman filter over a 10-frame window) to **±5–15 cm**.

#### Critical Assumption: Constant Depth

!!! warning "This Is the Most Important Section of Approach C"
    The accuracy numbers above assume the hook stays at approximately the **same distance from the camera** during operation as during calibration. If this assumption is violated, errors can reach **meters, not centimeters**. Read this section carefully.

**Why depth matters**: The pixel y-coordinate of the hook depends on both its real-world height AND its distance from the camera (depth along the optical axis). The two calibration points implicitly encode one specific depth. If the crane trolley later moves the hook toward or away from the camera, the depth changes and the calibration produces incorrect results.

**The math**: For a horizontal camera, the pixel y-coordinate is:

```
v = f_y × (H_cam - h) / D + c_y
```

The calibration computes `m = -f_y / D` and `b = f_y × H_cam / D + c_y`. If the hook is later at depth `D' = D + ΔD` instead of `D`:

```
height_error ≈ (H_cam - h) × ΔD / D
```

Where:

- `H_cam` = camera height (meters)
- `h` = actual hook height (meters)
- `D` = depth at calibration time (meters)
- `ΔD` = depth change since calibration (meters, positive = further from camera)

**Worked examples** (camera at H_cam = 10m, calibrated at D = 40m):

| Hook Height | ΔD (trolley move) | Height Error | Notes |
|---|---|---|---|
| 5m | +5m | **+0.63m** | Hook below camera: overestimates |
| 10m | +5m | **0.00m** | At camera height: immune to depth changes |
| 15m | +5m | **−0.63m** | Hook above camera: underestimates |
| 20m | +5m | **−1.25m** | Large height difference amplifies the error |
| 20m | +10m | **−2.50m** | Larger depth change, proportionally worse |

Two key observations:

1. **Error is zero when the hook is at the camera's height** — looking straight ahead, depth changes don't shift the hook vertically in the image.
2. **Error scales with `(H_cam − h) × ΔD`** — the further the hook is from the camera's altitude AND the more the depth changes, the worse the estimate gets.

**When is this a problem in practice?**

Cranes have three motions: hoist (up/down), trolley (along the boom), and slew/bridge travel. Within a single lift cycle (lower → pick → raise → travel → lower → place → raise), the vertical hoist motion happens at a roughly **fixed** trolley position. The depth-change problem only arises when the trolley moves the hook **toward or away** from the camera.

**Mitigation strategies:**

1. **Camera placement (most important)**: Mount the camera so the crane's primary horizontal travel (trolley/bridge) is **perpendicular to the camera's line of sight**. Perpendicular motion doesn't change depth — only parallel motion does. This single decision eliminates the biggest source of error.

    ```
    GOOD: Camera viewing from the side
    ┌──────────────────────────┐
    │     Crane Boom           │
    │  ←── trolley moves ──→   │
    │         hook             │
    │          ↕               │
    │       up / down          │
    └──────────────────────────┘
              ↑
          📷 Camera (perpendicular to boom)
    ```

    ```
    BAD: Camera viewing along the boom
    ┌──────────────────────────┐
    │     Crane Boom           │
    │  ←── trolley moves ──→   │
    │         hook             │
    └──────────────────────────┘
    📷 Camera ←──── depth changes as trolley moves
    ```

2. **Camera at mid-height**: Mount the camera at roughly the midpoint of the hook's operating height range. This minimizes `|H_cam − h|`, which is the multiplier in the error formula. A camera at 12m watching a hook that operates between 4m and 20m has a max `|H_cam − h|` of 8m, vs. a ground-level camera with max 20m.

3. **Re-calibrate per trolley position**: If the trolley moves to a new position, re-run the two-point calibration (~10 min). Practical for predictable, repetitive operations like container handling where the crane services the same positions repeatedly.

4. **Depth-change detection**: Use the bounding box height (apparent size) as an independent signal to detect when the depth has changed. See the next section.

#### Depth-Change Detection Using Bbox Height

The centroid y-coordinate gives the height estimate. The **bounding box height** (apparent size of the hook in pixels) provides an independent signal about the hook's distance from the camera: larger bbox → closer, smaller bbox → further. These two signals move independently — centroid y responds to height, bbox height responds to distance.

By recording the bbox height at both calibration points, you can compute the expected bbox height at any estimated height. If the observed bbox height deviates significantly, the depth has changed and the height estimate should be flagged as **low confidence**. This logic is implemented in the `estimate_with_confidence()` method of the `TwoPointCalibration` class above.

**How it works:**

1. Linearly interpolate the expected bbox height between the two calibration points based on the estimated real height
2. Compare the observed bbox height to the expected bbox height
3. If the ratio deviates from 1.0 by more than 30%, confidence drops to zero — indicating a significant depth change (~12m at 40m calibration distance)

!!! note "Detection, Not Correction"
    This tells you **when** the estimate is unreliable, not by **how much** it's wrong. For actual depth correction, you'd need the hook's real-world size (one additional measurement) to compute the distance from apparent size — which brings you partway toward Approach B. For most use cases, flagging low-confidence frames and excluding them from downstream decisions is sufficient.

#### Three-Point Extension (Near-Zero Interpolation Error)

Adding a **third calibration point** at an intermediate height lets you fit the exact Möbius transformation, eliminating interpolation error entirely regardless of camera tilt:

```
h = (A · v + B) / (C · v + 1)
```

Three calibration points give three equations in three unknowns (`A`, `B`, `C`). This model is exact for the projective geometry — it handles any camera tilt, distance, and mounting height without measuring them.

```python
import numpy as np

def fit_mobius(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """
    Fit a Möbius transformation h = (A*v + B) / (C*v + 1)
    from three (height, pixel_y) calibration pairs.

    This model is exact for the projective mapping between
    pixel y-coordinate and real-world height, regardless of
    camera tilt, distance, or mounting height.

    Args:
        points: [(h1, v1), (h2, v2), (h3, v3)]
            where h = height in meters, v = bbox centroid y in pixels

    Returns:
        (A, B, C) coefficients for h = (A*v + B) / (C*v + 1)
    """
    # Rearrange h = (A*v + B) / (C*v + 1) to:
    #   h * (C*v + 1) = A*v + B
    #   h*C*v + h = A*v + B
    #   A*v - B - h*C*v = -h
    #   [v, -1, -h*v] @ [A, B, C]^T = [-h]
    rows = []
    rhs = []
    for h, v in points:
        rows.append([v, -1.0, -h * v])
        rhs.append(-h)

    coeffs = np.linalg.solve(np.array(rows), np.array(rhs))
    return coeffs[0], coeffs[1], coeffs[2]  # A, B, C


def estimate_height_mobius(v: float, A: float, B: float, C: float) -> float:
    """Estimate height using fitted Möbius coefficients."""
    return (A * v + B) / (C * v + 1.0)
```

!!! note "When to Use Three Points"
    If you're already taking two reference images, taking a third at the midpoint adds ~5 minutes and eliminates all interpolation error. This is especially worthwhile if the camera is mounted with noticeable tilt (>10°) or relatively close to the crane (<20m).

#### When to Use This Approach

| Scenario | Recommendation |
|---|---|
| Quick deployment, camera < 15° tilt | ✅ Two-point calibration (this approach) |
| Camera tilt 15–25° | ⚠️ Use three-point extension (still this approach, just one more reference image) |
| Camera tilt > 25° or camera very close to crane | ❌ Use Approach B (full calibration) |
| Need < 5 cm accuracy | ⚠️ Add temporal smoothing, or upgrade to Approach A (sensor fusion) |
| Camera may be repositioned frequently | ✅ Re-calibration takes only 10 minutes (faster than Approach B) |
| Trolley moves **perpendicular** to camera view | ✅ Depth stays constant — ideal for this approach |
| Trolley moves **toward/away** from camera | ⚠️ Depth changes invalidate calibration. Use confidence scoring to flag unreliable frames, or re-calibrate per trolley position |
| Trolley has large travel range along camera axis (>10m) | ❌ Depth variation too large. Use Approach A (sensor fusion) or Approach B with per-frame distance estimation |

#### Advantages Over Approach B

- **Zero technical calibration**: No checkerboard, no OpenCV `calibrateCamera()`, no focal length computation
- **Operator-friendly**: Anyone who can read a laser rangefinder and click a button can calibrate this
- **10 minutes vs. 30+ minutes** setup time
- **Implicitly correct**: The two reference points absorb lens distortion, camera tilt, and mounting geometry without measuring them individually
- **No specialized tools**: No printed checkerboard pattern, no digital inclinometer, no sensor spec sheet

#### Limitations

- **Constant depth assumption (critical)**: The calibration implicitly encodes the hook's distance from the camera. If the crane trolley moves the hook toward or away from the camera after calibration, height estimates can be **off by meters**. See [Critical Assumption: Constant Depth](#critical-assumption-constant-depth) above for the error formula, worked examples, and mitigations. This is the single most important constraint to understand before deploying this approach.
- **Interpolation only**: Accurate within the calibration range. Extrapolation beyond the two reference heights is unreliable and should be flagged as low-confidence. Always calibrate at the full operating range.
- **Camera must not move**: Any change in camera position, angle, or zoom invalidates the calibration. Same constraint as Approach B, but recalibration is much faster (10 min vs. 30+ min).
- **Linear assumption**: The two-point model uses a linear approximation of the true projective mapping. This breaks down at high tilt angles (>20°). The three-point extension fixes this completely.
- **Hook sway (pendulum motion)**: If the hook swings laterally from wind or rapid trolley movement, the centroid shifts horizontally and introduces noise in the y-coordinate. Temporal smoothing (Kalman filter) mitigates this.
- **Detection dependency**: Requires the RF-DETR model to consistently detect the hook and produce stable bounding boxes. In frames where detection fails or the bbox is noisy, height estimation degrades.

---

## Part II: Accuracy Analysis

### Error Budget: Vision-Only at 20m Distance

For a **300mm crane hook** viewed from **20 meters** with a calibrated **1920x1080 camera** (focal length ~2500 pixels):

The hook appears as roughly `(2500 * 0.3) / 20 = 37.5 pixels` tall.

| Error Source | Magnitude | Impact on Height Estimate | Correctable? |
|---|---|---|---|
| **Bounding box uncertainty** | ±5 pixels in bbox height | ±2.7 m * (5/37.5) = **±36 cm** | Partially (better detector, temporal averaging) |
| **Calibration error** | 0.5 pixel reprojection | **±3 cm** | Yes (more calibration images) |
| **Lens distortion (uncorrected)** | 2–5% at image edges | **±40–100 cm** | Yes (apply distortion correction) |
| **Hook not facing camera** | 10 degree rotation changes apparent height | **±10 cm** | Partially (use width instead, or average) |
| **Atmospheric refraction** | Negligible below 100m | <1 cm | N/A |

**Total (root sum of squares, with distortion corrected):** ~±37 cm raw, improving to **±12 cm with Kalman filtering** over a 10-frame window.

### Error Budget: Sensor Fusion

| Error Source | Encoder Alone | Fusion (Encoder + Vision) |
|---|---|---|
| Rope slip (acceleration) | ±2–5 cm | Detected via vision disagreement, corrected |
| Rope stretch under load | ±1–3 cm | Compensated if load cell available |
| Encoder drift (hours) | Accumulates indefinitely | Vision resets to absolute reference |
| Encoder failure | Total loss | Vision takes over as sole source |
| **Typical total** | **±5–15 mm** | **±2–10 mm** |

### Accuracy vs. Distance Chart

| Distance to Hook | Hook Appears As | Vision-Only Accuracy | Fusion Accuracy |
|---|---|---|---|
| 5 m | 150 px tall | ±3 cm | ±2 mm |
| 10 m | 75 px tall | ±8 cm | ±3 mm |
| 20 m | 37 px tall | ±12 cm (filtered) | ±5 mm |
| 30 m | 25 px tall | ±25 cm (filtered) | ±8 mm |
| 50 m | 15 px tall | ±50+ cm (unreliable) | ±10 mm |

!!! warning "Vision Degrades with Distance"
    Beyond ~30m, the crane hook occupies so few pixels that bounding box noise dominates the measurement. At 50m+, vision-only is unreliable. This is where encoder data becomes essential. Alternatively, a higher-resolution camera (4K) or a longer focal length lens can extend the usable range.

---

## Part III: What Exists vs. What Needs Building

### Current Codebase State

| Component | Status | Location |
|---|---|---|
| RF-DETR detection with bounding boxes | Exists | `src/core/inference.py`, `backend/app/services/inference_runner.py` |
| Video streaming + frame extraction | Exists | `backend/app/services/video_processor.py` |
| Object tracking (ByteTrack) | Exists | `src/core/inference.py`, `backend/app/services/tracker.py` |
| FastAPI backend + React frontend | Exists | `backend/`, `frontend/` |
| Camera calibration module | **Does not exist** | Needs: `src/core/calibration.py` |
| Calibration storage + API | **Does not exist** | Needs: `backend/app/api/calibration.py` |
| Height estimation post-processor | **Does not exist** | Needs: hook into `inference_runner.py` |
| PLC/OPC-UA data connector | **Does not exist** | Needs: new service module |
| Kalman filter integration | **Does not exist** | Needs: new filtering module |
| Height overlay on video export | **Does not exist** | Needs: update `draw_detections()` |

### Integration Points (Where New Code Hooks In)

The detection pipeline currently flows:

```
Camera Frame
    |
    v
RF-DETR Model Predict
    |
    v
_parse_rfdetr_results()          <-- pixel xyxy -> normalized center xywh
    |
    v
Return detection dicts to API    <-- INSERT HEIGHT ESTIMATION HERE
    |
    v
draw_detections() for export     <-- INSERT HEIGHT OVERLAY HERE
```

The insertion is non-breaking: if no calibration data exists for a video, the height field is simply `null`. Existing behavior is unchanged.

---

## Part IV: Agent Harness for Autonomous Iteration

Once the core infrastructure is built (calibration module, height estimation function, validation data), there are many parameters to tune: Kalman filter noise matrices, detection confidence thresholds, outlier rejection bounds, distortion correction fidelity, temporal smoothing windows. Rather than hand-tuning these, we can build an **agent harness** that iterates autonomously until the system converges to a target accuracy.

### Design Principles

These principles are derived from studying Karpathy's `autoresearch`, LangGraph's `ml-pipeline.ai` critic loop, DSPy MIPROv2, and OpenCode's `ralph-loop`:

| Principle | Implementation | Rationale |
|---|---|---|
| **Fixed evaluation harness** | `evaluate.py` is read-only — the agent cannot modify it | Prevents the agent from gaming the metric (Goodhart's Law) |
| **Single mutable target** | Agent only modifies `calibration_experiment.py` | Constrains the search space; prevents scope creep |
| **Git-based rollback** | Commit before each experiment; `git reset --hard` on regression | Always returns to last known good state; no accumulated damage |
| **Fresh context per iteration** | State lives in files and git, not LLM memory | Prevents "context pollution" where the LLM accumulates confused state |
| **Multi-objective convergence** | Must pass ALL criteria simultaneously, not just one | Prevents overfitting one metric at the expense of others |
| **Bounded resources** | Max iterations, timeouts, cost limits | Prevents runaway loops and bill shock |

### Project Structure

```
crane-height-agent/
|
+-- evaluate.py                 # IMMUTABLE: Fixed evaluation harness
|   |-- load_ground_truth()     # Loads known heights from validation set
|   |-- run_height_estimation() # Runs the full pipeline on test frames
|   |-- compute_metrics()       # MAE, RMSE, max error, temporal std
|   +-- check_convergence()     # Multi-objective pass/fail gate
|
+-- calibration_experiment.py   # MUTABLE: Agent modifies this freely
|   |-- camera_params           # Intrinsics, distortion, extrinsics
|   |-- kalman_config           # Q matrix, R matrix, window size
|   |-- estimation_method       # Which formula to use
|   +-- postprocessing          # Outlier rejection, smoothing params
|
+-- ground_truth/               # IMMUTABLE: Validation data
|   |-- frames/                 # Video frames with known hook heights
|   +-- measurements.json       # {frame_id: actual_height_meters}
|
+-- results.tsv                 # Experiment log (append-only)
|   # commit | MAE_cm | RMSE_cm | max_err_cm | temporal_std | status
|
+-- program.md                  # Agent instructions
```

### The Iteration Loop

```
LOOP FOREVER:
  1. Read results.tsv to understand what has been tried and what worked
  2. Read calibration_experiment.py to see the current parameter state
  3. Propose ONE focused change:
     - Adjust Kalman Q/R matrices
     - Try different distortion correction approach
     - Change outlier rejection threshold
      - Switch estimation method (pinhole vs. two-point reference vs. Möbius)
     - Adjust temporal smoothing window
  4. git commit -m "Try: <concise description of change>"
  5. Run: python evaluate.py > run.log 2>&1     (timeout: 120 seconds)
  6. Extract metrics: grep "^MAE_cm:\|^RMSE_cm:\|^converged:" run.log
  7. IF crashed:
     - Read last 50 lines of run.log
     - Attempt fix (max 2 retries per crash)
     - If unfixable, log "crash" and move on
  8. Append result to results.tsv
  9. IF MAE_cm improved over best-so-far:
     - Keep the commit (advance branch)
     ELSE:
     - git reset --hard HEAD~1  (rollback)
  10. IF converged == true:
     - STOP. Report final metrics.
```

### Convergence Criteria

The system is "converged" when ALL of these pass simultaneously:

```python
def check_convergence(metrics: dict) -> bool:
    """
    All criteria must pass. This prevents the agent from optimizing
    one metric at the expense of others.
    """
    return (
        metrics["mae_cm"] < 5.0           # Mean absolute error under 5cm
        and metrics["rmse_cm"] < 8.0      # Root mean square error under 8cm
        and metrics["max_error_cm"] < 15.0 # No single frame is wildly wrong
        and metrics["temporal_std_cm"] < 2.0  # Smooth across consecutive frames
        and metrics["reprojection_px"] < 1.0  # Camera calibration is solid
        and metrics["num_valid_frames"] >= 100  # Statistically significant
    )
```

!!! note "Why Multi-Objective?"
    A single metric (like MAE) can be gamed. For example, the agent could make 99 frames perfect and 1 frame 500cm wrong — the MAE looks okay but the system is unusable. The `max_error_cm` criterion catches this. Similarly, `temporal_std_cm` ensures the output doesn't jitter even if the average is correct.

### Guardrails and Safety

| Guardrail | Value | What Happens When Triggered |
|---|---|---|
| **Max iterations** | 20 | Agent stops, reports best result so far |
| **Timeout per experiment** | 120 seconds | Experiment killed, logged as "timeout", agent continues |
| **Max consecutive failures** | 3 | Agent stops, requests human review |
| **Regression rollback** | Always | `git reset --hard` on any MAE regression — never degrade |
| **Human checkpoint** | At iteration 15 | Pause for human review before final attempts |
| **API call limit** | 100 calls | Hard stop to prevent cost runaway |
| **Cost limit** | $5 USD | Hard stop |

### How to Run It

**Option 1: OpenCode ralph-loop** (simplest, good for prototyping)

```bash
/ralph-loop "Iterate on crane height estimation parameters in
calibration_experiment.py. Read results.tsv for history. Run
'python evaluate.py' to test each change. Target: MAE < 5cm,
max_error < 15cm, temporal_std < 2cm. Git commit before each try,
git reset --hard on regression. Log all results to results.tsv.
Stop when converged or after 20 iterations."
```

**Option 2: LangGraph StateGraph** (production-grade, with persistent state)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

graph = StateGraph(CalibrationState)

# Define the pipeline nodes
graph.add_node("analyze_history", analyze_previous_experiments)
graph.add_node("propose_change", llm_propose_parameter_adjustment)
graph.add_node("run_experiment", execute_evaluate_py)
graph.add_node("check_results", evaluate_and_log)
graph.add_node("critic", decide_next_action)

# Wire the edges
graph.set_entry_point("analyze_history")
graph.add_edge("analyze_history", "propose_change")
graph.add_edge("propose_change", "run_experiment")
graph.add_edge("run_experiment", "check_results")
graph.add_edge("check_results", "critic")

# Critic decides: iterate more, or finalize
graph.add_conditional_edges("critic", route_decision, {
    "iterate": "analyze_history",  # Loop back
    "finalize": END,               # Done!
})

# Persistent state (survives restarts)
checkpointer = SqliteSaver("crane_calibration_agent.db")
app = graph.compile(checkpointer=checkpointer)
```

**Option 3: Standalone Python script** (no framework dependencies)

```python
import subprocess
import csv

MAX_ITERATIONS = 20
TIMEOUT_SEC = 120

for iteration in range(MAX_ITERATIONS):
    # 1. Agent proposes a change (via LLM API call)
    change = propose_change(read_history("results.tsv"))

    # 2. Apply change to calibration_experiment.py
    apply_change(change)

    # 3. Git commit
    subprocess.run(["git", "commit", "-am", f"Try: {change.description}"])

    # 4. Run experiment
    try:
        result = subprocess.run(
            ["python", "evaluate.py"],
            capture_output=True, timeout=TIMEOUT_SEC, text=True
        )
        metrics = parse_metrics(result.stdout)
    except subprocess.TimeoutExpired:
        log_result(iteration, "timeout", change.description)
        subprocess.run(["git", "reset", "--hard", "HEAD~1"])
        continue

    # 5. Evaluate
    if metrics["mae_cm"] < best_mae:
        best_mae = metrics["mae_cm"]
        log_result(iteration, "improved", change.description, metrics)
    else:
        subprocess.run(["git", "reset", "--hard", "HEAD~1"])
        log_result(iteration, "regressed", change.description, metrics)

    # 6. Check convergence
    if check_convergence(metrics):
        print(f"Converged at iteration {iteration}!")
        break
```

---

## Part V: Implementation Roadmap

| Phase | Timeline | Deliverable | Success Criteria |
|---|---|---|---|
| **1. Camera calibration** | Week 1–2 | `src/core/calibration.py` with pinhole math; calibration API endpoints; basic `Z = (f * H) / h` working | Reprojection error < 0.5px; height estimates within ±50cm of tape-measure ground truth |
| **2. Pipeline integration** | Week 3–4 | Height estimation wired into `inference_runner.py`; Kalman filter for smoothing; height overlaid on exported video | Height field present in API responses; smooth temporal output |
| **3. Validation dataset** | Week 4–5 | Ground truth collection: hook at 5+ known heights, 100+ frames total, measurements.json created | Statistically significant validation set |
| **4. Agent harness** | Week 5–6 | `evaluate.py` fixed harness; `calibration_experiment.py` mutable target; ralph-loop or LangGraph orchestration | Agent runs, modifies parameters, evaluates, rolls back on regression |
| **5. Autonomous iteration** | Week 6–7 | Agent runs 10–20 experiments, converges to target accuracy | MAE < 5cm, max_error < 15cm, temporal_std < 2cm |
| **6. PLC fusion** (if applicable) | Week 7–9 | OPC-UA connector; Kalman fusion of encoder + vision | ±2–10mm accuracy; graceful degradation on sensor failure |

---

## Appendix A: Operator Data Collection Checklist

Print this and hand it to the shipping yard operator. All measurements are one-time unless noted.

### Must-Have for Approach C (Two-Point Reference — simplest, ~10 min)

- [ ] **Hook at maximum operating height**: _______ meters (measure with laser rangefinder or PLC readout)
- [ ] **Capture frame at max height**: Save as JPEG, run detector, record bbox centroid y: _______ pixels and bbox height: _______ pixels
- [ ] **Hook at minimum operating height**: _______ meters
- [ ] **Capture frame at min height**: Save as JPEG, run detector, record bbox centroid y: _______ pixels and bbox height: _______ pixels
- [ ] **Optional third reference** (recommended if camera has noticeable tilt): Hook at mid height: _______ meters, centroid y: _______ pixels
- [ ] **Camera placement note**: Ensure the camera views the crane from the **side** (perpendicular to trolley travel), not along the boom. See [Critical Assumption: Constant Depth](#critical-assumption-constant-depth).

### Must-Have for Approach B (Full Camera Calibration — ~30 min)

- [ ] **Crane hook body height**: _______ mm (measure the part that the detector consistently boxes)
- [ ] **Camera calibration images**: 20–30 photos of a 9x6 checkerboard through the installed camera, from different angles. Store as JPEG files.
- [ ] **Camera mounting height above ground**: _______ meters
- [ ] **Camera tilt angle below horizontal**: _______ degrees
- [ ] **Ground truth measurements**: Place hook at 3–5 known heights (use laser rangefinder), record: Height 1: _______ m, Height 2: _______ m, Height 3: _______ m, Height 4: _______ m, Height 5: _______ m. Capture 30 seconds of video at each height.

### Should-Have (if PLC access available — dramatically improves accuracy)

- [ ] **PLC IP address and protocol**: _______ (OPC-UA / Modbus TCP)
- [ ] **Hoist encoder register address**: _______ (ask crane manufacturer or PLC programmer)
- [ ] **Encoder pulses per revolution**: _______ (from encoder datasheet)
- [ ] **Hoist drum diameter**: _______ mm (measure with tape, compute circumference = diameter * 3.14159)
- [ ] **Sheave (top pulley) height above ground**: _______ meters
- [ ] **Reeving factor**: _______ (count the rope falls, or check crane spec sheet)

### Nice-to-Have (improves robustness)

- [ ] **Load cell register address**: _______ (for rope stretch compensation)
- [ ] **Camera model and lens spec sheet**: Brand: _______ Model: _______ Focal length: _______ mm Sensor size: _______
- [ ] **Crane operating envelope**: Min height: _______ m, Max height: _______ m, Max speed: _______ m/s
- [ ] **Environmental notes**: Indoor/outdoor, temperature range, lighting conditions, common weather

---

## Appendix B: Key Formulas Reference

### Two-Point Reference Calibration

```
m = (v₁ - v₂) / (h₁ - h₂)
b = v₁ - m · h₁
height = (v_observed - b) / m
```

### Möbius Transformation (three-point extension)

```
height = (A · v + B) / (C · v + 1)

Solve [A, B, C] from three (height, pixel_y) calibration pairs:
  A·v - B - h·C·v = -h   (for each pair)
```

### Pinhole Model (distance from known object size)

```
Z = (f_y * H_real) / H_pixel
```

### Focal Length Conversion (mm to pixels)

```
f_pixels = (f_mm * image_width_pixels) / sensor_width_mm
```

### Encoder to Hook Height

```
rope_payout = (encoder_pulses / pulses_per_rev) * drum_circumference
hook_height = sheave_height - (rope_payout / reeving_factor)
```

### Height from Tilted Camera

```
vertical_angle = atan2(bbox_center_y - c_y, f_y)
total_angle = camera_tilt + vertical_angle
hook_height = camera_height - Z * sin(total_angle)
```

### Kalman Filter Update (simplified)

```
Predict:  x_hat = F * x_prev
Update:   K = P * H^T * (H * P * H^T + R)^-1    (Kalman gain)
          x = x_hat + K * (z_measured - H * x_hat)
```

---

## Appendix C: References

### Academic & Industry Research

- MDPI Applied Sciences (2026) — Tower crane vision system field study; 25% efficiency improvement
- MDPI Applied Sciences (2025) — FLE-YOLO crane hook detection; 97.3% precision, 98.3% AP50
- IEEE (2020) — Stereovision tracking for loader crane tip positioning
- Nature Scientific Reports (2026) — Crane structural dynamics and payload height effects
- OPC Foundation — OPC 40020-1: Cranes & Hoists standardized data model

### Software & Libraries

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) — Official calibration tutorial
- [FilterPy](https://github.com/rlabbe/filterpy) — Kalman filter implementations for Python
- [opcua-asyncio](https://github.com/FreeOpcUa/opcua-asyncio) — Python OPC-UA client for PLC communication
- [Supervision](https://github.com/roboflow/supervision) — Roboflow's CV toolkit for tracking and measurement
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — Ralph loop pattern for autonomous iteration
- [ml-pipeline.ai](https://github.com/iotlodge/ml-pipeline.ai) — LangGraph critic loop for autonomous ML pipelines

### Batman Codebase (integration points)

- `src/core/inference.py` — Detection dataclass, `draw_detections()`, video inference pipeline
- `backend/app/services/inference_runner.py` — `_parse_rfdetr_results()`, `run_on_image()`, `run_on_video()`
- `backend/app/models/annotation.py` — `BoundingBox` schema (normalized center xywh)
- `backend/app/services/video_processor.py` — Frame extraction, video metadata
