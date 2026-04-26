# Mind Graph — Spreader/Container Alignment
Last updated: 2026-04-23

### Spreader–Container 6-DoF Pose (Container-Side, Monocular + PnP)
- **Description**: Camera mounted on spreader looks down at the container; detect corner castings or lockholes, run PnP to recover relative pose.
- **Related topics**: Corner Casting & Lockhole Detection, Anti-Sway / Skew Control, Stereo Auto-Landing
- **Key papers**:
  - [mi2025-yolo11-epnp] High-Precision Pose Measurement of Containers... Dual-Trolley QSC (Sensors 2025) — current SOTA; YOLO11 detector + coplanar-constrained EPnP; mm-level accuracy.
  - [fu2019-monocularpose] Position–Pose Measurement of Crane Sway Based on Monocular Vision (IET JoE 2019) — classic monocular baseline with nonlinear optimization.
- **Other relevant papers**:
  - [improvedYolov5-2024-spreader] Improved YOLOv5 for spreader keypoint 3-D detection — keypoint-based variant.
  - [lockholeCNN2021] Recognition and Positioning of Container Lock Holes — CNN precursor.
  - [monocular-corner-alignment] Automatic Alignment of Container Corner (Researching.cn).

### Spreader 6-DoF Pose (Spreader-Side, Learning-Based)
- **Description**: Camera on gantry/trolley looks at the spreader itself; recover the spreader's pose directly (the container pose is then derived indirectly or via a second camera).
- **Related topics**: Synthetic Data for 6D Pose, Anti-Sway / Skew Control
- **Key papers**:
  - [pateraki2023-singleview] Crane Spreader Pose Estimation from a Single View (VISAPP 2023) — textured 3D model → synthetic training → EPOS detector → 2D–3D correspondences.
  - [lourakis2021-markerless] Markerless Visual Tracking of a Container Crane Spreader (ICCVW 2021) — model-based line-segment tracker.
- **Other relevant papers**:
  - [kawai2010-imagesensor] Image-sensor system for spreader position measurement — early work feeding anti-sway.

### Stereo / Auto-Landing
- **Description**: Depth from stereo cameras on the spreader to directly recover container top-plane geometry (centroid, slope, distance).
- **Related topics**: Spreader–Container 6-DoF Pose, LiDAR Profiling
- **Key papers**:
  - [yoon2010-stereoautoland] Real-time Container Position Estimation Using Stereo Vision for Auto-Landing (Control Eng. Practice 2010) — foundational stereo pipeline for landing.
- **Other relevant papers**:
  - [fastvision2024-pose] Fast Vision-Based Algorithm for Automated Container Pose Estimation (Springer 2024) — real-time deployment angle.

### LiDAR / Active-Sensing Alignment
- **Description**: Laser profiling and point-cloud methods for spreader/container/stack geometry; dominant in fielded automated terminals today.
- **Related topics**: Stereo / Auto-Landing, Survey
- **Key papers**:
  - [zhou2025-lcspose] LCSPose: Markerless 6-DoF Pose of a Quay Crane Spreader via LiDAR+Camera (IEEE 2025) — current fusion SOTA.
  - [liu2012-irstructuredlight] Automatic Spreader–Container Alignment Using IR Structured Lights (Appl. Opt. 2012) — active IR approach, reach-stacker use case.
- **Other relevant papers**:
  - [coilworkpieces2025-overhead] Coil-workpiece pose estimation from point clouds — transferable methodology.

### Corner Casting & Lockhole Detection (Detection-Only, Feeds Pose)
- **Description**: 2D detection and localization of ISO-1161 corner castings / twistlock holes; the usual first stage for any PnP pose pipeline.
- **Related topics**: Spreader–Container 6-DoF Pose
- **Key papers**:
  - [dl2019-cornercasting] Deep Learning–Assisted Real-Time Corner Casting Recognition (2019) — first DL detector for corner castings in this domain.
- **Other relevant papers**:
  - [lockholeCNN2021] Lock-Hole CNN (2021).
  - [mi2025-yolo11-epnp] (also listed above; detection is its front end).

### Skew / Yaw Estimation for Closed-Loop Control
- **Description**: Estimating only the rotation (skew/yaw) of the spreader for anti-skew control; a narrower, faster sub-problem than full 6-DoF.
- **Related topics**: Spreader 6-DoF Pose (Spreader-Side), Bad-Weather Robustness
- **Key papers**:
  - [chum2026-rotation] Crane Spreader Rotation Estimation for Vision-Based Automated Container Handling (IEEE 2026) — calibration-free, YOLOv8-based.
  - [schaper2014-2dof-skew] 2-DOF Skew Control of Boom Cranes (CEP 2014) — control-theoretic side (defines the signal spec).
- **Other relevant papers**:
  - [ngoHong-skewcontrol] Skew Control of a Container Crane — PID/fuzzy.

### Bad-Weather / Robustness
- **Description**: Pre-processing and sensor choices that keep the pose pipeline alive in rain, fog, low-light, and shocks.
- **Related topics**: Skew/Yaw Estimation, LiDAR / Active-Sensing Alignment
- **Key papers**:
  - [tanakaKaneko-badweather] Measurement of a Container Crane Spreader Under Bad Weather by Image Restoration.
- **Other relevant papers**:
  - [liu2012-irstructuredlight] IR structured light — active method for the same motivation.

### Surveys / Context
- **Description**: Field-level surveys and state-of-the-art comparisons, including the LiDAR-vs-camera trade.
- **Key papers**:
  - [benkert2023-survey] Chances and Challenges: Transformation from Laser-Based to Camera-Based Container Crane Automation (JMSE 2023) — the entry-point survey.

### Cross-Domain Toolbox (Applicable Techniques)
- **Description**: Not about cranes, but directly reusable algorithmic building blocks.
- **Related topics**: Spreader 6-DoF Pose, Corner Casting & Lockhole Detection
- **Key papers**:
  - [xin2022-uavlanding-review] Vision-Based Autonomous Landing for UAV: A Review — the "approach-and-dock on a flat rectangular target" problem solved in the UAV community.
  - [kalaitzakis2021-fiducial] Fiducial Marker Review — AprilTag/ArUco/WhyCon for cm-level pose if passive markers are acceptable.
