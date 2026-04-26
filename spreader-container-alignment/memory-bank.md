# Paper Memory Bank — Spreader/Container Alignment
Last updated: 2026-04-23

Scope: Estimating distance, offset, yaw/skew, and full 6-DoF pose between a crane spreader (STS / RTG / reach stacker) and a shipping container (or a container on a truck chassis or in a stack), for automated pickup and landing. Includes vision-based, LiDAR-based, and sensor-fusion approaches.

Legend for tiering:
- T1 = directly on spreader–container relative pose / landing
- T2 = closely adjacent (skew/sway control that consumes this pose, survey, bad-weather robustness, corner-casting detection without full pose)
- T3 = cross-domain methods that map cleanly (UAV autolanding, fiducial markers) — for toolbox ideas only

---

### [pateraki2023-singleview] Crane Spreader Pose Estimation from a Single View
- **Authors**: Maria Pateraki, Panagiotis Sapoutzoglou, Manolis Lourakis
- **Venue**: VISAPP (VISIGRAPP) 2023
- **URL**: https://www.scitepress.org/PublishedPapers/2023/117888/117888.pdf
- **Status**: discovered
- **Topics**: monocular-pose, synthetic-data, learning-based, spreader-side
- **Abstract**: Infers full 6-DoF pose of a crane spreader from a single RGB image. Builds a photorealistic textured 3D model of the spreader, renders synthetic training images, trains an EPOS-based detector, and establishes 2D–3D correspondences to recover pose. First reported single-view 6D spreader pose work.
- **Notes**: T1. Strong baseline for the "spreader-side" approach — camera on gantry looking at spreader. Complements container-side approaches. Discusses bootstrap/re-init problem that 3D tracking methods usually ignore.
---

### [lourakis2021-markerless] Markerless Visual Tracking of a Container Crane Spreader
- **Authors**: Manolis Lourakis, Maria Pateraki
- **Venue**: ICCVW 2021 (CV in HRC workshop)
- **URL**: https://openaccess.thecvf.com/content/ICCV2021W/CVinHRC/papers/Lourakis_Markerless_Visual_Tracking_of_a_Container_Crane_Spreader_ICCVW_2021_paper.pdf
- **Status**: discovered
- **Topics**: 3D-model-tracking, line-segments, monocular, spreader-side
- **Abstract**: Model-based 3D tracking of spreader position/orientation by matching visible wireframe lines of a triangle-mesh model to detected line segments in the image; robust to occlusion and partial visibility.
- **Notes**: T1. Companion work to `pateraki2023-singleview`. Good non-learning baseline; relies on strong straight-edge content on the spreader.
---

### [yoon2010-stereoautoland] Real-time Container Position Estimation Method Using Stereo Vision for Container Auto-Landing System
- **Authors**: Y. Yoon, H. Hwang, et al.
- **Venue**: Control Engineering Practice, 2010 (also ResearchGate PP)
- **URL**: https://www.semanticscholar.org/paper/24fa3330d06c0365946c3a9a8bf98bae05064313
- **Status**: discovered
- **Topics**: stereo-vision, auto-landing, container-side, centroid+slope
- **Abstract**: Real-time stereo pipeline to estimate container position, centroid, and slope from spreader-mounted cameras for container auto-landing.
- **Notes**: T1. Widely cited foundational stereo-vision paper on the landing side of the problem.
---

### [liu2012-irstructuredlight] Automatic Spreader–Container Alignment System Using Infrared Structured Lights
- **Authors**: S. Liu, et al.
- **Venue**: Applied Optics 51(16), pp. 3205–3215, 2012
- **URL**: https://opg.optica.org/abstract.cfm?uri=ao-51-16-3205  (mirror: https://pubmed.ncbi.nlm.nih.gov/22695551/)
- **Status**: discovered
- **Topics**: active-sensing, structured-light, reach-stacker, all-weather
- **Abstract**: Projects an IR structured-light pattern onto the container top and computes relative spreader–container position/orientation from the deformed pattern. Targeted at reach stackers, designed to work in poor ambient lighting.
- **Notes**: T1. One of few "active" approaches in this literature; relevant when RGB robustness is marginal.
---

### [mi2025-yolo11-epnp] High-Precision Pose Measurement of Containers on the Transfer Platform of the Dual-Trolley Quayside Container Crane Based on Machine Vision
- **Authors**: Mi et al.
- **Venue**: Sensors (MDPI) 25(9):2760, 2025
- **URL**: https://www.mdpi.com/1424-8220/25/9/2760  (mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC12074294/)
- **Status**: discovered
- **Topics**: YOLO11, EPnP, lockhole-detection, dual-trolley-STS, sub-mm-accuracy
- **Abstract**: Multi-scale adaptive-frequency YOLO11 detector for container lockholes on the transfer platform of a dual-trolley quay crane, with an enhanced EPnP using coplanar lockhole constraints. Reports MAE-P ≈ 0.024 m, MAE-θ ≈ 0.11°, mAP@0.5 = 95.1%.
- **Notes**: T1. Current SOTA for the "container-side, monocular, keypoint+PnP" family. Strong numbers, realistic scene. Main baseline for any new algorithm.
---

### [zhou2025-lcspose] LCSPose: Efficient, Accurate and Scalable Markerless 6-DoF Pose Estimation of a Quay Crane Spreader Based on LiDAR and Camera
- **Authors**: Zhou et al.
- **Venue**: IEEE (2025) — IEEE Xplore doc 11128533
- **URL**: https://ieeexplore.ieee.org/abstract/document/11128533/  (preprint: https://www.researchgate.net/publication/393002896)
- **Status**: discovered
- **Topics**: LiDAR-camera-fusion, semantic-geometric-segmentation, coarse-to-fine, spreader-6dof
- **Abstract**: Marker-free 6-DoF pose of a quay crane spreader via LiDAR + camera fusion using a semantic-geometric segmentation module and multi-view coarse-to-fine refinement.
- **Notes**: T1. Most recent sensor-fusion point of reference. Worth reading in full — likely sets scalability/robustness bar.
---

### [chum2026-rotation] Crane Spreader Rotation Estimation for Vision-Based Automated Container Handling
- **Authors**: Chum et al.
- **Venue**: IEEE (2026) — IEEE Xplore 11373178
- **URL**: https://ieeexplore.ieee.org/iel8/6287639/11323511/11373178.pdf
- **Status**: discovered
- **Topics**: yaw/skew, YOLOv8, calibration-free, spreader-rotation
- **Abstract**: Calibration-free, purely vision-based estimation of crane spreader rotation (skew) using an enhanced YOLOv8. Infers apparent orientation of fixed reference objects to recover spreader yaw.
- **Notes**: T1 (narrow). Explicitly the "rotation-only" sub-problem — fastest and most deployable piece of the full pose.
---

### [fastvision2024-pose] A Fast Vision-Based Algorithm for Automated Container Pose Estimation
- **Authors**: (Springer chapter authors)
- **Venue**: Springer LNEE, 2024 (chapter DOI 10.1007/978-981-97-1876-4_64)
- **URL**: https://link.springer.com/chapter/10.1007/978-981-97-1876-4_64
- **Status**: discovered
- **Topics**: real-time, container-pose, fast-algorithm
- **Abstract**: Device + method for recognizing and measuring container targets for automated loading/unloading with an emphasis on low-latency vision processing.
- **Notes**: T1. Check for a practical deployment-oriented comparison point.
---

### [improvedYolov5-2024-spreader] Improved YOLOv5 Network for High-Precision Three-Dimensional Detection of Spreader Keypoints
- **Authors**: (authors not yet extracted)
- **Venue**: 2024 (PubMed 39275386)
- **URL**: https://pubmed.ncbi.nlm.nih.gov/39275386/
- **Status**: discovered
- **Topics**: YOLOv5, attention, spreader-keypoints, 3D-pose
- **Abstract**: Adds attention mechanisms to YOLOv5 to improve detection of spreader keypoints used for 3-D pose / automated loading alignment.
- **Notes**: T1. Keypoint-based variant complementing lockhole-based methods.
---

### [monocular-corner-alignment] Automatic Alignment of Container Corner Based on Monocular Vision
- **Authors**: (unknown at time of record)
- **Venue**: Researching.cn (Chinese optics journal)
- **URL**: https://www.researching.cn/articles/OJ960f6adbb2eaf5b1
- **Status**: discovered
- **Topics**: monocular, corner-region-analysis, hoisting
- **Abstract**: Analyzes monocular images of hoisting scenes, focusing on regional characteristics of container corner components to drive alignment.
- **Notes**: T1. Bibliographic record only — full text likely in Chinese.
---

### [benkert2023-survey] Chances and Challenges: Transformation from a Laser-Based to a Camera-Based Container Crane Automation System
- **Authors**: Johannes Benkert, Robert Maack, Tobias Meisen
- **Venue**: J. Marine Science & Engineering (MDPI) 11(9):1718, 2023
- **URL**: https://www.mdpi.com/2077-1312/11/9/1718
- **Status**: discovered
- **Topics**: survey, lidar-vs-camera, ACT-automation
- **Abstract**: Narrative review of automated container terminals, contrasting LiDAR-based sampling of the environment with camera-based alternatives; covers container pose for pickup, loading-area localization on trucks, profile scanning, and open research gaps.
- **Notes**: T2. Best single entry point into the field. Read first before deciding sensor modality.
---

### [fu2019-monocularpose] Position-Pose Measurement of Crane Sway Based on Monocular Vision
- **Authors**: Fu et al.
- **Venue**: IET Journal of Engineering, 2019
- **URL**: https://digital-library.theiet.org/doi/full/10.1049/joe.2019.1072
- **Status**: discovered
- **Topics**: monocular, non-linear-optimization, sway, pose
- **Abstract**: Monocular-vision method estimating position and pose of a crane spreader via a non-linear optimization on spatial pose parameters; validated on an experimental rig.
- **Notes**: T2. Classical PnP-style pipeline; good baseline comparator.
---

### [kawai2010-imagesensor] Position Measurement of a Crane Spreader Using an Image Sensor System for Anti-Sway Controllers
- **Authors**: H. Kawai et al.
- **Venue**: IEEJ Transactions on Industry Applications, 130(1), 2010
- **URL**: https://ui.adsabs.harvard.edu/abs/2010IJTIA.130..102K
- **Status**: discovered
- **Topics**: image-sensor, spreader-position, anti-sway
- **Abstract**: Image-sensor system to measure spreader position and height for closed-loop suppression of sway motion.
- **Notes**: T2. Foundational reference for consuming pose in a control loop.
---

### [schaper2014-2dof-skew] 2-DOF Skew Control of Boom Cranes Including State Estimation and Reference Trajectory Generation
- **Authors**: M. Schaper, E. Arnold, O. Sawodny (group)
- **Venue**: Control Engineering Practice, 2014
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0967066114002202
- **Status**: discovered
- **Topics**: skew-control, state-estimation, harbor-crane
- **Abstract**: 2-DOF skew controller for boom cranes with state estimation and reference trajectory generation; experimental results on a full-size harbor crane.
- **Notes**: T2. Control-theoretic counterpart to the vision papers — defines what pose signal the controller actually needs.
---

### [ngoHong-skewcontrol] Skew Control of a Container Crane
- **Authors**: Q.H. Ngo, K.-S. Hong
- **Venue**: ResearchGate / journal version (pub. 224354748)
- **URL**: https://www.researchgate.net/publication/224354748_Skew_control_of_a_container_crane
- **Status**: discovered
- **Topics**: skew, vision-feedback, PID, fuzzy
- **Abstract**: Skew-motion modeling and control of container cranes; uses a vision system to measure skew and drives anti-skew devices via PID / fuzzy controllers.
- **Notes**: T2.
---

### [tanakaKaneko-badweather] Measurement of a Container Crane Spreader Under Bad Weather Conditions by Image Restoration
- **Authors**: (Tanaka, Kaneko et al.)
- **Venue**: (as listed on RG, pub. 220408582)
- **URL**: https://www.researchgate.net/publication/220408582
- **Status**: discovered
- **Topics**: image-restoration, rain/fog, spreader-measurement
- **Abstract**: Image-restoration preprocessing to robustify spreader measurement in rain/fog.
- **Notes**: T2. Important for 24/7 outdoor deployment.
---

### [dl2019-cornercasting] Deep Learning–Assisted Real-Time Container Corner Casting Recognition
- **Authors**: (authors not extracted)
- **Venue**: 2019 (RG pub. 330363226)
- **URL**: https://www.researchgate.net/publication/330363226
- **Status**: discovered
- **Topics**: corner-casting, detection, real-time
- **Abstract**: CNN detector for ISO-1161 corner castings to support automated crane alignment.
- **Notes**: T2. Detection-only (no pose), but a key building block.
---

### [lockholeCNN2021] Recognition and Positioning of Container Lock Holes for Intelligent Handling Terminal Based on Convolutional Neural Network
- **Authors**: (authors not extracted)
- **Venue**: 2021 (RG pub. 351557616)
- **URL**: https://www.researchgate.net/publication/351557616
- **Status**: discovered
- **Topics**: lockhole, CNN, detection+2D-position
- **Abstract**: CNN-based recognition and 2D positioning of container lock holes for intelligent terminal handling.
- **Notes**: T2. Precursor to the YOLO11+EPnP work (`mi2025-yolo11-epnp`).
---

### [coilworkpieces2025-overhead] Pose Estimation of Coil Workpieces by Automated Overhead Cranes
- **Authors**: (MDPI Sensors authors)
- **Venue**: Sensors 25(5):1462, 2025
- **URL**: https://www.mdpi.com/1424-8220/25/5/1462  (mirror https://pmc.ncbi.nlm.nih.gov/articles/PMC11902388/)
- **Status**: discovered
- **Topics**: point-cloud, overhead-crane, pose-estimation, non-container
- **Abstract**: 3-D point-cloud pose estimation of steel-coil workpieces under automated overhead cranes.
- **Notes**: T3 (cross-domain). Same pickup-pose problem on a different object class; methods transfer directly.
---

### [xin2022-uavlanding-review] Vision-Based Autonomous Landing for the UAV: A Review
- **Authors**: Xin et al.
- **Venue**: Aerospace (MDPI) 9(11):634, 2022
- **URL**: https://www.mdpi.com/2226-4310/9/11/634
- **Status**: discovered
- **Topics**: UAV, autonomous-landing, cooperative-markers, review
- **Abstract**: Reviews vision-based autonomous landing, categorized by cooperative vs natural targets and static vs dynamic landing zones.
- **Notes**: T3. The "spreader approaching a container" problem is structurally the same as a UAV approaching a landing pad; Hough lines, Canny, PnP pipelines map 1:1.
---

### [kalaitzakis2021-fiducial] Fiducial Markers for Pose Estimation: Overview, Applications and Experimental Comparison
- **Authors**: M. Kalaitzakis, et al.
- **Venue**: Journal of Intelligent & Robotic Systems, 2021
- **URL**: https://link.springer.com/article/10.1007/s10846-020-01307-9
- **Status**: discovered
- **Topics**: fiducial-markers, AprilTag, ArUco, WhyCon
- **Abstract**: Comparative review of fiducial marker families (AprilTag, ArUco, STag, WhyCon/WhyCode) for 6-DoF pose estimation.
- **Notes**: T3. If the deployment allows passive markers on corner castings or the spreader, this is the fastest path to cm-level pose.
---
