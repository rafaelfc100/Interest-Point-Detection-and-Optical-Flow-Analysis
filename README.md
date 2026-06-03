# Interest Point Detection and Optical Flow Analysis

This project explores classical Computer Vision techniques for feature detection, matching, and motion estimation in image sequences. The objective is to identify salient points within images and analyze their displacement across consecutive frames using optical flow algorithms.

The work is divided into two main components:

## Objective 1: Interest Point Detection

Different feature detection algorithms were implemented and compared to identify distinctive regions within images.

### FAST (Features from Accelerated Segment Test)

FAST is a corner detection algorithm designed for high-speed applications. It evaluates the intensity of pixels surrounding a candidate point and determines whether it belongs to a corner.

Characteristics:

* Extremely fast execution.
* Suitable for real-time applications.
* Low computational cost.

### Harris Corner Detector

The Harris detector identifies image regions where intensity changes significantly in multiple directions.

Characteristics:

* Robust corner detection.
* Good localization accuracy.
* Widely used in image matching and tracking.

### SIFT (Scale-Invariant Feature Transform)

SIFT detects and describes local image features that remain stable under scale, rotation, and illumination changes.

Characteristics:

* Scale invariant.
* Rotation invariant.
* Generates highly discriminative descriptors.
* Suitable for object recognition and image registration.

## Objective 2: Optical Flow Estimation

Optical flow techniques were applied to estimate pixel displacement between consecutive frames of a video sequence.

### Sparse Optical Flow

The movement of selected interest points was tracked through time.

Applications:

* Object tracking.
* Motion analysis.
* Scene understanding.

### Motion Visualization

Displacement vectors were drawn over image frames to visualize:

* Direction of movement.
* Magnitude of displacement.
* Dynamic scene behavior.

Different displacement thresholds were evaluated, including:

* Standard optical flow visualization.
* Optical flow with minimum displacement filtering (4 pixels).
* Background motion analysis.

## Processing Pipeline

```text
Input Video Frame
        ↓
Feature Detection
(FAST / Harris / SIFT)
        ↓
Selection of Keypoints
        ↓
Optical Flow Calculation
        ↓
Motion Vector Estimation
        ↓
Visualization and Analysis
```

## Technologies

* Python
* OpenCV
* NumPy
* Matplotlib

## Results

The project generated:

* FAST keypoint detection maps.
* Harris corner response maps.
* SIFT keypoint visualizations.
* Sparse optical flow trajectories.
* Motion vector overlays.
* Background motion analysis.

## Applications

The techniques explored in this project can be applied to:

* Object tracking.
* Visual odometry.
* Autonomous navigation.
* Video surveillance.
* Robotics.
* Motion analysis.
* Augmented reality.

## Learning Outcomes

This project provides practical experience in:

* Feature detection.
* Keypoint extraction.
* Image matching.
* Motion estimation.
* Optical flow computation.
* Classical Computer Vision algorithms.

## Author

Universidad de Guanajuato – Computer Vision

Rafael Alejandro Frías Cortez
