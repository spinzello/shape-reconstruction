# MaST: Marker-less Shape Tracking
![Image](/imgs/tracking.png)
This repository contains code that was created as part of Sebastian Pinzello's master's thesis.

### Purpose
The purpose of this project is to establish a way of tracking a soft robotic arm without the need for a motion-capture system.

### Idea
The main idea behind the tracking approach is to
1. use RGB-D cameras to record depth data alongside RGB
2. perform semantic segmentation on the RGB frames and apply the masks to the depth frames
3. calibrate the cameras extrinsically
4. create a 3D point cloud from the masked depth frames
5. fit the point cloud with a curve
6. track certain points on the curve

## Code Overview
The code is split up into multiple functions. The functions are thematically sorted into various files.

### main.py
The tracking is started from the main.py file. Various options can be toggled by setting the variables in the beginning of the script to _False_ or _True_:

- **tracking_time**:            The tracking duration in seconds.
- **masking**:                  Whether to apply segmentation or not. If turned off, the entire scene is reconstructed as a point-cloud. In this case, _fit_curve_ should be disabled as well.
- **postprocess**:              Whether to post process the point cloud (voxel downsampling, filtering outliers, clustering)
- **visualize_pointcloud**:     Whether to show an updating point-cloud while tracking using an Open3D visualization windows.
- **save_pointcloud**:          Whether to save each point-cloud frame (saved as .xyz in: "/output/pcds")
- **fit_curve**:                Whether to fit a curve to the point-cloud.
- **visualize_curve_fit**:      Whether to show a matplotlib plot of the point-cloud, the curve, and the tracking points during tracking.
- **save_tracking_points**:     Whether to save all the tracking data (saved as .pkl in: "/output/tracking_points")
- **save_video**:               Whether to save a video containing all the RGB and mask frames. If turned on, all these frames are stacked into a single frame. (Saved as .avi in: "/output/videos")

### setup.py
Contains a function **cameras()** to set up the Realsense cameras and a function **detectron()** to set up Detectron2.

**cameras()**: Here, one can set the resolution, framerate, and camera preset (found in "/calibration") to be used for tracking, as well as the magnitude of a decimation filter (reduces size of RGB & depth) and object class variable (if using multi class segmentation).

**detectron()**: This function registers a desired data set and loads its metadata. In addition, one can set the detections per image, confidence threshold, and the model weights to be loaded.

### calibration.py
This script performs the extrinsic calibration. It contains three functions, two different calibration approaches, and a function to load an existing calibration file (to be used in other files).
- **apriltag_calibration()**: This is the default calibration method that will be executed when running this file. It requires the calibration box to be placed in view of every Realsense camera, so that each camera sees an AprilTag.
- **chessboard_calibration()**: This calibration method works by placing a chessboard of known size into the view of the cameras.

### stream.py
Contains two functions that can be called to acquire RGB and depth data from the Realsense cameras.
- **get_all_frames_as_images()**: This function acquires color, depth, and infrared frames from the Realsense cameras. Post-processing an a colormap is applied to the depth stream. Therefore, this merely is used to visualize the quality of the depth. The function also creates a depth frame which is aligned to the RGB frame and outputs it separately.
- **get_aligned_frames_as_numpy()**: This function streams color and aligned depth images. Post-processing is applied to the depth, but no colormap is used. This function provides the data for the tracking pipeline.

### show.py
This file containts all the functions that visualize something. It can be run itself, rgb_infrared_depth() is set to run when running the entire file.
- **rgb_infrared_depth()***: Creates an OpenCV windows that shows live RGB and infrared frames. This is a blocking function which runs until it is exited by pressing "Escape".
- **segmenations()***: Creates an OpenCV windows that shows the current segmentation overlaid on the RGB and colored depth frames. This is a blocking function which runs until it is exited by pressing "Escape".
- **curve_fit()**: Creates a non-blocking MatPlotLib windows that shows the point-cloud, fitted curve, and tracking points. If run multiple times, it updates the plot's content.
- **depth_nonblocking()**: If used in a loop it shows the colored live depth frames.
- **segmentations_nonblockin()**: If used in a loop, it shows the live segmentations mask applied on the RGB image.

### kalman.py
This file contains a function to initialize the Kalman filter and one to apply it repeatedly in a loop.
- **initialize()**: Creates the initial state and covariance matrix
- **apply()*: Applies the kalman filter to a new segmentation mask. It takes in the previous Kalman parameters (state & covariance) and calculates new ones based on the new mask and its confidence.

### segmentation.py
Contains the code for the mask creation and the tracking algorithm that makes sure that the correct mask from the segmentation model's output is chosen.

### pointcloud.py
Contains all the functions that create or process the point-cloud.
- **create()**: Creates a point-cloud from depth frames, the calibration, and the segmentation masks (optional).
- **post_process()**: Applies downsampling, statistical outlier removal, and clustering to an existing point-cloud.
- **cluster_points()**: Contains the code that performs the clustering of the point-cloud (removes small blobs). Is used in the post_process() function.
- **pcd_to_points()**: Separates the X-, Y-, and Z-coordinates of the point-cloud. Useful for certain operations.
- **save()**: Saves the point-cloud to "output/pcds/" as .xyz files.

### /dataset_creation
This folder contains all the functions that were used to create the segmentation dataset.
- **01_video2images.py**: Extracts single images from a video.
- **02_segment_images.py**: Takes images of SoPrA, segments them using thresholding, and overlays them onto random images from the COCO segmentation dataset. The segmented SoPrA image is resized randomly and randomly placed on to the background image. The boundary is between SoPrA and background is blurred to create a seamless transition. The overlays serve as input images to the segmentation model. Separate ground truth masks are created. Artificial occlusions can be added if desired.
- **03_split_dataset.py**: Splits the dataset into training and testing sets according to desired ratio.
- **04_create_json.py**: Creates the json-file needed for training that contains all the necessary metadata.

### /training
All the scripts to train segmentation models with Detectron2 can be found here.
- **01_train_detectron.py**: This script starts training of a segmentation model for a custom dataset creation with the scripts from "/dataset_creation".
- **02_test_detectron.py**: This script evaluates the trained model on a test set. (Has not always worked.)
- **03_perform_inference.py**: This script goes through a directory of input images and segments them one after another. The script overlays the segmented mask on the input images and saves them separately in a different folder.

### /calibration
Here, the Realsense presets, the saved calibration files, and images of the calibration patterns can be found.

### /experiments
All the data collected for various experiments can be found here. There's also a script called **evaluate_results.py** to evaluate the saved data and plot results.

### /imports
All the imports that are used (mainly Realsense funcions) are saved here.

### /trained_models
The weights and training history of successful models are saved here.

### /meshing
Some experiments to create 3D surface meshes were tested. The code is saved in this folder.

###/imgs
Containts the image for this README.md
