####################################################################################################
    # Imports
####################################################################################################
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
from detectron2.engine.defaults import DefaultPredictor

import calibration, curve, kalman, pointcloud, segmentation, setup, show, stream
####################################################################################################
    # Main Function
####################################################################################################
def main():
    # Setup
    devices = setup.cameras()
    calibr = calibration.load(devices)

    # Settings
    tracking_time = 120 # [s]
    masking                 = True
    postprocess             = True
    visualize_pointcloud    = False
    save_pointcloud         = False

    fit_curve               = True
    visualize_curve_fit     = False
    save_tracking_points    = True

    save_video              = True

    # Initialize masking predictor
    if masking:
        print("Initializing predictor...\n")
        object_position = None
        object_count = None
        cfg = setup.detectron()
        color_frames, depth_frames = stream.get_aligned_frames_as_numpy(devices)
        predictor = DefaultPredictor(cfg)
        # state, covariance = kalman.initialize(devices, color_frames)
    else:
        predictor = None
        masks = None
   
    print("Start tracking...\n")
    timings = [] # Initialize list to save loop timings
    start_recording = time.time()
    tracking_point_list = []
    pcd = o3d.geometry.PointCloud()
    vis = None
    fig = None
    p1 = None
    p2 = None
    p3 = None

    if save_video:
        stacked_vertically = 3
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/videos/video.avi', fourcc, 5.0, (len(color_frames)*color_frames[1].shape[1], stacked_vertically*color_frames[0].shape[0]))

    while time.time()-start_recording < tracking_time:
        start_loop = time.time()
        color_frames, depth_frames = stream.get_aligned_frames_as_numpy(devices)
        
        # Perform masking if needed
        if masking:
            masks, confidences, object_position, object_count = segmentation.run(color_frames, depth_frames, predictor, object_position, object_count, calibr)
            # masks, state, covariance = kalman.apply(masks, state, covariance, confidences)

        # Create point-cloud
        pcd = pointcloud.create(depth_frames, calibr, masks, pcd, z_cutoff=-0.33)
        if postprocess: pcd = pointcloud.post_process(pcd)
        if save_pointcloud: pointcloud.save(pcd, start_loop)
        
        # Fit curve
        if fit_curve:
            coeff = curve.fit(pcd, order=4)
            tracking_points, curve_length = curve.track_points(coeff, point_count=3, distr='equal')
            tracking_point_list.append([tracking_points, coeff, time.time()])
        print_timing(timings, start_loop)
        if visualize_curve_fit: fig, p1, p2, p3 = show.curve_fit(pcd, coeff, tracking_points, fig, p1, p2, p3)

        if save_video:
            segmented_depth_frames = []
            segmented_rgb_frames = []
            for color_frame, depth_frame, mask in zip(color_frames, depth_frames, masks):
                segmented_rgb_frames.append(cv2.bitwise_or(color_frame, color_frame, mask=mask))
                depth_frame = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=255/20000), cv2.COLORMAP_JET)
                segmented_depth_frames.append(cv2.bitwise_or(depth_frame, depth_frame, mask=mask))

            rgb_frames = np.hstack(color_frames)
            segmented_depth_frames = np.hstack((segmented_depth_frames))
            segmented_rgb_frames = np.hstack(segmented_rgb_frames)
            out.write(np.vstack((rgb_frames, segmented_rgb_frames, segmented_depth_frames)))

        # Update visualization
        if visualize_pointcloud: vis = show_pointcloud(pcd, vis)
        # show.segmentations_nonblocking(color_frames, masks)
        # show.depth_nonblocking(depth_frames)
    
    # Save data
    if save_tracking_points: curve.save_points(tracking_point_list)
    if save_video: out.release()

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def show_pointcloud(pcd, vis):
    # This function needs to be in the main thread/loop due to open3d limitations
    if vis == None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(ord("S"), quit)
        vis.create_window()
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        ctr.set_up((0, 0, -1))
        ctr.set_front((0, 1, 0))
        ctr.set_lookat((0, 0, 0))
        ctr.set_zoom(1)
    else:
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
    return vis

def print_timing(timings, start):
    timings.append(time.time()-start)
    mean_time = np.mean(timings[-10:-1])
    print("Time: {0:.3f}  FPS: {1:.1f}".format(mean_time, 1/mean_time))

if __name__ == "__main__":
    main()
    print("Finished")
