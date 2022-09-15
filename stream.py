####################################################################################################
    # Imports
####################################################################################################
import cv2
import numpy as np

import pyrealsense2 as rs
from imports.realsense_device_manager import post_process_depth_frame
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import setup

####################################################################################################
    # Functions
####################################################################################################
def get_all_frames_as_images(devices):
    max_dist = 20000 # Defines the colormap limit
    framesets = devices.poll_frameset()
    color_imgs = []
    aligned_depth_imgs = []
    processed_depth_frames = []
    infrared_imgs = []
    align = rs.align(rs.stream.color)

    for frameset in framesets:
        processed_frameset = post_process_depth_frame(frameset, decimation_magnitude=setup.DECIMATION)
        processed_depth_frame = np.asanyarray(processed_frameset.get_depth_frame().get_data())
        processed_depth_frames.append(cv2.applyColorMap(cv2.convertScaleAbs(processed_depth_frame, alpha=255/max_dist), cv2.COLORMAP_JET))

        aligned_frameset = align.process(processed_frameset)
        aligned_depth_img = np.asanyarray(aligned_frameset.get_depth_frame().get_data())
        aligned_depth_imgs.append(cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_img, alpha=255/max_dist), cv2.COLORMAP_JET))

        color_img = np.asanyarray(processed_frameset.get_color_frame().get_data())
        color_imgs.append(color_img)

        infrared_img = np.asanyarray(aligned_frameset.get_infrared_frame().get_data())
        infrared_img = cv2.resize(infrared_img, (color_img.shape[1], color_img.shape[0]))
        infrared_img = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
        infrared_imgs.append(infrared_img)

    return color_imgs, processed_depth_frames, aligned_depth_imgs, infrared_imgs

def get_aligned_frames_as_numpy(devices):
    framesets = devices.poll_frameset()
    color_img = []
    aligned_depth_img = []
    align = rs.align(rs.stream.color)

    for frameset in framesets:
        color_img.append(np.asanyarray(frameset.get_color_frame().get_data()))
        processed_frameset = post_process_depth_frame(frameset, decimation_magnitude=setup.DECIMATION)
        aligned_frameset = align.process(processed_frameset)
        aligned_depth_img.append(np.asanyarray(aligned_frameset.get_depth_frame().get_data()))

    return color_img, aligned_depth_img