####################################################################################################
    # Imports
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np

import pyrealsense2 as rs
from imports.realsense_device_manager import post_process_depth_frame
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

import setup, segmentation, stream

####################################################################################################
    # Main Function
####################################################################################################
def main():
    devices = setup.cameras(width=848, height=480, decimation=1)
    segmentations(devices, magnification=2)

####################################################################################################
    # Functions
####################################################################################################
def rgb_infrared_depth(devices, magnification=2):
    color_imgs, depth_imgs, aligned_depth_imgs, infrared_imgs = stream.get_all_frames_as_images(devices)
    width = color_imgs[0].shape[1]
    height = color_imgs[0].shape[0]
    cv2.namedWindow("Exit by pressing 'Esc'",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exit by pressing 'Esc'", magnification*3*width, magnification*2*height)
    while True:
        img = []
        color_imgs, depth_imgs, aligned_depth_imgs, infrared_imgs = stream.get_all_frames_as_images(devices)
        
        for (color_img, infrared_img, aligned_depth_img) in zip(color_imgs, infrared_imgs, aligned_depth_imgs):
            img.append( np.hstack((color_img, infrared_img, aligned_depth_img)) )

        img = np.vstack((img))

        cv2.imshow("Exit by pressing 'Esc'", img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) == 114:
            return 'repeat'

def segmentations(devices, magnification=1, record=False):
    cfg = setup.detectron()
    predictor = DefaultPredictor(cfg)
    my_metadata = MetadataCatalog.get(cfg['DATASETS']['TEST'][0])
    color_imgs, depth_imgs, aligned_depth_imgs, infrared_imgs = stream.get_all_frames_as_images(devices)
    width = color_imgs[0].shape[1]
    height = color_imgs[0].shape[0]
    cv2.namedWindow("Exit by pressing 'Esc'",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Exit by pressing 'Esc'", magnification*width, magnification*height)

    if record == True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (width, height))

    while True:
        img = []
        color_imgs, depth_imgs, aligned_depth_imgs, infrared_imgs = stream.get_all_frames_as_images(devices)

        for i, (color_img, depth_img, aligned_depth_img) in enumerate(zip(color_imgs, depth_imgs, aligned_depth_imgs)):
            # Segment image   
            prediction = predictor(color_img)
            instances = prediction["instances"].to('cpu')

            # Visualize segmentation masks
            vis_color = Visualizer(color_img, metadata=my_metadata) #
            vis_color_output = vis_color.draw_instance_predictions(predictions=instances)
            color_img = cv2.cvtColor(vis_color_output.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)

            if i == 0 and record == True:
                out.write(color_img)

            vis_depth = Visualizer(aligned_depth_img, metadata=my_metadata)
            vis_depth_output = vis_depth.draw_instance_predictions(predictions=instances)
            aligned_depth_img = cv2.cvtColor(vis_depth_output.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)

            img.append(np.hstack((color_img, aligned_depth_img)))

        img = np.vstack((img))

        cv2.imshow("Exit by pressing 'Esc'", img)
        if cv2.waitKey(1) == 27:
            if record == True:
                out.release()
            cv2.destroyAllWindows()
            break

def curve_fit(pcd, coeff, tracking_points, fig, p1, p2, p3):
    points = pcd_to_points(pcd)
    t = np.linspace(0,1,100)

    if fig == None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1, 1, 2))
        p1 = ax.plot(tracking_points[0,:], tracking_points[1,:], tracking_points[2,:], 'ro', markersize=10)
        p2 = ax.plot(np.polyval(coeff[0], t), np.polyval(coeff[1], t), np.polyval(coeff[2], t), 'k-', linewidth=4,)
        p3 = ax.plot(points[0], points[1], points[2], 'ko', alpha= 0.3, markersize=2)
        ax.set_xlim3d(min(points[0])-0.1, max(points[0])+0.1)
        ax.set_ylim3d(min(points[1])-0.1, max(points[1])+0.1)
        ax.set_zlim3d(min(points[2])-0.1, max(points[2])+0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig, p1, p2, p3

    p1[0].set_data_3d(tracking_points[0,:], tracking_points[1,:], tracking_points[2,:])
    p2[0].set_data_3d(np.polyval(coeff[0], t), np.polyval(coeff[1], t), np.polyval(coeff[2], t))
    p3[0].set_data_3d(points[0], points[1], points[2])

    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, p1, p2, p3

def pcd_to_points(pcd):
    points = np.asarray(pcd.points)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    return x, y, z

def depth_nonblocking(depth_frames):
        img = np.hstack((depth_frames))
        img = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=255/20000), cv2.COLORMAP_JET)
        cv2.imshow("depth frames", img)
        cv2.waitKey(1)

def segmentations_nonblocking(color_frames, masks):
        imgs = []
        for color_frame, mask in zip(color_frames, masks):
            color_frame = cv2.bitwise_or(color_frame, color_frame, mask=mask)

            mask = cv2.erode(mask, kernel=np.ones((3,3)), iterations=3)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            img = np.hstack((color_frame, mask))
            imgs.append(img)
        
        img = np.vstack((imgs))
        cv2.imshow('RGB & masks', img)
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            exit()

if __name__ == "__main__":
    main()