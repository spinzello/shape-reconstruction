####################################################################################################
    # Imports
####################################################################################################
import time
import numpy as np
import pickle
import cv2
from scipy.spatial.transform import Rotation as R

import pyrealsense2 as rs
from imports.calibration_kabsch import PoseEstimation, Transformation
from imports.apriltag.apriltag import apriltag


import setup
####################################################################################################
    # Main Function
####################################################################################################

def main():
    apriltag_calibration()

####################################################################################################
    # Functions
####################################################################################################
def apriltag_calibration():
    tag_width_in_meter = 0.069

    devices = setup.cameras(width=1280, height=720, fps=30, decimation=1)
    devices.enable_emitter(on=False, always=False, max_power=True)
    frames = devices.poll_frames()
    intrinsics = devices.get_device_intrinsics(frames)
    detector = apriltag("tagStandard41h12")
    transformations = []

    print("Calibrating cameras using apriltag box...")
    for frame_idx, (frame, intr) in enumerate(zip(frames.items(), intrinsics.items())):
        frame = np.array(frame[1][rs.stream.color].get_data())

        # Detect tag
        detections = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(detections) == 0:
            print("Camera", frame_idx,": No april tags found!")
            continue
        else:
            # Find largest tag
            corners = []
            circumference = []
            for detection in detections:
                corners.append(detection['lb-rb-rt-lt'])
                circumference.append(np.sum(np.linalg.norm(np.roll(corners[-1], 1, axis=0) - corners[-1], axis=1)))
            best_detection_idx = np.argmax(circumference)
            detection = detections[best_detection_idx]
            corners = corners[best_detection_idx][None, ::-1].astype(np.float32)
            print("Camera", frame_idx,":", len(detections),"april tag(s) found ( chose tag with ID", detection['id'],")")

            # Get camera intrinsics
            camera_matrix = np.array([intr[1][2].fx, 0.000000, intr[1][2].ppx, 0.000000, intr[1][2].fy, intr[1][2].ppy, 0.000000, 0.000000, 1.000000]).reshape(3, 3)
            dist_coeffs = np.array(intr[1][2].coeffs)

            # Estimate pose of tag (transformtation from camera to tag)
            rvecs, tvecs, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, tag_width_in_meter, camera_matrix, dist_coeffs)
            transf_to_tag = np.eye(4)
            transf_to_tag[:3,:3] = R.from_rotvec(np.squeeze(rvecs)).as_matrix()
            transf_to_tag[:3,3] = np.squeeze(tvecs)

            # Choose transformation to box corner depending on tag id
            if detection['id'] == 0:
                transf = np.eye(4)
                translation = np.array([-0.0965, 0.1345, 0.0]) 
                r1 = R.from_euler('x', 180, degrees=True)
                r2 = R.from_euler('z', 0, degrees=True)
                rotation = (r1*r2).as_matrix()
                transf[:3,:3] = rotation
                transf[:3,3] = translation
            elif detection['id'] == 1:
                transf = np.eye(4)
                translation = np.array([0.062, 0.1345, 0.0])
                r1 = R.from_euler('y', -90, degrees=True)
                r2 = R.from_euler('z', 180, degrees=True)
                rotation = (r1*r2).as_matrix()
                transf[:3,:3] = rotation
                transf[:3,3] = translation
            elif detection['id'] == 2:
                transf = np.eye(4)
                translation = np.array([0.0965 - 0.005, 0.062, -0.02]) # Translation in z-direction is to correct tendency of depth data to be closer 
                r1 = R.from_euler('x', 90, degrees=True)
                r2 = R.from_euler('z', 180, degrees=True)
                rotation = (r1*r2).as_matrix()
                transf[:3,:3] = rotation
                transf[:3,3] = translation
            else:
                print('Aborted due to unknown tag.')
                return

            # Apply new transformation to previous transformation
            transf_to_corner = np.linalg.inv(np.matmul(transf_to_tag, transf))

            # Plot coordinate system for verification
            new_rvecs = R.from_matrix(transf_to_corner[:3,:3]).as_rotvec()
            new_tvecs = transf_to_corner[:3,3]
            img_csys = cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, new_rvecs, new_tvecs, 0.08)
            img_csys = cv2.aruco.drawAxis(img_csys, camera_matrix, dist_coeffs, rvecs, tvecs, 0.02)
            cv2.imwrite('output/box_calibration/csys_cam_{}.png'.format(frame_idx), img_csys)

            # Save transformation as Transformation object
            transf_to_corner = Transformation(transf_to_corner[:3,:3], transf_to_corner[:3,3])
            transformations.append([transf_to_corner])

    with open('calibration/calibration.pkl', 'wb') as f:
        pickle.dump(transformations, f)

    print("Cameras are calibrated! (calibration saved to 'calibration/calibration.pkl')\n")
    return

def chessboard_calibration():
    devices = setup.cameras()

    chessboard_width = 4 # squares (normal: 6, large :4)
    chessboard_height = 5 	# squares (normal: 9, large: 5)
    square_size = 0.0423 # meters old: 0.0248, large board: 0.0423
    dispose_frames_for_stablisation = 30  # frames

    # Allow some frames for the auto-exposure controller to stablise
    for frame in range(dispose_frames_for_stablisation):
        frames = devices.poll_frames()

    device_extrinsics = []
    device_2_color = devices.get_depth_to_color_extrinsics(frames)
    for serial, value in device_2_color.items():
        device_extrinsics.append(Transformation(np.reshape(value.rotation, (3,3)), np.array(value.translation)).inverse())
    
    # Get the intrinsics of the realsense device
    device_intrinsics = []
    intrinsics = devices.get_device_intrinsics(frames)
    for key, value in intrinsics.items():
        device_intrinsics.append(value[rs.stream.color])

    # Get a clean infrared image without IR patter (makes it easier to find chessboard corners)
    devices.enable_emitter(False)
    time.sleep(0.1) # Gives the emitter time to toggle
    frames = devices.poll_frames()
    infrared_frames = []
    for (serial, frameset) in frames.items():
        infrared_frames.append(frameset[(rs.stream.infrared, 1)])

    # Get a good depth frame (requires IR pattern again)
    devices.enable_emitter(True)
    time.sleep(0.1) # Gives the emitter time to toggle
    frames = devices.poll_frames()
    depth_frames = []
    serials = []
    for (serial, frameset) in frames.items():
        depth_frames.append(frameset[rs.stream.depth])
        serials.append(serial)

    # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
    chessboard_params = [chessboard_height, chessboard_width, square_size]
    calibrated_device_count = 0
    while calibrated_device_count < len(devices._available_devices):
        pose_estimator = PoseEstimation(depth_frames, infrared_frames, serials, intrinsics, chessboard_params)
        transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
        calibrated_device_count = 0
        for device in devices._available_devices:
            if not transformation_result_kabsch[device][0]:
                print("Place the chessboard on the plane where the object needs to be detected..")
            else:
                calibrated_device_count += 1

    world_transformation = []
    for key, value in transformation_result_kabsch.items():
        world_transformation.append(value[1].inverse())


    # Create list with all calibrations for each device
    calibration_info_devices = [[] for device in range(len(devices._available_devices))]
    for device in range(len(devices._available_devices)):
        calibration_info_devices[device].append(world_transformation[device])
        calibration_info_devices[device].append(device_extrinsics[device])
    
    # Contains a list for each device [world_transformation, device_intrinsics, device_extrinsics]
    with open('calibration/calibration.pkl', 'wb') as f:
        pickle.dump(calibration_info_devices, f)

    print("Cameras are calibrated! - Calibration saved to calibration.pkl\n")

def load(devices):
    # Load extrinsic calibration from file
    with open('calibration/calibration.pkl', 'rb') as f:
        calibration_info_devices = pickle.load(f)

    # Get the intrinsics of the realsense device
    frames = devices.poll_frames()
    intrinsics = devices.get_device_intrinsics(frames)
    for idx, (key, value) in enumerate(intrinsics.items()):
        calibration_info_devices[idx].insert(1, value[rs.stream.color])

    print("Calibration loaded!\n")
    return calibration_info_devices

if __name__ == "__main__":
    main()