####################################################################################################
    # Imports
####################################################################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

import open3d as o3d
from imports.helper_functions import convert_depth_frame_to_pointcloud
####################################################################################################
    # Functions
####################################################################################################
def create(depth_frames, calibration, masks, pcd, z_cutoff):
        point_clouds = []

        if masks is not None:
            i = 1
            for depth_frame, calibr, mask in zip(depth_frames, calibration, masks):
                # Erode mask
                mask = cv2.erode(mask, kernel=np.ones((3,3)), iterations=3)

                # Apply mask
                depth_frame = depth_frame * np.uint8(mask / 255)

                # Create point-cloud
                point_cloud = convert_depth_frame_to_pointcloud(depth_frame, calibr[1]) # Requires device intrinsics

                # # Transform point-cloud's origin from RGB sensor to depth sensor
                # point_cloud = calibr[2].apply_transformation(point_cloud) # Requires device extrinsics

                # Transform point-cloud to world coordinates
                point_cloud = calibr[0].apply_transformation(point_cloud) # Requires world frame transformation

                point_cloud = point_cloud[:,point_cloud[2,:] > z_cutoff]

                point_clouds.append(point_cloud)

        else:

            for depth_frame, calibr in zip(depth_frames, calibration):
                # Create point-cloud
                point_cloud = convert_depth_frame_to_pointcloud(depth_frame, calibr[1]) # Requires device intrinsics

                # # Transform point-cloud's origin from RGB sensor to depth sensor
                # point_cloud = calibr[2].apply_transformation(point_cloud) # Requires device extrinsics

                # Transform point-cloud to world coordinates
                point_cloud = calibr[0].apply_transformation(point_cloud) # Requires world frame transformation

                point_clouds.append(point_cloud)

        pcd.points = o3d.utility.Vector3dVector(np.transpose(np.hstack(point_clouds)))
        return pcd

def post_process(pcd, downsample=0.005):
    pcd.points = pcd.voxel_down_sample(voxel_size=downsample).points
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)
    pcd.points = pcd.select_by_index(ind).points
    pcd = cluster_points(pcd, visualize=False)
    return pcd

def cluster_points(pcd, visualize=False):
    cluster_ids = np.asarray(pcd.cluster_dbscan(eps=0.07, min_points=500)) 
    points = np.asarray(pcd.points)[cluster_ids==0, :]
    pcd.points = o3d.utility.Vector3dVector(points)

    # cluster_ids = np.asarray(pcd.cluster_dbscan(eps=0.07, min_points=500)) # eps = 0.04 works
    # points = np.asarray(pcd.points)[cluster_ids==0, :]
    # pcd_2.points = o3d.utility.Vector3dVector(points)
    # print(len(pcd_2.points))

    if visualize == True:
        max_label = cluster_ids.max()
        colors = plt.get_cmap("tab20")(cluster_ids / (max_label if max_label > 0 else 1))
        colors[cluster_ids < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        if max_label > 0 or max_label == -1:
            print(f"point cloud has {max_label + 1} clusters")
            print("Cluster 0:", np.sum(cluster_ids==0), "Cluster 1:", np.sum(cluster_ids==1))
            o3d.visualization.draw_geometries([pcd])
    return pcd

def pcd_to_points(pcd):
    points = np.asarray(pcd.points)
    return points[:,0], points[:,1], points[:,2]

def save(pcd, start_loop):
    point_cloud_name = "output/pcds/pcd_{}.xyz".format(start_loop)
    o3d.io.write_point_cloud(point_cloud_name, pcd)
    return