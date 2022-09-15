def tsdf(devices, calibration_devices):
    cfg = set_cfg()
    predictor = DefaultPredictor(cfg)
    color_frames, depth_frames = get_aligned_frames_as_numpy(devices)
    masks, confidences, object_position, object_count = get_masks(color_frames, depth_frames, predictor, prev_object_position=None, prev_object_count=None, calibration_info_devices=calibration_devices)

    n_imgs = len(depth_frames)

    # print("Estimating voxel volume bounds...")
    # vol_bnds = np.zeros((3,2))
    # for depth_frame, mask, calibration in zip(depth_frames, masks, calibration_devices):
    #     depth_frame = (depth_frame / 10000.) * (mask/255)
    #     # depth_frame = depth_frame / 10000.
    #     intr = calibration[1]
    #     cam_intr = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    #     cam_pose = np.matmul(calibration[2].pose_mat, calibration[0].pose_mat)
    #     view_frust_pts = fusion.get_view_frustum(depth_frame, cam_intr, cam_pose)
    #     vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    #     vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))

    print("Initializing voxel volume...")
    vol_bnds = np.array([[-0.2, 0.2], [-0.2, 0.2], [-0.3, 0]])
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.001)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i, (depth_frame, color_frame, mask, calibration) in enumerate(zip(depth_frames, color_frames, masks, calibration_devices)):
        print("Fusing frame %d/%d"%(i+1, n_imgs))

        depth_frame = (depth_frame / 10000.) * (mask/255)
        color_frame = cv2.bitwise_or(color_frame, color_frame, mask=mask)

        intr = calibration[1]
        cam_intr = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        cam_pose = np.matmul(calibration[2].pose_mat, calibration[0].pose_mat)

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_frame, depth_frame, cam_intr, cam_pose, obs_weight=1.)
    
    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(norms)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors/255)

    csys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    box = o3d.geometry.PointCloud()
    box_points = np.array([[vol_bnds[0,0], vol_bnds[1,0], vol_bnds[2,0]],
                           [vol_bnds[0,0], vol_bnds[1,1], vol_bnds[2,0]],
                           [vol_bnds[0,1], vol_bnds[1,0], vol_bnds[2,0]],
                           [vol_bnds[0,1], vol_bnds[1,1], vol_bnds[2,0]],
                           [vol_bnds[0,0], vol_bnds[1,0], vol_bnds[2,1]],
                           [vol_bnds[0,0], vol_bnds[1,1], vol_bnds[2,1]],
                           [vol_bnds[0,1], vol_bnds[1,0], vol_bnds[2,1]],
                           [vol_bnds[0,1], vol_bnds[1,1], vol_bnds[2,1]]])
    box.points = o3d.utility.Vector3dVector(box_points)
    o3d.visualization.draw_geometries([csys, mesh, box], mesh_show_wireframe=True, mesh_show_back_face=True, point_show_normal=True)

    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()