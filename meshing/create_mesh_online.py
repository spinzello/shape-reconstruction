import time
import numpy as np
import open3d as o3d
import matplotlib.cm as plt
from scipy.spatial import distance
import probreg as reg


####################################################################################################
    # Main Function
####################################################################################################

def main():
    pcds = load_point_clouds()
    mesh = load_cad_mesh()
    mesh = initial_alignment(mesh, pcds[0])
    vis = create_visualizer(mesh, pcds[0])
    for pcd in pcds:
        # mesh = new_mesh_for_every_pointcloud(pcd)
        # visualize_static(mesh, pcd)
        
        # mesh = deform_mesh_to_point_cloud_probreg(mesh, pcd)
        mesh = deform_mesh_to_point_cloud(mesh, pcd)
        update_visualizer(vis, mesh, pcd)
        
        
        time.sleep(0.1)

####################################################################################################
    # Auxiliary Functions
####################################################################################################
def deform_mesh_to_point_cloud(mesh, pcd):
    # Sample mesh vertices uniformly
    sample_count = 4
    mesh_ids_along_arm(mesh)
    sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=sample_count)
    sampled_mesh_ids = find_closest_points(source_points=sampled_pcd.points, target_points=mesh.vertices)
    sampled_mesh_points = o3d.utility.Vector3dVector( np.asarray(mesh.vertices)[sampled_mesh_ids,:] )

    # Find closest points to sampled vertices
    closest_pcd_point_ids = find_closest_points(sampled_mesh_points, pcd.points)
    new_sample_positions = o3d.utility.Vector3dVector( np.asarray(pcd.points)[closest_pcd_point_ids,:] )
    sampled_mesh_vectors = np.asarray(new_sample_positions) - np.asarray(sampled_mesh_points)

    # For every point, find distance to all the sample points
    distances_mesh_to_samples = distance.cdist(mesh.vertices, sampled_mesh_points)

    # # Sort based on distance and only choose N nearest neighbors
    # n_neighbors = 3
    # closest_sample_ids = np.argsort(distances_mesh_to_samples, axis=1)[:,:n_neighbors]
    # row_indexes = np.transpose(np.tile(np.arange(closest_sample_ids.shape[0]), (n_neighbors,1)))
    # distances_mesh_to_sample_neighbors = distances_mesh_to_samples[row_indexes, closest_sample_ids]

    # Transform distances into weights
    p = 1
    weights = np.reciprocal(np.power(distances_mesh_to_samples, p))
    summed_weights = np.sum(weights, axis=1)
    summed_weights = np.transpose(np.tile(summed_weights, (sample_count, 1)))
    normalized_weights = np.nan_to_num(weights / summed_weights, nan=1.0)

    # Use weights to calculate translation vectors for all mesh vertices
    mesh_vectors = np.matmul(normalized_weights, sampled_mesh_vectors)
    new_mesh_positions = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + mesh_vectors)
    # mesh.vertices = o3d.utility.Vector3dVector(new_mesh_positions)
    mesh_ids = o3d.utility.IntVector( range(np.asarray(mesh.vertices).shape[0]) )
    mesh = mesh.deform_as_rigid_as_possible(mesh_ids, new_mesh_positions, max_iter=10, energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Spokes)
    return mesh

    print("")
    print("")

def deform_mesh_to_point_cloud_probreg(mesh, target):
    source = o3d.geometry.PointCloud()
    source.points = mesh.vertices
    tf_param,_,_ = reg.cpd.registration_cpd(source=source, target=target, tf_type_name='nonrigid', w=0.0)
    mesh.vertices = tf_param.transform(source.points)
    return mesh

def get_uniform_mesh_ids(mesh):
    pcd = mesh.sample_points_poisson_disk(number_of_points=10)
    pcd.paint_uniform_color([1, 0, 0])
    return pcd

def mesh_ids_along_arm(mesh, count=5):
    # Find mesh ids spread along arm
    vert = np.asarray(mesh.vertices)
    mesh_ids = np.argsort(vert[:,0], axis=0)
    step = int(len(mesh_ids)/5)
    mesh_ids = o3d.utility.IntVector( list(mesh_ids[::step]) )
    return mesh_ids

def deform_mesh_to_point_cloud_simple(mesh, pcd):
    closest_points = find_closest_points(mesh.vertices, pcd.points)
    mesh_ids = o3d.utility.IntVector( range(np.asarray(mesh.vertices).shape[0]) )
    new_positions = o3d.utility.Vector3dVector( np.asarray(pcd.points)[closest_points[mesh_ids],:] )
    mesh = mesh.deform_as_rigid_as_possible(mesh_ids, new_positions, max_iter=10, energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed)
    return mesh

def find_closest_points(source_points, target_points):
    # Finds closest point-cloud point for every mesh node
    D = distance.cdist(source_points, target_points)
    closest_all = np.argsort(D, axis=1)
    closest = closest_all[:,0]
    return closest

def load_cad_mesh():
    mesh = o3d.io.read_triangle_mesh("meshes/soft_arm.ply")
    mesh = mesh.scale(0.001, center=mesh.get_center())
    return mesh

def load_point_clouds():
    pcds = []
    for i in range(328):
        pcds.append(o3d.io.read_point_cloud("recording/pcd/pcd_" + str(i) + ".pcd"))
    return pcds

def new_mesh_for_every_pointcloud(pcd):
    def mesh_from_pcd(pcd, mesh):
        def color_mesh_with_densities(mesh, densities):
            densities = np.asarray(densities)
            density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
            density_colors = density_colors[:, :3]
            density_mesh = o3d.geometry.TriangleMesh()
            density_mesh.vertices = mesh.vertices
            density_mesh.triangles = mesh.triangles
            density_mesh.triangle_normals = mesh.triangle_normals
            density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
            return density_mesh

        new_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
        new_mesh = color_mesh_with_densities(new_mesh, densities)
        mesh.triangles = new_mesh.triangles
        mesh.vertices = new_mesh.vertices
        mesh.vertex_colors = new_mesh.vertex_colors
        mesh.vertex_normals = new_mesh.vertex_normals
        return mesh

    mesh = o3d.geometry.TriangleMesh()

    # Reconstruct surface
    pcd = estimate_normals(pcd)
    new_mesh = mesh_from_pcd(pcd, mesh)

    return new_mesh

def initial_alignment(source_mesh, target_pcd):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = source_mesh.vertices
    source_pcd.normals = source_mesh.vertex_normals

    target_pcd = estimate_normals(target_pcd)
    transl_vec = target_pcd.get_center() - source_pcd.get_center()
    source_pcd = source_pcd.translate(transl_vec)
    reg = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, max_correspondence_distance=0.02, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    source_pcd = source_pcd.transform(reg.transformation)
    
    source_mesh.vertices = source_pcd.points
    return source_mesh

def estimate_normals(pcd):
    param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10)
    pcd.estimate_normals(search_param=param)
    pcd.orient_normals_consistent_tangent_plane(15)
    normals = np.asarray(pcd.normals)
    CoM = np.mean(np.asarray(pcd.points), axis=0)
    position_vectors = np.asarray(pcd.points) - CoM
    mean_dot_product = np.sum(normals*position_vectors)
    if mean_dot_product < 0:
        pcd.normals = o3d.utility.Vector3dVector(-normals)
    return pcd

def create_visualizer(mesh, pcd):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord("S"), exit_program)
    vis.create_window()

    if pcd is not None:
        pcd_for_visualization = o3d.geometry.PointCloud()
        pcd_for_visualization.points = pcd.points
        vis.add_geometry(pcd_for_visualization)
    else:
        pcd_for_visualization = None

    if mesh is not None:
        mesh_for_visualization = o3d.geometry.TriangleMesh()
        mesh_for_visualization.vertices = mesh.vertices
        mesh_for_visualization.triangles = mesh.triangles
        vis.add_geometry(mesh_for_visualization)
    else:
        mesh_for_visualization = None

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.mesh_show_wireframe = True
    return [vis, mesh_for_visualization, pcd_for_visualization]

def update_visualizer(visualizer_objects, mesh, pcd):
    visualizer = visualizer_objects[0]
    mesh_for_visualization = visualizer_objects[1]
    pcd_for_visualization = visualizer_objects[2]

    if pcd is not None:
        pcd_for_visualization.points = pcd.points
        visualizer.update_geometry(pcd_for_visualization)

    if mesh is not None:
        mesh_for_visualization.vertices = mesh.vertices
        mesh_for_visualization.triangles = mesh.triangles
        visualizer.update_geometry(mesh_for_visualization)

    visualizer.poll_events()
    visualizer.update_renderer()

def visualize_static(mesh, pcd):
    csys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d.visualization.draw_geometries([csys, pcd], mesh_show_wireframe=True, mesh_show_back_face=True, point_show_normal=True)

def exit_program(arg):
    quit()

if __name__ == '__main__':
    main()