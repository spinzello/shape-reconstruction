import time
import numpy as np
import open3d as o3d
import matplotlib.cm as plt

def main():
    meshes = []
    pcds = []
    csys = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # Load point-clouds
    pcds.append(o3d.io.read_point_cloud("point_cloud_data/pcd_1.pcd"))
    pcds.append(o3d.io.read_point_cloud("point_cloud_data/pcd_2.pcd"))
    pcds.append(o3d.io.read_point_cloud("point_cloud_data/pcd_3.pcd"))

    for pcd in pcds:
        # Estimate normals
        param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10)
        pcd.estimate_normals(search_param=param)
        pcd.orient_normals_consistent_tangent_plane(15)
        normals = np.asarray(pcd.normals)
        CoM = np.mean(np.asarray(pcd.points), axis=0)
        position_vectors = np.asarray(pcd.points) - CoM
        mean_dot_product = np.sum(normals*position_vectors)
        if mean_dot_product < 0:
            pcd.normals = o3d.utility.Vector3dVector(-normals)

        # Reconstruct surface
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)
        meshes.append(color_mesh_with_densities(mesh, densities))

    # Visualize point-cloud
    o3d.visualization.draw_geometries([meshes[0], pcds[1], pcds[2]], mesh_show_wireframe=True, mesh_show_back_face=True, point_show_normal=False)

def create_mesh_from_point_cloud(pcd):
    # Estimate normals
    param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10)
    pcd.estimate_normals(search_param=param)
    pcd.orient_normals_consistent_tangent_plane(15)
    normals = np.asarray(pcd.normals)
    CoM = np.mean(np.asarray(pcd.points), axis=0)
    position_vectors = np.asarray(pcd.points) - CoM
    mean_dot_product = np.sum(normals*position_vectors)
    if mean_dot_product < 0:
        pcd.normals = o3d.utility.Vector3dVector(-normals)
    
    # Reconstruct surface
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5)

    # Change color so that it showŝ the mesh density
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)

    return mesh

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

def color_mesh_single_color(mesh):
    # Visualize mesh with one color
    colors = np.zeros((np.asarray(mesh.vertices).shape[0], 3))
    colors[:,2] = 1
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def exit_program(arg):
    quit()

if __name__ == '__main__':
    main()