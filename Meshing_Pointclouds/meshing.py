import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import sys

pcd = o3d.io.read_point_cloud("/Users/michael/Dev/OpenSfM/data/2/undistorted/depthmaps/merged.ply", format='ply')
pcd.normals = o3d.utility.Vector3dVector(np.zeros((1,3)))
print("Estimate Normals")

pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(100)
print("Poisson Meshing")


with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 15)
o3d.visualization.draw_geometries([mesh], point_show_normal = True)




radii = [0.1]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([mesh], point_show_normal = True)


o3d.io.write_triangle_mesh("test.obj", mesh)

# o3d.visualization.draw_geometries([pcd], point_show_normal = True)
                                  # zoom=0.9412,
                                  # front=[0.4257, -0.2125, -0.8795],
                                  # lookat=[2.6172, 1.0475, 1.532],
                                  # up=[-0.0694, -0.9768, 0.2024]