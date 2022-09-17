import numpy as np
from math import sqrt
import trimesh
import cv2
import os

scaling = False

# Kabsch algorithm - best fit
# https://gist.github.com/oshea00/dfb7d657feca009bf4d095d4cb8ea4be
def scaled_rigid_transform_3D(A, B, scale):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    if scale:
        H = np.transpose(BB) * AA / N
    else:
        H = np.transpose(BB) * AA

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T * U.T

    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R * centroid_B.T + centroid_A.T

    return c, R, t

registered_frame_points = np.load("path/to/registered_frame_3d_points.npy")
frame_mesh = trimesh.load("path/to/frame/mesh.obj")
surface = trimesh.load("path/to/surface/mesh.obj")

segmentation_folder = "path/to/segmentation/folder"



segmentations = []
for filename in os.listdir(segmentation_folder):
    f = os.path.join(segmentation_folder, filename)
    f_name, extension = splitext(filename)
    if os.path.isfile(f) and extension==".png" :
        segmentation = cv2.load(f)
        segmentations.append(f)

surfce_mesh_rail_points = np.asarray()
for segmentation in segmentations:
    #TODO: Project 2d segmentations to 3d mesh surface
    # surfce_mesh_rail_points = intersection of projected 2d ray from segmentation to 3d surface mesh

s, R, t = rigid_transform_3D(surfce_mesh_rail_points, registered_frame_points)
S = scale_matrix(1.23, origin)

T = translation_matrix([1, 2, 3])

R = random_rotation_matrix(np.random.rand(3))

M = concatenate_matrices(T, R, S)
frame_mesh.apply_transform(M)
