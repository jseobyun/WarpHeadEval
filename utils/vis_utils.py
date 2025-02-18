import numpy as np
import open3d as o3d


def make_pcd(points, colors=None):

    points = points.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    return pcd