
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def segment_pcd(pcd):
    #getting lebels ?
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    print("labels" , labels)

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    #getting colors array
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0

    print("Colors ", colors)

    #assigning colors in the pcd ?
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd

#getting the data in pcd
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
pcd = segment_pcd(pcd)
o3d.visualization.draw_geometries([pcd], zoom=0.455, front=[-0.4999, -0.1659, -0.8499], lookat=[2.1813, 2.0619, 2.0999], up=[0.1204, -0.9852, 0.1215])
