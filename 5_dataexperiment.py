
import open3d as o3d
import numpy as np

def highlight_point(pcd, target_point):
    #target_point = np.array([1.5, 2.0, 0.5])  # Replace with your (x, y, z)
    # Convert point cloud to a NumPy array
    points = np.asarray(pcd.points)

    # Find the closest point in the point cloud
    distances = np.linalg.norm(points - target_point, axis=1)  # Compute distances
    closest_index = np.argmin(distances)  # Index of the closest point
    closest_point = points[closest_index]  # Coordinates of the closest point
    print(f"Closest point found: {closest_point} at index {closest_index}")

    # Create a sphere to mark the point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Small sphere
    sphere.translate(closest_point)  # Move the sphere to the specific point
    sphere.paint_uniform_color([1, 0, 0])  # Red color for the marker
    return sphere

def highlight_vector(vector):
    # Define the start and end points of the vector
    start_point = np.array([0, 0, 0])  # Origin
   # vector = np.array([1, 2, 3])  # Vector direction (for example)

    # Define the end point of the vector (start + vector)
    end_point = start_point + vector

    # Create a LineSet to visualize the vector
    lines = [[0, 1]]  # Line connecting points 0 (start) and 1 (end)
    colors = [[1, 0, 0]]  # Red color for the vector

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start_point, end_point])
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def main():
    print("Load a ply point cloud, print it, and render it")
    front_vector = [0.4257, -0.2125, -0.8795]
    lookat_point = [2.6172, 2.0475, 1.532]
    up_vector = [-0.0694, -0.9768, 0.2024]

    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print(pcd)
    print(np.asarray(pcd.points))
    #o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])
    # Add a coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    point = highlight_point(pcd, np.array(lookat_point))
    front_vector_line = highlight_vector(np.asarray(front_vector))
    up_vector_line = highlight_vector(np.asarray(up_vector))
    # Visualize the point cloud with axes
    o3d.visualization.draw_geometries([downpcd, coordinate_frame, point, front_vector_line, up_vector_line], zoom=0.3412, front=front_vector, lookat=lookat_point, up=up_vector, point_show_normal=True)

main()
