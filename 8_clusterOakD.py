#!/usr/bin/env python3

import depthai as dai
from time import sleep
import numpy as np
import cv2
import time
import sys
import open3d as o3d
import matplotlib.pyplot as plt
from math import radians

class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps

def pointcloud_to_image(pointcloud, image_size):
    """
    Convert a point cloud to an OpenCV depth image while preserving point cloud colors.

    Args:
        pointcloud (open3d.geometry.PointCloud): The input point cloud.
        image_size (tuple): (width, height) of the output image.

    Returns:
        depth_image (np.ndarray): Depth image (grayscale).
        color_image (np.ndarray): Color image (RGB).
    """
    # Extract points and colors from the point cloud
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    if points.shape[0] == 0:
        raise ValueError("The point cloud is empty.")

    # Image dimensions and camera parameters
    image_width, image_height = image_size
    focal_length = 525.0  # Example focal length in pixels
    cx, cy = image_width // 2, image_height // 2  # Principal point (image center)

    # Initialize images
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)
    color_image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    for i, point in enumerate(points):
        x, y, z = point
        if z <= 0:  # Ignore points behind the camera
            continue

        # Project 3D points to 2D (pinhole camera model)
        u = int(focal_length * x / z + cx)
        v = int(focal_length * y / z + cy)

        # Check if the projected point falls within the image bounds
        if 0 <= u < image_width and 0 <= v < image_height:
            depth = z
            color = colors[i]

            # Assign depth and color to the corresponding pixel
            if depth_image[v, u] == 0 or depth < depth_image[v, u]:  # Keep the nearest point
                depth_image[v, u] = depth
                color_image[v, u] = color

    # Normalize depth image for visualization
    depth_image[depth_image == 0] = np.nan  # Set missing values to NaN for clarity
    normalized_depth = (depth_image - np.nanmin(depth_image)) / (
        np.nanmax(depth_image) - np.nanmin(depth_image)
    )
    normalized_depth[np.isnan(normalized_depth)] = 0  # Replace NaN with 0

    # Convert normalized depth to 8-bit for OpenCV visualization
    depth_image_8bit = (normalized_depth * 255).astype(np.uint8)

    # Convert color image to 8-bit for OpenCV visualization
    color_image_8bit = (color_image * 255).astype(np.uint8)

    return depth_image_8bit, color_image_8bit

def pointcloud_to_depth_image(pointcloud, image_size):
    """
    Convert a depth point cloud into a depth image with preserved colors.

    Parameters:
    - pointcloud: Open3D point cloud object with colors.
    - image_size: Tuple (height, width) specifying the output image resolution.

    Returns:
    - depth_image: 2D numpy array of depth values (grayscale).
    - color_image: 3D numpy array (height, width, 3) with RGB colors.
    """
    # Extract points and colors from the point cloud
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)

    # Extract x, y, and depth (z) values
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Normalize x and y to pixel coordinates in the image
    height, width = image_size
    x_normalized = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (width - 1)
    y_normalized = ((y - np.min(y)) / (np.max(y) - np.min(y))) * (height - 1)

    # Convert to integer pixel indices
    x_pixels = np.round(x_normalized).astype(int)
    y_pixels = np.round(y_normalized).astype(int)

    # Initialize depth and color images
    depth_image = np.zeros((height, width), dtype=np.float32)
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Populate the images
    for xi, yi, zi, ci in zip(x_pixels, y_pixels, z, colors):
        if 0 <= xi < width and 0 <= yi < height:
            # Set depth values
            depth_image[yi, xi] = zi

            # Set RGB colors (convert to 0-255 range)
            color_image[yi, xi] = (ci * 255).astype(np.uint8)

    # Normalize depth image for visualization
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = depth_image_normalized.astype(np.uint8)

    return depth_image_normalized, color_image

def generate_depth_image_with_colors(pcd, image_size=(640, 480), intrinsic_matrix=None):
    # Extract 3D points and their colors from point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # If you don't have intrinsic parameters, set a default one
    if intrinsic_matrix is None:
        intrinsic_matrix = np.array([[image_size[0], 0, image_size[0] / 2],
                                     [0, image_size[1], image_size[1] / 2],
                                     [0, 0, 1]])

    # Create empty depth image and color image
    depth_image = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    color_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    for point, color in zip(points, colors):
        # Get the (x, y, z) coordinates and color (R, G, B)
        x, y, z = point
        r, g, b = color  # Color is in range [0, 1], will convert to [0, 255]

        # Project the 3D point to 2D using the intrinsic matrix
        point_2d = np.dot(intrinsic_matrix, np.array([x, y, z]))

        # Normalize the projected coordinates
        u = int(point_2d[0] / point_2d[2])  # x coordinate in 2D
        v = int(point_2d[1] / point_2d[2])  # y coordinate in 2D

        # Check if the 2D point is within the image boundaries
        if 0 <= u < image_size[0] and 0 <= v < image_size[1]:
            # Set the depth value (z-coordinate) at the corresponding pixel
            depth_image[v, u] = z  # Depth value is the z-coordinate of the point

            # Convert color to [0, 255] range and set the color at the corresponding pixel
            color_image[v, u] = [int(r * 255), int(g * 255), int(b * 255)]

    # Normalize depth image for better visualization (optional)
    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image_normalized = np.uint8(depth_image_normalized)

    return depth_image_normalized, color_image

def get_intrinsic(width, height):
    return o3d.core.Tensor([[1, 0, width * 0.5],
                            [0, 1, height * 0.5],
                            [0, 0, 1]])

def get_extrinsic(x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0):
    extrinsic = np.eye(4)
    extrinsic[:3,  3] = (x, y, z)
    extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([radians(rx),radians(ry), radians(rz)])
    return extrinsic

def compute_show_reprojection(pcd, width, height, intrinsic, extrinsic, window_wait=3000):
    depth_reproj = pcd.project_to_depth_image(width,
                                              height,
                                              intrinsic,
                                              extrinsic,
                                              depth_scale=5000.0,
                                              depth_max=10.0)

    cv2.imshow("depth", np.asarray(depth_reproj.to_legacy()))
    return cv2.waitKey(window_wait)

def segment_pcd(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=70, min_points=5, print_progress=False))

    valid_indices = labels >= 0
    pcd = pcd.select_by_index(np.where(valid_indices)[0])

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd

def main():
    FPS = 30

    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    pointcloud = pipeline.create(dai.node.PointCloud)
    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.input.setBlocking(False)


    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setIspScale(1,3)
    camRgb.setFps(FPS)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")
    monoLeft.setFps(FPS)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")
    monoRight.setFps(FPS)

    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(True)
    depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(pointcloud.inputDepth)
    camRgb.isp.link(sync.inputs["rgb"])
    pointcloud.outputPointCloud.link(sync.inputs["pcl"])
    sync.out.link(xOut.input)
    xOut.setStreamName("out")

    with dai.Device(pipeline) as device:
        isRunning = True
        def key_callback(vis, action, mods):
            global isRunning
            if action == 0:
                isRunning = False

        q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_action_callback(81, key_callback)
        pcd = o3d.geometry.PointCloud()
        dwn_geo = o3d.geometry.PointCloud()
        segment_geo = o3d.geometry.PointCloud()
        coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0,0,0])
        vis.add_geometry(coordinateFrame)

        # lookat = [0.0, 0.0, 500.0]  # Center of the scene
        # front = [0.0, 0.0, 1.0]  # Camera direction towards negative z-axis
        # up = [0.0, 1.0, 0.0]  # Up direction (along positive y-axis)
        #
        # # Get the view control object
        # view_control = vis.get_view_control()
        #
        # # Set the camera parameters
        # view_control.set_lookat(lookat)
        # view_control.set_front(front)
        # view_control.set_up(up)
        # view_control.set_zoom(0.455)

        first = True
        fpsCounter = FPSCounter()
        while isRunning:
            inMessage = q.get()
            inColor = inMessage["rgb"]
            inPointCloud = inMessage["pcl"]
            cvColorFrame = inColor.getCvFrame()
            # Convert the frame to RGB
            cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
            fps = fpsCounter.tick()
            # Display the FPS on the frame
            cv2.putText(cvColorFrame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("color", cvColorFrame)
            print("showing images")
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if inPointCloud:
                print("Found point cloud")
                t_before = time.time()
                points = inPointCloud.getPoints().astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(points)
                colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                dwn = pcd.voxel_down_sample(voxel_size=50)
                seg = segment_pcd(dwn)

                dwn_geo.clear()
                dwn_geo.points = dwn.points
                dwn_geo.colors = dwn.colors

                segment_geo.clear()
                segment_geo.points = seg.points
                segment_geo.colors = seg.colors

                # width, height = 640, 480
                # intrinsic = get_intrinsic(width, height)
                # extrinsic = get_extrinsic()
                #
                # x, y, z = -0.95, -0.95, -0.95
                # compute_show_reprojection(segment_geo, width, height, intrinsic, get_extrinsic(x, y, z), 40)

                if first:
                    vis.add_geometry(segment_geo)
                    first = False
                else:
                    vis.update_geometry(segment_geo)
                    print("Updating visual")
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()

main()
