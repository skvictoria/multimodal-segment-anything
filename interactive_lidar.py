import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    # Load the point cloud data from a file.
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return point_cloud

def colorize_point_cloud_by_intensity(point_cloud):
    # Normalize intensity values to [0, 1].
    intensities = point_cloud[:, 3]  # Assuming intensity is the fourth column
    max_intensity = np.max(intensities)
    normalized_intensities = intensities / max_intensity

    # Map the normalized intensity values to a colormap (e.g., jet)
    cmap = plt.get_cmap("jet")
    colors = cmap(normalized_intensities)[:, :3]  # Get RGB values from the colormap

    # Create a colored Open3D point cloud
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return colored_pcd

def set_view(vis, front, lookat, up, zoom):
    """
    Set the view for the visualizer window.

    Parameters:
    - vis: open3d.visualization.Visualizer() object
    - front: The front vector of the camera.
    - lookat: The lookat vector of the camera.
    - up: The up vector of the camera.
    - zoom: Zoom value of the camera.
    """
    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

def set_initial_view(vis):
    ctr = vis.get_view_control()
    ctr.set_front([-1, 0, 0])
    ctr.set_lookat([1, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(2)

def pick_points(pcd):
    # This function is used to pick points from the rendered point cloud.
    print("Please pick points using [shift + left click]")
    print("After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    
    vis.add_geometry(pcd)
    vis.register_animation_callback(lambda vis: set_initial_view(vis))
    #set_view(vis, front=[-1, 0, 0], lookat=[1, 0, 0], up=[0, 1, 0], zoom=2)
    vis.run()  # User picks points here. This will block until the user exits the window.
    vis.destroy_window()
    return vis.get_picked_points()

def main(file_path):
    # Load the point cloud, pick points, and print their coordinates.
    point_cloud = load_point_cloud(file_path)
    colored_pcd = colorize_point_cloud_by_intensity(point_cloud)
    picked_points = pick_points(colored_pcd)
    for index in picked_points:
        print("Selected point coordinates:", np.asarray(colored_pcd.points)[index])

# Replace 'path_to_your_kitti_data.bin' with the actual path to your point cloud file.
main('notebooks/velodyne/002697.bin')