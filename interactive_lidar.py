import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from kittiObjectFinder import KittiObjectFinder

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

def create_bounding_box(center, extents, R=np.eye(3), color=[1,0,0]):
    """
    Create a bounding box given the center, width, height, and length.

    Parameters:
    - center: The center of the bounding box (x, y, z).
    - extents: Width, height, and length of the bounding box.
    """
    center = np.array(center, dtype=np.float64)
    extents = np.array(extents, dtype=np.float64)
    # bbox = o3d.geometry.OrientedBoundingBox(center, R, extents)
    # # Create a LineSet representation of the bounding box to add color
    # lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(bbox)
    # lines.colors = o3d.utility.Vector3dVector([color for i in range(len(lines.lines))])
    # return lines
    # Create an axis-aligned bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - extents / 2, 
                                               max_bound=center + extents / 2)
    aabb.color = color
    return aabb

def main(file_path, all_objects):
    # Load the point cloud, pick points, and print their coordinates.
    point_cloud = load_point_cloud(file_path)
    colored_pcd = colorize_point_cloud_by_intensity(point_cloud)
    
    # Visualize point cloud and bounding box
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(colored_pcd)
    for i in range(len(all_objects)):
        bbox = create_bounding_box(all_objects[i]["center"], all_objects[i]["dimensions"])
        print(all_objects[i]["center"], all_objects[i]["dimensions"])
        #print(bbox)
        vis.add_geometry(bbox)
    # Set the view so the bounding box is visible
    set_view(vis, front=[-6, -6, -6], lookat=[0, 0, 0], up=[0, 1, 0], zoom=0.5)

    # Pick points
    print("Please pick points using [shift + left click]")
    print("After picking points, press 'Q' to close the window")
    vis.run()  # User picks points here. This will block until the user exits the window.
    vis.destroy_window()

    picked_points = vis.get_picked_points()
    for index in picked_points:
        print("Selected point coordinates:", np.asarray(colored_pcd.points)[index])

    finder = KittiObjectFinder(annotation_file)
    selected_objects = finder.find_objects_containing_points(picked_points, colored_pcd.points)
    for obj in selected_objects:
        print(obj)


annotation_file = 'notebooks/label/002697.txt'  # Path to the annotation file
finder = KittiObjectFinder(annotation_file)
all_objects = finder.find_objects_with_labels()

main('notebooks/velodyne/002697.bin', all_objects)