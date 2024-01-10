import open3d as o3d
import numpy as np
import os
import plotly.graph_objects as go
import cv2

# Define the rotation matrix around the Y-axis
def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

class bboxMaker:
    def __init__(self, annotation_file, calib_file):
        self.bbox_list = []
        self.annotation_file = annotation_file
        self.calib_file = calib_file

    def _calculate_calib(self):
        with open(self.calib_file, 'r') as f:
            lines = f.readlines()

            P0 = np.array(list(map(float, lines[0].split()[1:])))
            P1 = np.array(list(map(float, lines[1].split()[1:])))
            P2 = np.array(list(map(float, lines[2].split()[1:])))
            P3 = np.array(list(map(float, lines[3].split()[1:])))
            R0_rect = np.array(list(map(float, lines[4].split()[1:])))
            Tr_velo_to_cam = np.array(list(map(float, lines[5].split()[1:])))
            Tr_imu_to_velo = np.array(list(map(float, lines[6].split()[1:])))
            Tr_cam_to_road = np.array(list(map(float, lines[7].split()[1:])))

        
        self.Tr = np.zeros((4, 4))
        self.Tr[:3, :] = Tr_velo_to_cam.reshape(3, 4)
        self.Tr[3, :] = [0, 0, 0, 1]

        self.R0 = np.eye(4)
        self.R0[:3, :3] = R0_rect.reshape(3, 3)

        self.P = P2.reshape(3, 4)

    def _overlay_to_image(self, data, img_path):
        # Read image
        img = cv2.imread(img_path)

        self._calculate_calib()

        #data shall be (nx4) array.
        XYZ1 = np.vstack((data[:, :3].T, np.ones(data.shape[0])))
        xy1 = self.P @ self.R0 @ self.Tr @ XYZ1
        s = xy1[2, :]
        x = xy1[0, :] / s
        y = xy1[1, :] / s
        k = s > 0

        # Display the result
        for i in range(len(s)):
            ix = int(x[i] + 0.5)
            iy = int(y[i] + 0.5)
            if s[i] <= 0 or iy < 0 or iy >= img.shape[0] or ix < 0 or ix >= img.shape[1]:
                continue
            img[iy, ix, :] = [0, 255, 0]

        cv2.imwrite("camera_kitti_overlayed.png", img)

    def _parse_line(self, line):
        parts = line.strip().split()
        if len(parts) == 15:
            # Extract the relevant object parameters
            obj_type = parts[0]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            return {
                "type": obj_type,
                "dimensions": np.array([h, w, l]),
                "center": np.array([x, y, z]),
                "rotation_y": ry
            }
        return None

    def camera_to_lidar(self, x, y, z):
        # This is a placeholder function. You'll need the actual transformation
        # matrix (or its inverse) that converts camera coordinates to LiDAR coordinates.
        # If you don't have this, you'll need to calibrate your sensors to find it.
        # Here we assume an identity matrix (no transformation), which is not correct.
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # Placeholder rotation matrix
        T = np.array([0, 0, 0]) # Placeholder translation vector
        p = np.array([x, y, z])
        p = np.dot(R, p) + T
        return p[0], p[1], p[2]
    
    def _is_point_in_box(self, point):
        # check if a point is inside a 3d box.
        self.bbox_list_maker()

        for vectors in self.bbox_list:
            x_coords = [point[0] for point in vectors]
            y_coords = [point[1] for point in vectors]
            z_coords = [point[2] for point in vectors]

            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)
            min_z = np.min(z_coords)
            max_z = np.max(z_coords)

            if(point[0]<=max_x+1 and point[0]>=min_x-1 and
               point[1]<=max_y+1 and point[1]>=min_y-1 and
               point[2]<=max_z+1 and point[2]>=min_z-1):
                print("Find!! - ", point[0], point[1], point[2])
                return True
        return False 

    def bbox_list_maker(self):
        with open(self.annotation_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                obj = self._parse_line(line)
                if obj:
                    # Compute the rotation matrix
                    R = roty(obj["rotation_y"])
                    
                    # 3D bounding box corners in camera coordinate system
                    #l, w, h = obj["dimensions"]
                    h, w, l = obj["dimensions"]
                    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                    y_corners = [-w/2, w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2]
                    z_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                    
                    # Rotate and translate 3D bounding box corners
                    corners_3d_camera = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
                    corners_3d_camera[0, :] += obj["center"][0]
                    corners_3d_camera[1, :] += obj["center"][1]
                    corners_3d_camera[2, :] += obj["center"][2]
                    
                    # Transform the corners to LiDAR coordinates
                    corners_3d_lidar = np.zeros_like(corners_3d_camera)
                    for i in range(8):
                        corners_3d_lidar[:, i] = self.camera_to_lidar(
                            corners_3d_camera[0, i], corners_3d_camera[1, i], corners_3d_camera[2, i]
                        )
                    
                    # Store the transformed corners
                    self.bbox_list.append(corners_3d_lidar.T)

    def show_point_cloud(self, points, color_axis=3, width_size=1500, height_size=800, coordinate_frame=True):
        '''
        points : (N, 4) size of ndarray (in order to see the intensity value of the lidar)
        color_axis : 0, 1, 2
        '''
        #points = np.asarray(pointcloud.points)
        #colors = np.asarray(pointcloud.colors)
        assert points.shape[1] == 4 #3

        # Create a scatter3d Plotly plot
        plotly_fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                #color=colors,
                size=1,#2
                #color=colors[:,0],
                color=points[:, color_axis], # Set color based on Z-values
                colorscale='jet', # Choose a color scale, jet
                colorbar=dict(title='value') # Add a color bar with a title
            )
        )])

        x_range = points[:, 0].max()*0.9 - points[:, 0].min()*0.9
        y_range = points[:, 1].max()*0.9 - points[:, 1].min()*0.9
        z_range = points[:, 2].max()*0.9 - points[:, 2].min()*0.9

        # Adjust the Z-axis scale
        plotly_fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=x_range, y=y_range, z=z_range), # Here you can set the scale of the Z-axis     
            ),
            width=width_size, # Width of the figure in pixels
            height=height_size, # Height of the figure in pixels
            showlegend=False
        )

        # draw bounding box
        self.bbox_list_maker()
        for vectors in self.bbox_list:
        
            x_coords = [point[0] for point in vectors]
            y_coords = [point[1] for point in vectors]
            z_coords = [point[2] for point in vectors]

            x_coords = np.array(np.mean(x_coords))
            y_coords = np.array(np.mean(y_coords))
            z_coords = np.array(np.mean(z_coords))

            bbox_vertices = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=5,
                    color='red'
                ),
            )
            plotly_fig.add_trace(bbox_vertices)
        # draw lines
        # for vectors in self.bbox_list:
        #     #vectors = bbox.get_box_points()
        #     color_str = 'red'
        #     lines_for_3dbbox = [
        #             go.Scatter3d(x=[vectors[0][0], vectors[0][0]+0.1], 
        #                         y=[vectors[0][1], vectors[0][1]+0.1], 
        #                         z=[vectors[0][2], vectors[0][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             go.Scatter3d(x=[vectors[1][0], vectors[1][0]+0.1], 
        #                         y=[vectors[1][1], vectors[1][1]+0.1], 
        #                         z=[vectors[1][2], vectors[1][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             go.Scatter3d(x=[vectors[2][0], vectors[2][0]+0.1], 
        #                         y=[vectors[2][1], vectors[2][1]+0.1], 
        #                         z=[vectors[2][2], vectors[2][2]+0.1], mode='lines', line=dict(color=color_str)),
        #             go.Scatter3d(x=[vectors[4][0], vectors[4][0]+0.1], 
        #                         y=[vectors[4][1], vectors[4][1]+0.1], 
        #                         z=[vectors[4][2], vectors[4][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             go.Scatter3d(x=[vectors[5][0], vectors[5][0]+0.1], 
        #                         y=[vectors[5][1], vectors[5][1]+0.1], 
        #                         z=[vectors[5][2], vectors[5][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             go.Scatter3d(x=[vectors[6][0], vectors[6][0]+0.1], 
        #                         y=[vectors[6][1], vectors[6][1]+0.1], 
        #                         z=[vectors[6][2], vectors[6][2]+0.1], mode='lines', line=dict(color=color_str)),
        #             # go.Scatter3d(x=[vectors[1][0], vectors[6][0]], 
        #             #             y=[vectors[1][1], vectors[6][1]], 
        #             #             z=[vectors[1][2], vectors[6][2]], mode='lines', line=dict(color=color_str)),
            
        #             # go.Scatter3d(x=[vectors[1][0], vectors[7][0]], 
        #             #             y=[vectors[1][1], vectors[7][1]], 
        #             #             z=[vectors[1][2], vectors[7][2]], mode='lines', line=dict(color=color_str)),
            
        #             # go.Scatter3d(x=[vectors[2][0], vectors[5][0]], 
        #             #             y=[vectors[2][1], vectors[5][1]], 
        #             #             z=[vectors[2][2], vectors[5][2]], mode='lines', line=dict(color=color_str)),
        #             go.Scatter3d(x=[vectors[7][0], vectors[7][0]+0.1], 
        #                         y=[vectors[7][1], vectors[7][1]+0.1], 
        #                         z=[vectors[7][2], vectors[7][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             go.Scatter3d(x=[vectors[3][0], vectors[3][0]+0.1], 
        #                         y=[vectors[3][1], vectors[3][1]+0.1], 
        #                         z=[vectors[3][2], vectors[3][2]+0.1], mode='lines', line=dict(color=color_str)),
            
        #             # go.Scatter3d(x=[vectors[3][0], vectors[6][0]], 
        #             #             y=[vectors[3][1], vectors[6][1]], 
        #             #             z=[vectors[3][2], vectors[6][2]], mode='lines', line=dict(color=color_str))
        #     ]
        #     for line in lines_for_3dbbox:
        #         plotly_fig.add_trace(line)

        
        if coordinate_frame:
            # Length of the axes
            axis_length = 1

            # Create lines for the axes
            lines = [
                go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red')),
                go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines', line=dict(color='green')),
                go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines', line=dict(color='blue'))
            ]

            # Create cones (arrows) for the axes
            cones = [
                go.Cone(x=[axis_length], y=[0], z=[0], u=[axis_length], v=[0], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
                go.Cone(x=[0], y=[axis_length], z=[0], u=[0], v=[axis_length], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
                go.Cone(x=[0], y=[0], z=[axis_length], u=[0], v=[0], w=[axis_length], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False)
            ]

            # Add lines and cones to the figure
            for line in lines:
                plotly_fig.add_trace(line)
            for cone in cones:
                plotly_fig.add_trace(cone)

        # Show the plot
        plotly_fig.show()

if __name__=="__main__":
    BIN = 1
    PCD = 0

    lidar_path = "./notebooks"
    num = "002697"

    if (BIN):
        bin_path = os.path.join(lidar_path, "velodyne")
        numpy_path = np.fromfile(os.path.join(bin_path, num+".bin"), dtype=np.float32).reshape((-1,4))
        #numpy_path = numpy_path[:,:3]
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(numpy_path)
    elif (PCD):
        pcd_path = os.path.join(lidar_path, "pcd", num+".pcd")
        pcd = o3d.io.read_point_cloud(pcd_path)


    #pcd_array = np.asarray(pcd.points)
    annotation_file = 'notebooks/label/002697.txt'
    calib_file = annotation_file.replace('label', 'calib')
    bboxmaker = bboxMaker(annotation_file, calib_file)
    bboxmaker.show_point_cloud(numpy_path)