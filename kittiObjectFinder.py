import numpy as np

class KittiObjectFinder:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file

    def _parse_line(self, line):
        parts = line.strip().split()
        if len(parts) == 15:
            obj_type, _, _, _, _, _, _, _, height, width, length, x, y, z, _ = parts
            return {
                "type": obj_type,
                "center": [float(x), float(y), float(z)],
                "dimensions": [float(height), float(width), float(length)]
            }
        return None

    def find_objects_with_labels(self):
        objects_with_labels = []
        with open(self.annotation_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                obj = self._parse_line(line)
                if obj:
                    objects_with_labels.append(obj)
        return objects_with_labels

    def find_objects_containing_points(self, points, pointarray):
        selected_objects = []
        with open(self.annotation_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                obj = self._parse_line(line)
                if obj:
                    for index in points:
                        if self._is_point_in_box(np.asarray(pointarray)[index], obj['center'], obj['dimensions']):
                            selected_objects.append(obj)
                            break  # Break if any of the points is inside this object
        return selected_objects

    def _is_point_in_box(self, point, box_center, box_dimensions):
        # Check if a point is inside a 3D box
        for i in range(3):
            if point[i] < box_center[i] - box_dimensions[i] / 2 or point[i] > box_center[i] + box_dimensions[i] / 2:
                return False
        return True