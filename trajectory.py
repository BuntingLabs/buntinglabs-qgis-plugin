# Copyright 2023 Bunting Labs, Inc.
from qgis.core import QgsSpatialIndex, QgsPointXY, QgsFeature, QgsGeometry, \
    QgsVectorLayer

from typing import Tuple

def serialize_pos(x1, y1):
    return f"{x1}_{y1}"
def deserialize_pos(pos_str):
    return tuple(map(int, pos_str.split('_')))

class TrajectoryClick:
    def __init__(self):
        self.continuations = dict()

class Trajectory:
    def __init__(self):
        self.pos = TrajectoryClick()

    def to_dict(self):
        serialized = dict()
        for serialized_pos, continuation in self.pos.continuations.items():
            serialized[serialized_pos] = continuation.to_dict()
        return serialized

    @staticmethod
    def from_dict(node_dict):
        traj = Trajectory()
        for serialized_pos, continuation_dict in node_dict.items():
            traj.pos.continuations[serialized_pos] = Trajectory.from_dict(continuation_dict)
        return traj

    def merge(self, other):
        def merge_continuations(cont1, cont2):
            for key, value in cont2.items():
                if key in cont1:
                    merge_continuations(cont1[key].pos.continuations, value.pos.continuations)
                else:
                    cont1[key] = value

        merge_continuations(self.pos.continuations, other.pos.continuations)

    # Searches the trajectory for the completion nearest (x1, y1) and returns the path.
    def search(self, pos: Tuple[int, int]):
        def euclidean_distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        min_distance = float('inf')
        closest_node = None
        parent_map = {}

        stack = [(self.pos.continuations, '')]
        while stack:
            current_continuations, current_serialized_pos = stack.pop()
            for k, v in current_continuations.items():
                deserialized_pos = deserialize_pos(k)
                distance = euclidean_distance(pos, deserialized_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_node = k
                parent_map[k] = current_serialized_pos
                stack.append((v.pos.continuations, k))

        path = []
        while closest_node:
            path.append(deserialize_pos(closest_node))
            closest_node = parent_map.get(closest_node)

        return path[::-1]

    def to_spatial_index(self):
        index = QgsSpatialIndex()
        stack = [(self.pos.continuations, '')]
        parent_map = {}
        points = []

        vector_layer = QgsVectorLayer("Point?crs=EPSG:3857", "points_layer", "memory")
        provider = vector_layer.dataProvider()

        i = 0
        while stack:
            current_continuations, current_serialized_pos = stack.pop()
            for k, v in current_continuations.items():
                x, y = deserialize_pos(k)
                point = QgsPointXY(x, y)
                feature = QgsFeature(i)
                feature.setGeometry(QgsGeometry.fromPointXY(point))
                index.addFeature(feature)
                points.append(point)
                parent_map[i] = current_serialized_pos
                provider.addFeature(feature)
                i += 1
                stack.append((v.pos.continuations, k))

        return (index, points, parent_map, vector_layer)

    @staticmethod
    def path_from_i(i, parent_map):
        path = []
        while i:
            if '_' not in i:
                print('i', i)
                break
            path.append(deserialize_pos(i))
            i = parent_map.get(i)
        return path[::-1]