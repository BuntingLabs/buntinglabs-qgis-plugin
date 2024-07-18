# Copyright 2024 Bunting Labs, Inc.

import heapq
from collections import defaultdict
from functools import lru_cache

import numpy as np
from qgis.core import QgsPointXY

class TrajectoryTree:
    def __init__(self, pts_costs, params, img_params, trajectory_root):
        self.params = params # (x_min, y_max, dxdy, y_max)
        self.img_params = img_params # (img_height, img_width)
        self.trajectory_root = trajectory_root

        # Hidden, with a setter
        self.graph_neighbors = defaultdict(list)
        for path, cost in pts_costs.items():
            orig, dest = map(int, path.split('_'))
            if orig == dest:
                continue

            self.graph_neighbors[orig].append((dest, cost))

    @lru_cache(maxsize=1)
    def _graph_nodes_coords(self):
        graph_nodes = set(self.graph_neighbors.keys()).union(dest for dest, _ in sum(self.graph_neighbors.values(), []))
        return [(np.unravel_index(int(node), (self.img_params[0], self.img_params[1])), node) for node in graph_nodes]

    def closest_nodes_to(self, pt: QgsPointXY, n: int):
        x_min, y_min, dxdy, y_max = self.params

        img_x, img_y = (pt.x() - x_min*256*dxdy) / dxdy, (y_max*256*dxdy - pt.y()) / dxdy
        graph_nodes_coords = self._graph_nodes_coords()
        dists = [((img_x - x) ** 2 + (img_y - y) ** 2, node) for ((y, x), node) in graph_nodes_coords]
        # return closest n
        return [node for _, node in sorted(dists)[:n]]

    @lru_cache(maxsize=100)
    def dijkstra(self, end: int):
        start = self.trajectory_root

        queue = [(0, start)]
        distances = {start: 0}
        previous_nodes = {start: None}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_node == end:
                break

            for neighbor, edge_weight in self.graph_neighbors[current_node]:
                new_distance = current_distance + edge_weight
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (new_distance, neighbor))

        # If the end node wasn't reachable from the start
        if end not in previous_nodes:
            return [], float('inf')

        path, current = [], end
        while current is not None:
            path.append(current)
            current = previous_nodes[current]
        path = path[::-1]

        return path, distances.get(end, float('inf'))
