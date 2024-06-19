# Copyright 2024 Bunting Labs, Inc.
import heapq
from collections import defaultdict
from functools import lru_cache

import numpy as np
from qgis.core import QgsPointXY

class TrajectoryTree:
    def __init__(self, pts_costs, pts_paths, params):
        self.params = params # (dx, dy, x_min, y_max)

        # Hidden, with a setter
        self.graph_neighbors = defaultdict(list)
        for path, cost in pts_costs.items():
            orig, dest = map(int, path.split('_'))
            if orig == dest:
                continue

            self.graph_neighbors[orig].append((dest, cost))
            self.graph_neighbors[dest].append((orig, cost))

    @lru_cache(maxsize=1)
    def _graph_nodes_coords(self):
        # I think this union is unnecessary but doesn't hurt
        graph_nodes = list(self.graph_neighbors.keys())
        return [(np.unravel_index(int(node), (600, 600)), node) for node in graph_nodes]

    def closest_node_to(self, pt: QgsPointXY):
        img_x, img_y = (pt.x() - self.params[2]) / self.params[0], -(pt.y() - self.params[3]) / self.params[1]
        graph_nodes_coords = self._graph_nodes_coords()
        dists = [((img_x - x) ** 2 + (img_y - y) ** 2, node) for ((y, x), node) in graph_nodes_coords]
        return min(dists)[1]

    @lru_cache(maxsize=100)
    def dijkstra(self, start: int, end: int):
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
