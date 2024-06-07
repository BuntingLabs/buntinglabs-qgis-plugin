# Copyright 2024 Bunting Labs, Inc.
import heapq
from collections import defaultdict
from functools import lru_cache

from qgis.core import QgsPointXY

class TrajectoryTree:
    def __init__(self, pts_costs, pts_paths, params):
        self.params = params # (dx, dy, x_min, y_max)

        self.graph_nodes = []
        for path in pts_paths.keys():
            ix, iy, jx, jy = map(int, path.split('_'))

            self.graph_nodes.extend([(ix, iy), (jx, jy)])

        self.graph_neighbors = defaultdict(list)
        for path, cost in pts_costs.items():
            ix, iy, jx, jy = map(int, path.split('_'))
            assert (ix, iy) != (jx, jy)

            self.graph_neighbors[(ix, iy)].append(((jx, jy), cost))
            self.graph_neighbors[(jx, jy)].append(((ix, iy), cost))

    def idx_for_closest(self, pt: QgsPointXY):
        # (dx, dy, x_min, y_max)
        img_x, img_y = (pt.x() - self.params[2]) / self.params[0], -(pt.y() - self.params[3]) / self.params[1]

        dists = [(img_x - x) ** 2 + (img_y - y) ** 2 for x, y in self.graph_nodes]
        return dists.index(min(dists))

    @lru_cache(maxsize=100)
    def dijkstra(self, start_idx: int, end_idx: int):
        start, end = self.graph_nodes[start_idx], self.graph_nodes[end_idx]
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
