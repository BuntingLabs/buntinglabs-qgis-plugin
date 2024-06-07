# Copyright 2023 Bunting Labs, Inc.
import time
from enum import Enum
from collections import namedtuple
import cProfile, pstats, io
import heapq

from qgis.PyQt.QtCore import Qt, QSettings, QUrl
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtGui import QColor, QDesktopServices
from qgis.gui import QgsMapToolCapture, QgsRubberBand, QgsVertexMarker, \
    QgsSnapIndicator
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsPoint, QgsWkbTypes, QgsLayerTreeLayer, QgsSpatialIndex
from PyQt5.QtCore import pyqtSignal
import numpy as np

from .tracing_task import AutocompleteTask, HoverTask
from .trajectory import Trajectory, TrajectoryClick

def get_complement(color):
    r, g, b, a = color.red(), color.green(), color.blue(), color.alpha()
    # Calculate the complement
    comp_r = 255 - r
    comp_g = 255 - g
    comp_b = 255 - b
    # Return the complement color
    return QColor(comp_r, comp_g, comp_b, a)

# line_segment_idx is zero-indexed to the start coordinate
def find_closest_projection_point(pts, pt):
    min_distance = float('inf')
    projected_pt = None
    line_segment_index = None
    for i in range(len(pts) - 1):
        start, end = pts[i], pts[i+1]
        segment = QgsGeometry.fromPolylineXY([start, end])
        projected_point = segment.nearestPoint(QgsGeometry.fromPointXY(pt))
        distance = pt.distance(projected_point.asPoint())
        if distance < min_distance:
            min_distance = distance
            projected_pt = projected_point.asPoint()
            line_segment_index = i
    return QgsPointXY(projected_pt), line_segment_index

# DFS to find all visible raster layers, even those in groups
def find_raster_layers(node):
    layers = []
    for child in node.children():
        if isinstance(child, QgsLayerTreeLayer) and isinstance(child.layer(), QgsRasterLayer) and child.itemVisibilityChecked():
            layers.append(child.layer())
        elif child.children():
            layers.extend(find_raster_layers(child))
    return layers

class ShiftClickState(Enum):
    HAS_NOT_CUT = 1
    HAS_CUT = 2

from collections import OrderedDict

# (n_px, n_py) are "normalized" positions
AutocompleteCacheEntry = namedtuple('AutocompleteCacheEntry', ['uniq_id', 'n_px', 'n_py'])

class AutocompleteCache:
    def __init__(self, max_size, round_px=1.0):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.round_px = round_px

    def get(self, uniq_id: str, px: float, py: float):
        key = AutocompleteCacheEntry(uniq_id, int(px / self.round_px), int(py / self.round_px))

        # Cache hit, use it.
        if key in self.cache:
            # Move the key to the end to show that it was recently used
            self.cache.move_to_end(key)
            return self.cache[key]

        # Cache miss
        return None

    def set(self, uniq_id: str, px: float, py: float, value):
        key = AutocompleteCacheEntry(uniq_id, int(px / self.round_px), int(py / self.round_px))

        if key in self.cache:
            # Move the key to the end to show that it was recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Remove the first item (least recently used)
            self.cache.popitem(last=False)


MapCacheEntry = namedtuple('MapCacheEntry', ['uniq_id', 'dx', 'dy', 'x_min', 'y_max', 'window_size'])

# QgsMapToolCapture is a subclass of QgsMapToolEdit that provides
# additional functionality for map tools that capture geometry. It
# is an abstract base class for map tools that capture line and
# polygon geometries. It handles the drawing of rubber bands on the
# map canvas and the capturing of clicks to build the geometry.
class AIVectorizerTool(QgsMapToolCapture):

    # predictedPointsReceived = pyqtSignal(tuple)

    def __init__(self, plugin):
        # Extend QgsMapToolCapture
        cadDockWidget = plugin.iface.cadDockWidget()
        super(AIVectorizerTool, self).__init__(plugin.iface.mapCanvas(), cadDockWidget, QgsMapToolCapture.CaptureNone)

        self.plugin = plugin
        self.rb = self.initRubberBand()

        # Options
        self.num_completions = 100

        # List of QgsPointXY that represents the new feature
        # via QgsMapTool.toMapCoordinates(), it's in project CRS
        self.vertices = []

        self.autocomplete_task = None
        self.hover_task_is_active = False

        # And take control and go full on editing mode
        self.activate()
        # Capturing mode determines whether or not the rubber band
        # will follow the moving cursor, once there's a vertex in the chamber
        self.startCapturing()

        # self.scissors_icon = QgsVertexMarker(plugin.iface.mapCanvas())
        # self.scissors_icon.setIconType(QgsVertexMarker.ICON_X)
        # self.scissors_icon.setColor(get_complement(self.digitizingStrokeColor()))
        # self.scissors_icon.setIconSize(18)
        # self.scissors_icon.setPenWidth(5)
        # self.scissors_icon.setZValue(1000)

        # For snapping
        self.snapIndicator = QgsSnapIndicator(plugin.iface.mapCanvas())
        self.snapper = plugin.iface.mapCanvas().snappingUtils()

        self.streamingToleranceInPixels = int(QSettings().value('qgis/digitizing/stream_tolerance', 2))

        self.predicted_points = []

        self.dx = None
        self.dy = None
        self.x_min = None
        self.y_max = None

        # listen to event
        # self.predictedPointsReceived.connect(lambda pts: )

        self.map_cache = None # MapCacheEntry
        self.autocomplete_cache = AutocompleteCache(250, round_px=3.0)

        self.traj_tries = dict()
        self.spatial_indices = dict()
        self.graphs = dict()

    # This will only be called in QGIS is older than 3.32, hopefully.
    def supportsTechnique(self, technique):
        # we do not support shape or circular
        return (technique in [
            Qgis.CaptureTechnique.StraightSegments,
            Qgis.CaptureTechnique.Streaming
        ])

    # Wrap currentCaptureTechnique() because it was only added in 3.32.
    # def isStreamingCapture(self):
    #     if hasattr(self, 'currentCaptureTechnique') and hasattr(Qgis, 'CaptureTechnique'):
    #         if hasattr(Qgis.CaptureTechnique, 'Streaming'):
    #             return self.currentCaptureTechnique() == Qgis.CaptureTechnique.Streaming
    #     return False

    def initRubberBand(self):
        if self.mode() == QgsMapToolCapture.CaptureLine:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
        elif self.mode() == QgsMapToolCapture.CapturePolygon:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        else:
            # TODO not sure when we get here.
            # But it shouldn't matter because rb.setToGeometry "also
            # change[s] the geometry type of the rubberband."
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.LineGeometry)

        rb.setFillColor(self.digitizingFillColor())
        rb.setStrokeColor(self.digitizingStrokeColor())
        rb.setWidth(self.digitizingStrokeWidth())
        rb.setLineStyle(Qt.DotLine)

        return rb

    # msg_type is Qgis.Critical, Qgis.Info, Qgis.Warning, Qgis.success
    def notifyUserOfMessage(self, msg, msg_type, link_url, link_text):
        widget = self.plugin.iface.messageBar().createMessage("AI Vectorizer", msg)
        button = QPushButton(widget)

        if link_url is not None and link_text is not None:
            button.setText(link_text)
            button.pressed.connect(lambda: QDesktopServices.openUrl(QUrl(link_url)))
        else:
            button.setText("Open Settings")
            button.pressed.connect(self.plugin.openSettings)

        widget.layout().addWidget(button)
        self.plugin.iface.messageBar().pushWidget(widget, msg_type, duration=15)

    def handlePointReceived(self, args):
        self.addVertex(QgsPointXY(*args[0]))
        self.vertices.append(QgsPointXY(*args[0]))

    def trimVerticesToPoint(self, vertices, pt):
        assert len(vertices) >= 2

        last_point, last_point_idx = find_closest_projection_point(vertices, pt)
        points = vertices[:last_point_idx+1] + [last_point]

        return points

    def canvasMoveEvent(self, e):
        start_time = time.time()

        if self.isAutoSnapEnabled():
            snapMatch = self.snapper.snapToMap(e.pos())
            self.snapIndicator.setMatch(snapMatch)

        if len(self.vertices) == 0:
            # Nothing to do!
            return

        if self.snapIndicator.match().type():
            pt = self.snapIndicator.match().point()
        else:
            pt = self.toMapCoordinates(e.pos())

        # Relative to map_cache
        if self.map_cache is not None and self.map_cache.uniq_id in self.graphs:
            pr = cProfile.Profile()
            pr.enable()

            (x_min, dx, y_max, dy) = (self.map_cache.x_min, self.map_cache.dx, self.map_cache.y_max, self.map_cache.dy)
            hover_px, hover_py = (pt.x() - x_min) / dx, -(pt.y() - y_max) / dy

            # print('hover', self.graphs[self.map_cache.uniq_id])
            (pts_costs, pts_paths, sindex_points) = self.graphs['penis']#self.map_cache.uniq_id]

            curr_pt = np.array([[pt.x(), pt.y()]]) # [1, 2]
            dists = np.linalg.norm(sindex_points - curr_pt, axis=1) # [N,]

            nearest_node_id = np.argmin(dists)
            nearest_node_pt = QgsPointXY(sindex_points[nearest_node_id][0], sindex_points[nearest_node_id][1])

            # print('nearest', nearest_node_pt.x(), nearest_node_pt.y(), 'pt', pt.x(), pt.y())

            # create list of nodes
            graph_nodes = []
            for path in pts_paths.keys():
                ix, iy, jx, jy = map(int, path.split('_'))

                graph_nodes.extend([(ix, iy), (jx, jy)])

            def idx_for_closest(pt: QgsPointXY):
                img_x, img_y = (pt.x() - x_min) / dx, -(pt.y() - y_max) / dy

                dists = [(img_x - x) ** 2 + (img_y - y) ** 2 for x, y in graph_nodes]
                return dists.index(min(dists))
            # print('graph_nodes', graph_nodes)

            def dijkstra(graph_nodes, pts_costs, start_idx, end_idx):
                start, end = graph_nodes[start_idx], graph_nodes[end_idx]
                queue = [(0, start)]
                distances = {start: 0}
                previous_nodes = {start: None}

                while queue:
                    current_distance, current_node = heapq.heappop(queue)

                    if current_node == end:
                        break

                    ix, iy = current_node
                    for neighbor in graph_nodes:
                        if neighbor == current_node:
                            continue
                        jx, jy = neighbor
                        edge_key1 = f"{ix}_{iy}_{jx}_{jy}"
                        edge_key2 = f"{jx}_{jy}_{ix}_{iy}"
                        edge_weight = min(pts_costs.get(edge_key1, float('inf')), pts_costs.get(edge_key2, float('inf')))

                        if edge_weight == float('inf'):
                            continue

                        new_distance = current_distance + edge_weight
                        if new_distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = new_distance
                            previous_nodes[neighbor] = current_node
                            heapq.heappush(queue, (new_distance, neighbor))

                path, current = [], end
                while current is not None:
                    path.append(current)
                    current = previous_nodes[current]
                path = path[::-1]

                return path, distances.get(end, float('inf'))

            path, cost = dijkstra(graph_nodes, pts_costs, idx_for_closest(self.vertices[-1]), idx_for_closest(pt))

            # [(251, 262), (251, 255), (288, 84), (220, 49)]
            # Replace bits of the path as possible
            minimized_path = [path[0]]
            for i in range(len(path)-1):
                prev, next = path[i], path[i+1]
                (ix, iy), (jx, jy) = (prev, next)

                if f"{ix}_{iy}_{jx}_{jy}" in pts_paths:
                    minimized_path.extend(pts_paths[f"{ix}_{iy}_{jx}_{jy}"][1:])
                elif f"{jx}_{jy}_{ix}_{iy}" in pts_paths:
                    minimized_path.extend(reversed(pts_paths[f"{jx}_{jy}_{ix}_{iy}"][:-1]))
                else:
                    minimized_path.append(next)

            path_map_pts = [ QgsPointXY(node[1] * dx + x_min, y_max - node[0] * dy) for node in minimized_path ]

            # closest_node = min(graph_nodes, key=lambda node: (node[0] - hover_px) ** 2 + (node[1] - hover_py) ** 2)
            # # print('closest_node', closest_node)

            # closest_map_x, closest_map_y = closest_node[0] * dx + x_min, y_max - closest_node[1] * dy
            # print('closest_map_x', closest_map_x, 'closest_map_y', closest_map_y)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
            print(s.getvalue())

            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY(path_map_pts),
                None
            )
            
            return
            # First, see if we have a cached route at this point.
            # nearest_id = self.spatial_indices[self.map_cache.uniq_id][0].nearestNeighbor(QgsPointXY(hover_px, hover_py), 1)[0]
            # nearest_point = self.spatial_indices[self.map_cache.uniq_id][1][nearest_id]

            # # print('nearest_point', nearest_point)
            # # get nearest_point as serialized

            # from .trajectory import serialize_pos
            # path_to_nearest_point = self.traj_tries[self.map_cache.uniq_id].path_from_i(
            #     serialize_pos(int(nearest_point.x()), int(nearest_point.y())),
            #     self.spatial_indices[self.map_cache.uniq_id][2]
            # )

            # print('nearest_point', nearest_point)
            # print('nearest_id', nearest_id)
            # print('path_to_nearest_point', path_to_nearest_point)
        else:
            pass
            # print('map cache', self.map_cache)
            # print('self.spatial_indices', self.spatial_indices)

        # print('canvasMoveEvent took ', time.time() - start_time)


        hover_cache_entry = []
        # Check if the shift key is being pressed
        # We have special existing-line-editing mode when shift is hit
        if len(self.vertices) >= 2 and self.map_cache is not None and not (e.modifiers() & Qt.ShiftModifier):

            # print('creating HoverTask')
            # -(y2 - self.y_max) / self.dy, (x2 - self.x_min) / self.dx
            pxys = [self.vertices[-1], (pt.x(), pt.y())]
            (x_min, dx, y_max, dy) = (self.map_cache.x_min, self.map_cache.dx, self.map_cache.y_max, self.map_cache.dy)
            pxys = [((px - x_min) / dx, -(py - y_max) / dy) for (px, py) in pxys]

            # Check cache
            hover_px, hover_py = pxys[-1][0], pxys[-1][1]
            hover_cache_entry = self.autocomplete_cache.get(self.map_cache.uniq_id, hover_px, hover_py)
            if hover_cache_entry is not None:
                pass
                # print('cache HIT', hover_cache_entry)
                # self.predicted_points = [QgsPointXY(*pt) for pt in hover_cache_entry]
                # self.predictedPointsReceived.emit((None,))
            else:
                # TODO behavior when we can't immediately load something?
                hover_cache_entry = []

                # print('cache MISS', self.hover_task_is_active, 'hover', hover_px, hover_py)
                if not self.hover_task_is_active:
                    ht = HoverTask(self, self.map_cache, pxys)

                    ht.messageReceived.connect(lambda e: print('error', e))#self.notifyUserOfMessage(*e))
                    def handleTrajectoriesReceived(self, trajectories):
                        other = Trajectory.from_dict(trajectories)

                        start_time = time.time()
                        self.traj_tries[self.map_cache.uniq_id].merge(other)
                        print(f"Merge took {time.time() - start_time:.4f} seconds")

                        start_time = time.time()
                        self.spatial_indices[self.map_cache.uniq_id] = self.traj_tries[self.map_cache.uniq_id].to_spatial_index()
                        print(f"Spatial index creation took {time.time() - start_time:.4f} seconds")

                        # QgsProject.instance().addMapLayer(self.spatial_indices[self.map_cache.uniq_id][-1])
                        # for trajectory in trajectories:
                        #     transformed_points = [((jx * dx) + x_min, y_max - (ix * dy)) for (ix, jx) in trajectory['trajectory']]

                        #     trajectory_py, trajectory_px = trajectory['simulation_endpoint']
                        #     print('hover', hover_px, hover_py, 'trajectory', trajectory_px, trajectory_py)
                            # self.autocomplete_cache.set(self.map_cache.uniq_id, trajectory_px, trajectory_py, transformed_points)

                        # self.predicted_points = [QgsPointXY(*pt) for pt in transformed_points]
                        # self.predictedPointsReceived.emit((None,))

                    # Replace the lambda with the method call
                    ht.trajectoriesReceived.connect(lambda pts: handleTrajectoriesReceived(self, pts))
                    ht.taskCompleted.connect(lambda: self.dropHoverTask())
                    ht.taskTerminated.connect(lambda: self.dropHoverTask())

                    QgsApplication.taskManager().addTask(
                        ht,
                    )
                    self.hover_task_is_active = True

            last_point = self.vertices[-1]
            # if self.isCutting(last_point):
            #     # Shift means our last vertex should effectively be the closest point to the line
            #     self.scissors_icon.setCenter(last_point)
            #     self.scissors_icon.show()

            #     # Hide the rubber band
            #     self.rb.reset()

            #     return
            # else:
            #     # We've already cut, so now we're drawing lines without autocomplete.
            #     last_point = self.vertices[-1]
            #     self.scissors_icon.hide()

            #     # Use complement color
            #     self.rb.setFillColor(get_complement(self.digitizingFillColor()))
            #     self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))
        else:
            # Either they're holding down shift, or we don't have the map cache for this line.
            # self.predicted_points = []
            last_point = self.vertices[-1]

            # self.scissors_icon.hide()

            # Close it!
            points = [ self.vertices[0], last_point, QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

            self.rb.setFillColor(get_complement(self.digitizingFillColor()))
            self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine and not (e.modifiers() & Qt.ShiftModifier):
            self.updateRubberBand(last_point, hover_cache_entry, [pt])
        elif self.mode() == QgsMapToolCapture.CapturePolygon and not (e.modifiers() & Qt.ShiftModifier):
            self.rb.setToGeometry(
                poly_geo,
                None
            )

    def updateRubberBand(self, last_point, trajectory_points, appended_points=[]):
        closest_predicted_points = [QgsPointXY(*pt) for pt in trajectory_points]

        start_time = time.time()

        if len(appended_points) > 0 and self.map_cache is not None:
            mouse_pt = appended_points[-1]
            # print('mouse_pt', mouse_pt)

            (x_min, dx, y_max, dy) = (self.map_cache.x_min, self.map_cache.dx, self.map_cache.y_max, self.map_cache.dy)
            px, py = mouse_pt.x(), mouse_pt.y()
            hover_px, hover_py = ((px - x_min) / dx, -(py - y_max) / dy)

            closest = self.traj_tries[self.map_cache.uniq_id].search((hover_py, hover_px))
            # print("closest", closest)

            if closest is not None:
                # convert to raster coordinates
                raster_coordinates = []
                for (iy, jx) in closest:
                    xn = (jx * dx) + x_min
                    yn = y_max - (iy * dy)
                    raster_coordinates.append((xn, yn))

            closest_predicted_points = [QgsPointXY(*pt) for pt in raster_coordinates]
            appended_points = []

        end_time = time.time()
        # print(f"Hover, search time: {end_time - start_time} seconds")

        if self.map_cache is not None:
            if self.map_cache.uniq_id not in self.spatial_indices:
                pass
                # print('spatial indices', self.spatial_indices)
            else:
                nearest_id = self.spatial_indices[self.map_cache.uniq_id][0].nearestNeighbor(QgsPointXY(hover_px, hover_py), 1)[0]
                # print('nearest_id', nearest_id)
                nearest_point = self.spatial_indices[self.map_cache.uniq_id][1][nearest_id]
                # print('nearest_point', nearest_point)
                # nearest_feature = self.spatial_indices[self.map_cache.uniq_id].feature(nearest_id)
                # print(nearest_feature.geometry().asPoint())

        if len(closest_predicted_points) > 0:
            # trim predicted points
            if len(appended_points) > 0 and len(closest_predicted_points) >= 2:
                closest_predicted_points = self.trimVerticesToPoint(closest_predicted_points, appended_points[0])

            points = [last_point] + closest_predicted_points + appended_points

            self.rb.setFillColor(get_complement(self.digitizingFillColor()))
            self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))
        else:
            points = [last_point] + appended_points

        self.rb.setToGeometry(
            QgsGeometry.fromPolylineXY(points),
            None
        )

    def canvasPressEvent(self, e):
        pass

    def dropHoverTask(self):
        self.hover_task_is_active = False

    def canvasReleaseEvent(self, e):
        # Either click will cancel an ongoing autocomplete
        self.maybeCancelTask()

        vlayer = self.plugin.iface.activeLayer()
        if not isinstance(vlayer, QgsVectorLayer):
            self.plugin.iface.messageBar().pushMessage(
                "Bunting Labs AI Vectorizer",
                "No active vector layer.",
                Qgis.Warning,
                duration=15)
            return
        elif vlayer.wkbType() not in [QgsWkbTypes.LineString, QgsWkbTypes.MultiLineString,
                                    QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon]:
            self.plugin.iface.messageBar().pushMessage(
                "Bunting Labs AI Vectorizer",
                "Unsupported vector layer type for AI autocomplete.",
                Qgis.Warning,
                duration=15)
            return

        if e.button() == Qt.RightButton:
            # Will be converted to the relevant geometry
            curve = self.captureCurve()

            # Null fields
            f = QgsFeature(vlayer.fields(), 0)

            if self.mode() == QgsMapToolCapture.CaptureLine:
                g = QgsGeometry(curve.curveToLine())
                f.setGeometry(g)
            elif self.mode() == QgsMapToolCapture.CapturePolygon:
                poly = QgsPolygon()
                poly.setExteriorRing(curve.curveToLine())

                f.setGeometry(QgsGeometry(poly))
            else:
                raise ValueError

            vlayer.addFeature(f)
            # Don't let vertices cross over
            self.vertices = []

            self.stopCapturing()
            self.rb.reset()
        # elif e.button() == Qt.LeftButton and self.isStreamingCapture():
        #     # Forces adding a vertex manually
        #     if self.snapIndicator.match().type():
        #         point = self.snapIndicator.match().point()
        #     else:
        #         point = self.toMapCoordinates(e.pos())

        #     self.addVertex(point)
        #     self.vertices.append(point)

        #     self.startCapturing()
        elif e.button() == Qt.LeftButton and e.modifiers() & Qt.ShiftModifier:# and len(self.predicted_points) > 0:
            # Add points including the predicted_points
            # for pt in self.predicted_points:
            #     self.addVertex(pt)
            #     self.vertices.append(pt)

            # self.predicted_points = []
            pass
        elif e.button() == Qt.LeftButton:
            # QgsPointXY with map CRS
            if self.snapIndicator.match().type():
                point = self.snapIndicator.match().point()
            else:
                point = self.toMapCoordinates(e.pos())

            wasDoubleClick = len(self.vertices) >= 1 and point.distance(self.vertices[-1]) == 0

            # self.predicted_points = []

            self.addVertex(point)
            self.vertices.append(point)

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            # Analyze the map if we have >=2 vertices
            if len(self.vertices) >= 2 and not (e.modifiers() & Qt.ShiftModifier) and not wasDoubleClick:
                root = QgsProject.instance().layerTreeRoot()
                rlayers = find_raster_layers(root)
                project_crs = QgsProject.instance().crs()

                self.autocomplete_task = AutocompleteTask(
                    self,
                    vlayer,
                    rlayers,
                    project_crs
                )

                # self.autocomplete_task.pointReceived.connect(lambda args: self.handlePointReceived(args))
                self.autocomplete_task.messageReceived.connect(lambda e: self.notifyUserOfMessage(*e))
                self.autocomplete_task.cacheEntryCreated.connect(lambda args: self.handleCacheEntryCreated(*args))

                self.autocomplete_task.graphConstructed.connect(lambda args: self.handleGraphConstructed(*args))

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )

    def handleGraphConstructed(self, pts_cost, pts_paths):
        graph_nodes = list(pts_paths.keys())
        # graph_vector_points = []
        dx, dy, x_min, y_max = self.map_cache.dx, self.map_cache.dy, self.map_cache.x_min, self.map_cache.y_max
        # for node in graph_nodes:
        #     jx, iy = map(int, node.split('_'))
        #     xn = (jx * dx) + x_min
        #     yn = y_max - (iy * dy)
        #     graph_vector_points.append((xn, yn))
        # Create a vector layer for graph nodes
        node_layer = QgsVectorLayer("Point?crs=EPSG:3857", "Graph Nodes", "memory")
        node_pr = node_layer.dataProvider()
        
        # Create a vector layer for graph paths
        path_layer = QgsVectorLayer("LineString?crs=EPSG:3857", "Graph Paths", "memory")
        path_pr = path_layer.dataProvider()
        
        import numpy as np
        sindex_points = []
        for nodes, path_in_between in pts_paths.items():
            ix, iy, jx, jy = map(int, nodes.split('_'))

            # Add nodes to the node layer and sindex_points
            for x, y in [(ix, iy), (jx, jy)]:
                point = QgsPointXY(x * dx + x_min, y_max - y * dy)
                feature = QgsFeature()
                feature.setGeometry(QgsGeometry.fromPointXY(point))
                node_pr.addFeature(feature)

                sindex_points.append([point.x(), point.y()])
            
            # Add path to the path layer
            line_points = [(ix * dx + x_min, y_max - iy * dy)] + \
                          [(x * dx + x_min, y_max - y * dy) for y, x in path_in_between] + \
                          [(jx * dx + x_min, y_max - jy * dy)]
            line_string = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in line_points])
            feature = QgsFeature()
            feature.setGeometry(line_string)
            path_pr.addFeature(feature)
        node_layer.updateExtents()
        path_layer.updateExtents()
        QgsProject.instance().addMapLayers([node_layer, path_layer], True)

        #     coord_ix = (ix * dx) + x_min
        #     coord_iy = y_max - (iy * dy)
        #     coord_jx = (jx * dx) + x_min
        #     coord_jy = y_max - (jy * dy)
        #     print(f'nodes is {ix}_{iy}_{jx}_{jy}')
        #     print('nodes', nodes)
        #     print('path_in_between', path_in_between)
        # import numpy as np
        # pts_cost_np = np.array(pts_cost)
        # print(pts_cost_np)

        # layer = QgsVectorLayer("Point?crs=EPSG:3857", "Graph Points", "memory")
        # pr = layer.dataProvider()
        # for x, y in graph_vector_points:
        #     pt = QgsPointXY(x, y)
        #     feature = QgsFeature()
        #     feature.setGeometry(QgsGeometry.fromPointXY(pt))
        #     pr.addFeature(feature)
        # layer.updateExtents()
        # QgsProject.instance().addMapLayer(layer)

        # print('pts_cost', pts_cost)
        # print('pts_paths', pts_paths)
        sindex_points = np.array(sindex_points)
        if self.map_cache is not None:
            self.graphs[self.map_cache.uniq_id] = (pts_cost, pts_paths, sindex_points)
            self.graphs['penis'] = (pts_cost, pts_paths, sindex_points)

    def handleCacheEntryCreated(self, uniq_id, dx, dy, x_min, y_max, window_size):
        self.map_cache = MapCacheEntry(uniq_id, dx, dy, x_min, y_max, window_size)
        self.traj_tries[uniq_id] = Trajectory()

    def maybeCancelTask(self):
        # Cancels the task if it's running
        if self.autocomplete_task is not None:
            # QgsTasks passed to a task manager end up being owned
            # in C++ land which leads us ... here.
            try:
                self.autocomplete_task.cancel()
            except RuntimeError:
                pass
            finally:
                self.autocomplete_task = None

            return True
        else:
            return False

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Backspace, Qt.Key_Delete) and len(self.vertices) >= 2:
            self.maybeCancelTask()

            if not e.isAutoRepeat():
                self.undo()
                self.vertices.pop()

                e.accept()
                return
        elif e.key() == Qt.Key_Escape:
            if self.maybeCancelTask():
                # escape will just cancel the task if it existed
                return

            self.stopCapturing()
            self.vertices = []
            self.rb.reset()
            self.map_cache = None

            e.accept()
            return

        e.ignore()

    def deactivate(self):
        self.rb.reset()

        # self.scissors_icon.hide()
        self.plugin.action.setChecked(False)