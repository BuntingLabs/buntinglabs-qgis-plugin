# Copyright 2023 Bunting Labs, Inc.
import time
from enum import Enum
from collections import namedtuple
import cProfile, pstats, io
from typing import List
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
from qgis.core import QgsField
from PyQt5.QtCore import QVariant


from .tracing_task import AutocompleteTask, HoverTask
from .trajectory import Trajectory, TrajectoryClick
from .trajectory_tree import TrajectoryTree

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
        self.trees = dict()

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

    def solvePathToPoint(self, pt: QgsPointXY) -> List[QgsPointXY]:
        if self.map_cache is None or len(self.vertices) == 0:
            return [pt]

        (x_min, dx, y_max, dy) = (self.map_cache.x_min, self.map_cache.dx, self.map_cache.y_max, self.map_cache.dy)
        (_, pts_paths) = self.graphs['penis']

        cur_tree = self.trees[self.map_cache.uniq_id]
        path, cost = cur_tree.dijkstra(
            cur_tree.idx_for_closest(self.vertices[-1]),
            cur_tree.idx_for_closest(pt)
        )
        if len(path) == 0:
            return None
        # Convert path to coordinates
        # path = []

        # Replace bits of the path as possible
        minimized_path = [path[0]]
        # minimized_path = [[path[0][1], path[0][0]]] # Coordinate order gets flipped for some reason
        for i in range(len(path)-1):
            prev, next = path[i], path[i+1]
            # (ix, iy), (jx, jy) = np.unravel_index(prev, (1200, 1200)), np.unravel_index(next, (1200, 1200))

            if f"{prev}_{next}" in pts_paths:
                minimized_path.extend(pts_paths[f"{prev}_{next}"][1:])
            elif f"{next}_{prev}" in pts_paths:
                minimized_path.extend(reversed(pts_paths[f"{next}_{prev}"][:-1]))
            else:
                minimized_path.append(next)

        # convert to coordinates
        minimized_path = [np.unravel_index(node, (600, 600)) for node in minimized_path]
        minimized_path = [[int(node[0]), int(node[1])] for node in minimized_path]

        path_map_pts = [ QgsPointXY(node[1] * dx + x_min, y_max - node[0] * dy) for node in minimized_path ]
        return path_map_pts

    def canvasMoveEvent(self, e):
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
            path_map_pts = self.solvePathToPoint(pt)

            # None = failed to navigate
            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY(path_map_pts if path_map_pts is not None else []),
                None
            )
            return
        else:
            pass

        hover_cache_entry = []

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

            queued_points = self.solvePathToPoint(point)

            for completed_pt in queued_points:
                self.addVertex(completed_pt)
                self.vertices.append(completed_pt)

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            # Analyze the map if we have >=2 vertices
            if len(self.vertices) >= 2 and not (e.modifiers() & Qt.ShiftModifier):
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
        dx, dy, x_min, y_max = self.map_cache.dx, self.map_cache.dy, self.map_cache.x_min, self.map_cache.y_max

        # Create a vector layer for graph nodes
        # node_layer = QgsVectorLayer("Point?crs=EPSG:3857", "Graph Nodes", "memory")
        # node_pr = node_layer.dataProvider()
        
        # # Create a vector layer for graph paths with a cost attribute
        # path_layer = QgsVectorLayer("LineString?crs=EPSG:3857", "Graph Paths", "memory")
        # path_layer.dataProvider().addAttributes([QgsField("cost", QVariant.Double)])
        # path_layer.updateFields()
        # path_pr = path_layer.dataProvider()
        
        # for nodes, path_in_between in pts_paths.items():
        #     ix, iy, jx, jy = map(int, nodes.split('_'))

        #     # Add nodes to the node layer and sindex_points
        #     for x, y in [(ix, iy), (jx, jy)]:
        #         point = QgsPointXY(x * dx + x_min, y_max - y * dy)
        #         feature = QgsFeature()
        #         feature.setGeometry(QgsGeometry.fromPointXY(point))
        #         node_pr.addFeature(feature)
            
        #     # Add path to the path layer with cost attribute
        #     line_points = [(ix * dx + x_min, y_max - iy * dy)] + \
        #                   [(x * dx + x_min, y_max - y * dy) for y, x in path_in_between] + \
        #                   [(jx * dx + x_min, y_max - jy * dy)]
        #     line_string = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in line_points])
        #     length = line_string.length() + 1.0
        #     feature = QgsFeature()
        #     feature.setGeometry(line_string)
        #     feature.setAttributes([pts_cost[nodes] / length])
        #     path_pr.addFeature(feature)
        # node_layer.updateExtents()
        # path_layer.updateExtents()
        # QgsProject.instance().addMapLayers([node_layer, path_layer], True)

        self.trees[self.map_cache.uniq_id] = TrajectoryTree(pts_cost, pts_paths, (dx, dy, x_min, y_max))

        if self.map_cache is not None:
            self.graphs[self.map_cache.uniq_id] = (pts_cost, pts_paths)
            self.graphs['penis'] = (pts_cost, pts_paths)

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