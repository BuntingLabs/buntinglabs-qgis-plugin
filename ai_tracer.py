# Copyright 2024 Bunting Labs, Inc.

import uuid
from enum import Enum
from collections import namedtuple
from typing import List
import time
from functools import reduce, lru_cache

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtWidgets import QPushButton, QProgressBar, QLabel
from qgis.PyQt.QtGui import QColor, QDesktopServices
from qgis.gui import QgsMapToolCapture, QgsRubberBand, QgsSnapIndicator
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsWkbTypes, QgsLayerTreeLayer, QgsRectangle
from qgis.core import QgsField
from PyQt5.QtCore import QVariant
import numpy as np

from .tracing_task import UploadChunkAndSolveTask, \
    rasterUnitsPerPixelEstimate, layerDoesIntersect
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

class Chunk:
    DEFAULT_CHUNK_SIZE = 256

    def __init__(self, x, y, dxdy, chunk_size=DEFAULT_CHUNK_SIZE):
        self.x = x
        self.y = y
        self.dxdy = dxdy
        self.chunk_size = chunk_size

    def __str__(self):
        if self.chunk_size == self.DEFAULT_CHUNK_SIZE:
            return f"Chunk({self.x},{self.y},{self.dxdy})"
        else:
            return f"Chunk({self.x},{self.y},{self.dxdy},{self.chunk_size})"

    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return NotImplemented
        return (self.x, self.y, self.dxdy, self.chunk_size) == (other.x, other.y, other.dxdy, other.chunk_size)

    def __hash__(self):
        return hash((self.x, self.y, self.dxdy, self.chunk_size))

    @staticmethod
    def strToChunk(chunk_str: str):
        parts = chunk_str[6:-1].split(',')
        x, y, dxdy = map(float, parts[:3])
        chunk_size = int(parts[3]) if len(parts) > 3 else Chunk.DEFAULT_CHUNK_SIZE
        return Chunk(int(x), int(y), dxdy, chunk_size)

    @staticmethod
    def pointToChunk(pt: QgsPointXY, dxdy: float, chunk_size=DEFAULT_CHUNK_SIZE):
        ix, iy = (pt.x() / dxdy, pt.y() / dxdy)
        return Chunk(int(ix // chunk_size), int(iy // chunk_size), dxdy, chunk_size)

    def toPolygon(self) -> QgsGeometry:
        x_min = self.dxdy * self.x * self.chunk_size
        x_max = self.dxdy * (self.x + 1) * self.chunk_size
        y_min = self.dxdy * self.y * self.chunk_size
        y_max = self.dxdy * (self.y + 1) * self.chunk_size

        points = [
            QgsPointXY(x_min, y_min),
            QgsPointXY(x_max, y_min),
            QgsPointXY(x_max, y_max),
            QgsPointXY(x_min, y_max),
            QgsPointXY(x_min, y_min)
        ]

        return QgsGeometry.fromPolygonXY([points])

    def toRectangle(self) -> QgsRectangle:
        x_min = self.dxdy * self.x * self.chunk_size
        x_max = self.dxdy * (self.x + 1) * self.chunk_size
        y_min = self.dxdy * self.y * self.chunk_size
        y_max = self.dxdy * (self.y + 1) * self.chunk_size
        return QgsRectangle(x_min, y_min, x_max, y_max)

    def distanceToPoint(self, pt: QgsPointXY) -> float:
        return self.toRectangle().distance(pt)

    @staticmethod
    def rectangleToChunks(extent: QgsRectangle, dxdy: float, chunk_size=DEFAULT_CHUNK_SIZE) -> list:
        x_min, y_min, x_max, y_max = extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()

        start_chunk = Chunk.pointToChunk(QgsPointXY(x_min, y_min), dxdy, chunk_size)
        end_chunk = Chunk.pointToChunk(QgsPointXY(x_max, y_max), dxdy, chunk_size)

        chunks = []
        for x in range(start_chunk.x, end_chunk.x + 1):
            for y in range(start_chunk.y, end_chunk.y + 1):
                chunks.append(Chunk(x, y, dxdy, chunk_size))

        return chunks

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


# QgsMapToolCapture is a subclass of QgsMapToolEdit that provides
# additional functionality for map tools that capture geometry. It
# is an abstract base class for map tools that capture line and
# polygon geometries. It handles the drawing of rubber bands on the
# map canvas and the capturing of clicks to build the geometry.
class AIVectorizerTool(QgsMapToolCapture):

    def __init__(self, plugin):
        # Extend QgsMapToolCapture
        cadDockWidget = plugin.iface.cadDockWidget()
        super(AIVectorizerTool, self).__init__(plugin.iface.mapCanvas(), cadDockWidget, QgsMapToolCapture.CaptureNone)

        self.plugin = plugin
        self.rb = self.initRubberBand()

        self.chunk_rb = QgsRubberBand(plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        self.chunk_cache = dict() # True=uploaded, False=uploading
        self.fly_instance_id = None

        # FOG OF WAR
        self.fow_rb = QgsRubberBand(plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        self.fow_rb.setFillColor(QColor(0, 0, 0, 0))  # transparent
        self.fow_rb.setStrokeColor(QColor(0, 255, 0, 128))  # 50% transparent lime green
        self.fow_rb.setWidth(6)
        self.fow_rb.setLineStyle(Qt.SolidLine)

        # List of QgsPointXY that represents the new feature
        # via QgsMapTool.toMapCoordinates(), it's in project CRS
        self.vertices = []

        # And take control and go full on editing mode
        self.activate()
        # Capturing mode determines whether or not the rubber band
        # will follow the moving cursor, once there's a vertex in the chamber
        self.startCapturing()

        # For snapping
        self.snapIndicator = QgsSnapIndicator(plugin.iface.mapCanvas())
        self.snapper = plugin.iface.mapCanvas().snappingUtils()

        self.last_tree = None
        self.last_graph = None
        self.included_chunks = []

        # QgsTasks that aren't kept around as objects can sometimes not run!
        # so if we don't track them, we get issues
        self.task_trash = []
        # Track the time of the last solve request to avoid spamming the server
        # is either None or a time.time()
        self.last_solve = None

        # For showing chunks remaining, status, and instructions to the user.
        self.is_message_bar_visible = False

        # Declare properties
        self.progressMessageBar = None
        self.statusLabel = None
        self.chunksRemainingPB = None

    def handleMetadata(self, chunks_today, chunks_left_today, pricing_tier, fly_instance_id):
        if fly_instance_id:
            self.fly_instance_id = fly_instance_id

        if self.is_message_bar_visible:
            # Merely update the text
            if pricing_tier != 'full-time' and chunks_today > chunks_left_today * 0.75:
                self.chunksRemainingPB.setValue(chunks_today)
                self.chunksRemainingPB.setMaximum(chunks_left_today)
        else:
            # Only show the progress bar if we're more than 75% into their quota,
            # AND they're not on the full time tier (which is generous)
            if pricing_tier != 'full-time' and chunks_today > chunks_left_today * 0.75:
                self.progressMessageBar = self.plugin.iface.messageBar().createMessage("AI Vectorizer")
                self.statusLabel = QLabel(f"Approaching today's chunk quota for current plan {pricing_tier}")
                self.chunksRemainingPB = QProgressBar()
                self.chunksRemainingPB.setValue(chunks_today)
                self.chunksRemainingPB.setMaximum(chunks_left_today)
                self.chunksRemainingPB.setFormat("%v / %m chunks")
                self.chunksRemainingPB.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
                self.progressMessageBar.layout().addWidget(self.statusLabel)
                self.progressMessageBar.layout().addWidget(self.chunksRemainingPB)
                self.plugin.iface.messageBar().pushWidget(self.progressMessageBar, Qgis.Info)

                self.is_message_bar_visible = True

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
    def notifyUserOfMessage(self, msg, msg_type, link_url, link_text, duration):
        widget = self.plugin.iface.messageBar().createMessage("AI Vectorizer", msg)
        button = QPushButton(widget)

        if link_url is not None and link_text is not None:
            button.setText(link_text)
            button.pressed.connect(lambda: QDesktopServices.openUrl(QUrl(link_url)))
        else:
            button.setText("Open Settings")
            button.pressed.connect(self.plugin.openSettings)

        widget.layout().addWidget(button)
        self.plugin.iface.messageBar().pushWidget(widget, msg_type, duration=duration)

    def trimVerticesToPoint(self, vertices: List[QgsPointXY], pt: QgsPointXY) -> List[QgsPointXY]:
        assert len(vertices) >= 2

        last_point, last_point_idx = find_closest_projection_point(vertices, pt)
        points = vertices[:last_point_idx+1] + [last_point]

        return points

    def indexToPoint(self, idx: int) -> QgsPointXY:
        x_min, y_min, dxdy, y_max = self.last_tree.params

        (_, _, opt_points) = self.last_graph
        if str(idx) not in opt_points:
            img_height, img_width = self.last_tree.img_params
            node = np.unravel_index(idx, (img_height, img_width))
        else:
            node = opt_points[str(idx)]# if idx in opt_points else node

        return QgsPointXY(node[1] * dxdy + x_min * 256 * dxdy, y_max * 256 * dxdy - node[0] * dxdy)

    def solvePathToPoint(self, pt: QgsPointXY) -> List[QgsPointXY]:
        if self.last_tree is None or len(self.vertices) == 0:
            return None

        # (x_min, y_max, dxdy) = self.last_tree.params
        (_, pts_paths, _) = self.last_graph

        cur_tree = self.last_tree
        # Bad trees
        if len(cur_tree._graph_nodes_coords()) == 0:
            return None

        # Because we are clipping paths, we need to check the two closest nodes.
        path = cur_tree.dijkstra(cur_tree.closest_nodes_to(pt, 2)[0])[0]
        if len(path) == 0:
            return None

        # Replace bits of the path as possible
        minimized_path = [path[0]]
        for i in range(len(path)-1):
            prev, next = path[i], path[i+1]

            if f"{prev}_{next}" in pts_paths:
                minimized_path.extend(pts_paths[f"{prev}_{next}"][1:])
            elif f"{next}_{prev}" in pts_paths:
                minimized_path.extend(reversed(pts_paths[f"{next}_{prev}"][:-1]))
            else:
                minimized_path.append(next)

        # convert to coordinates
        coordinates = [ self.indexToPoint(idx) for idx in minimized_path ]

        # If snapping is enabled
        if self.isAutoSnapEnabled():
            # Snap only if the snapping result is not empty
            snapped_points = [ self.snapper.snapToMap(coord) for coord in coordinates ]
            coordinates = [ snapMatch.point() if not snapMatch.point().isEmpty() else coord for snapMatch, coord in zip(snapped_points, coordinates) ]

        # Trim to the closest point to the cursor
        # If we trim the path, then new tree root is NOT
        # where the last vertex is.
        if len(coordinates) > 2:
            coordinates = self.trimVerticesToPoint(coordinates, pt)

        # minimized_paths is [ QgsPointXY, ... ]
        return coordinates

    # This is cached and we manually reset it when we get a new feature.
    @lru_cache(maxsize=1)
    def calculateDxDy(self):
        project_crs = QgsProject.instance().crs()
        # By default, we zoom out 2.5x from the user's perspective.
        proj_crs_units_per_screen_pixel = 2.5 * (self.plugin.iface.mapCanvas().extent().width() / self.plugin.iface.mapCanvas().width())

        # We require the first two vertices to be drawn before we consider
        # how large a chunk is.
        if len(self.vertices) < 2:
            raise ValueError

        rlayers = find_raster_layers(QgsProject.instance().layerTreeRoot())
        # Assuming self.rlayers is a list of QgsRasterLayer objects
        # If the user drags in a raster layer without a CRS, default behavior is to give it "unknown"
        # aka invalid CRS, which (to my knowledge) does not reproject and is equivalent to being in the same CRS.
        intersecting_layers = [ rlayer for rlayer in rlayers if layerDoesIntersect(rlayer, project_crs, self.vertices[0]) ]

        # ( units in project CRS ) / ( 1 raster layer's pixel ), independent of raster CRS based on Euclidean approximation
        rupps = [ rasterUnitsPerPixelEstimate(r, project_crs, self.vertices[:2]) for r in intersecting_layers ]

        # Use the resolution of the topmost raster layer
        topmost_res_at_pt = rupps[0] if len(intersecting_layers) >= 1 else proj_crs_units_per_screen_pixel

        # Rendering resolution in units per pixel
        dx = max(proj_crs_units_per_screen_pixel, topmost_res_at_pt)

        return dx

    @lru_cache(maxsize=1)
    def currentUuid(self):
        return str(uuid.uuid4())

    def suggestChunksToLoad(self, cursor_pt: QgsPointXY):
        # Decide which chunks to load based on a set of chunks and relative
        # priorities to eachother.
        # Chunks with negative scores are musts, so these are never rate limited
        # in terms of what to upload.
        # The more positive a score, the less important it is.

        priority_chunks = []
        dxdy = self.calculateDxDy()

        # Get all chunks under the drawn vertices
        priority_chunks.append(Chunk.pointToChunk(self.vertices[-1], dxdy))
        # Current chunk under the cursor
        priority_chunks.append(Chunk.pointToChunk(cursor_pt, dxdy))

        # High priority chunks should not be slowed down by preloading
        return list(set([c for c in priority_chunks if str(c) not in self.chunk_cache ]))

    def canvasMoveEvent(self, e):
        if self.isAutoSnapEnabled():
            snapMatch = self.snapper.snapToMap(e.pos())
            self.snapIndicator.setMatch(snapMatch)

        if self.snapIndicator.match().type():
            pt = self.snapIndicator.match().point()
        else:
            pt = self.toMapCoordinates(e.pos())

        if len(self.vertices) == 0:
            # Nothing to do!
            return

        # Shift key means ignore autocomplete, or we force manual completion on first vertex
        if e.modifiers() & Qt.ShiftModifier or len(self.vertices) == 1:
            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY([self.vertices[-1], pt]),
                None
            )
            return

        dxdy = self.calculateDxDy()
        # Highlight chunk
        cur_chunk = Chunk.pointToChunk(pt, dxdy)
        self.chunk_rb.setToGeometry(cur_chunk.toPolygon(), None)
        self.updateFogOfWar()

        if str(cur_chunk) not in self.chunk_cache or self.chunk_cache[str(cur_chunk)] == False:
            # Pink with more transparency = uploading or not in chunk cache
            self.chunk_rb.setFillColor(QColor(255, 192, 203, 122))  # more transparent pink
        else:
            # Totally transparent = uploaded and solved
            self.chunk_rb.setFillColor(QColor(0, 0, 0, 0))  # completely transparent

        if self.maybeNewSolve(hover_point=pt):
            return

        # Last solve contains this chunk
        elif self.last_tree is not None:
            path_map_pts = self.solvePathToPoint(pt)

            # None = failed to navigate
            if path_map_pts is not None:
                self.rb.setToGeometry(
                    QgsGeometry.fromPolylineXY(path_map_pts),
                    None
                )
                return

        # Draw from last vertex to this one
        self.rb.setToGeometry(
            QgsGeometry.fromPolylineXY([self.vertices[-1], pt]),
            None
        )

    def updateFogOfWar(self):
        if len(self.included_chunks) > 0:
            self.fow_rb.setToGeometry(reduce(
                lambda g1, g2: g1.combine(g2),
                [ Chunk.strToChunk(chunk).toPolygon() for chunk in self.included_chunks ]
            ), None)

    def handleChunkUploaded(self, chunk_strs):
        for chunk_str in chunk_strs:
            self.chunk_cache[chunk_str] = True
    def handleChunkUploadFailed(self, chunk_strs):
        for chunk_str in chunk_strs:
            if chunk_str in self.chunk_cache:
                del self.chunk_cache[chunk_str]

    def canvasPressEvent(self, e):
        pass

    def canvasReleaseEvent(self, e):
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

            # Clean up the plugin between features
            self.clearState()
        elif e.button() == Qt.LeftButton:
            # QgsPointXY with map CRS
            if self.snapIndicator.match().type():
                point = self.snapIndicator.match().point()
            else:
                point = self.toMapCoordinates(e.pos())

            # Solve beforehand, because if no path is found, it should be treated like
            # a normal vectorizer
            queued_points = self.solvePathToPoint(point)
            vertex_px_added = 0

            # Shift key ignores autocomplete and just adds a single vertex
            if e.modifiers() & Qt.ShiftModifier or queued_points is None:
                self.addVertex(point)
                self.vertices.append(point)
            else:
                # Don't duplicate the first point
                if len(queued_points) > 1 and self.vertices[-1].distance(queued_points[0]) < 1e-8:
                    queued_points = queued_points[1:]

                # No shift key, add autocompletions as expected
                for completed_pt in queued_points:
                    self.addVertex(completed_pt)
                    self.vertices.append(completed_pt)

                vertex_px_added = sum([queued_points[i].distance(queued_points[i-1]) for i in range(1, len(queued_points))]) / self.calculateDxDy()

            # We've changed the last vertex, so the previous tree is no
            # longer valid.
            self.last_tree = None
            self.last_graph = None

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            if len(self.vertices) < 2:
                # We don't propose any AI completions until the user has
                # drawn the first line (two vertices).
                return

            # Shift key or not, we re-solve based on the newest point.
            # While the last tree is a DAG, solvePathToPoint often interpolates
            # between nodes in the tree, meaning we can't always use the closest
            # last point and keep the tree.

            should_clear_chunk_cache = len(self.vertices) == 2

            self.maybeNewSolve(hover_point=point, clear_chunk_cache=should_clear_chunk_cache,
                               vertex_px_added=int(vertex_px_added))

    def handleGraphConstructed(self, pts_cost, pts_paths, params, img_params, included_chunks, opt_points, trajectory_root, cur_uuid):
        (x_min, y_min, dxdy, y_max) = params

        # This prevents race conditions where a previously created tree gets stored as the current tree
        # after we right click the feature.
        if cur_uuid != self.currentUuid():
            return

        self.last_tree = TrajectoryTree(pts_cost, (x_min, y_min, dxdy, y_max), img_params, trajectory_root)
        self.last_graph = (pts_cost, pts_paths, opt_points)
        self.included_chunks = included_chunks

        # If the server communicates to us that a chunk we've previously uploaded (chunk_cache[key] == True)
        # is now included in its newest computation, then we should invalidate it from the chunk cache.
        previously_uploaded_chunks = list(self.chunk_cache.keys())
        for prev_uploaded in previously_uploaded_chunks:
            if self.chunk_cache[prev_uploaded] == True and prev_uploaded not in self.included_chunks:
                del self.chunk_cache[prev_uploaded]

        self.updateFogOfWar()

    def keyPressEvent(self, e):
        # e.ignore() is .accept() for some reason
        if e.key() in (Qt.Key_Backspace, Qt.Key_Delete) and len(self.vertices) >= 2:
            if not e.isAutoRepeat():
                self.undo()
                self.vertices.pop()

                e.ignore()

                # Deleting should re-solve the trajectory tree
                if len(self.vertices) >= 2:
                    self.maybeNewSolve(hover_point=self.vertices[-1])
                return
        elif e.key() == Qt.Key_Escape:
            self.stopCapturing()
            self.vertices = []
            self.rb.reset()

            e.ignore()
            return

        e.accept()

    # Determines if we need a new upload + solve task. Returns True if that task was fired,
    # or False if the current tree likely works, or otherwise cannot solve.
    def maybeNewSolve(self, hover_point, clear_chunk_cache=False,
                      vertex_px_added=0):
        # Clear chunk cache first
        if clear_chunk_cache:
            self.chunk_cache = dict()

        chunks_to_load = self.suggestChunksToLoad(hover_point)

        # Only preload chunks on move if we're not currently uploading
        # (chunk_cache[x] is False if we're uploading)
        if not all(self.chunk_cache.values()):
            return False

        # We need a new solve if there's more chunks or the
        # root has changed since our last solve.
        if self.last_tree is not None and len(chunks_to_load) == 0:
            # No new chunks, last tree still works.
            return False
        elif self.last_tree is None:
            # No tree, so solve unless we have a request in flight.
            # Current timeout: 5 seconds
            if self.last_solve is not None and (time.time() - self.last_solve) < 5:
                return False

        root = QgsProject.instance().layerTreeRoot()
        rlayers = find_raster_layers(root)
        project_crs = QgsProject.instance().crs()

        vlayer = self.plugin.iface.activeLayer()
        if not isinstance(vlayer, QgsVectorLayer):
            self.notifyUserOfMessage("The active layer is not a vector layer.", Qgis.Warning, None, None, 10)
            return False
        elif vlayer.wkbType() not in [QgsWkbTypes.LineString, QgsWkbTypes.MultiLineString,
                                    QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon]:
            self.notifyUserOfMessage("The active layer's geometry type is not compatible with this plugin. Please use LineString, MultiLineString, Polygon, or MultiPolygon.", Qgis.Warning, None, None, 10)
            return False

        chunk_task = UploadChunkAndSolveTask(
            self,
            vlayer,
            rlayers,
            project_crs,
            chunks=chunks_to_load,
            should_solve=True,
            clear_chunk_cache=clear_chunk_cache,
            vertex_px_added=vertex_px_added
        )

        self.last_solve = time.time()
        for c in chunks_to_load:
            self.chunk_cache[str(c)] = False

        chunk_task.taskCompleted.connect(lambda: self.handleChunkUploaded([ str(c) for c in chunks_to_load ]))
        chunk_task.taskTerminated.connect(lambda: self.handleChunkUploadFailed([ str(c) for c in chunks_to_load ]))
        chunk_task.taskCompleted.connect(lambda: self.clearSolve())
        chunk_task.taskTerminated.connect(lambda: self.clearSolve())
        chunk_task.clearCache.connect(lambda: self.clearCache())
        chunk_task.messageReceived.connect(lambda e: self.notifyUserOfMessage(*e))
        chunk_task.graphConstructed.connect(lambda args: self.handleGraphConstructed(*args))
        chunk_task.metadataReceived.connect(lambda args: self.handleMetadata(*args))

        QgsApplication.taskManager().addTask(chunk_task)
        self.task_trash.append(chunk_task)

    def clearSolve(self):
        self.last_solve = None

    def clearCache(self):
        self.chunk_cache = dict()
        self.included_chunks = []
        self.updateFogOfWar()

    def clearState(self):
        self.rb.reset()
        self.chunk_rb.reset()
        self.fow_rb.reset()
        self.stopCapturing()

        # Delete current state
        self.last_tree = None
        self.last_graph = None
        self.chunk_cache = dict()
        self.fly_instance_id = None
        self.vertices = []

        # dxdy aka raster resolution
        self.calculateDxDy.cache_clear()
        # unique uuid for each feature
        self.currentUuid.cache_clear()

        # drop message
        if self.is_message_bar_visible:
            self.is_message_bar_visible = False
            self.plugin.iface.messageBar().popWidget(self.progressMessageBar)

    def deactivate(self):
        self.clearState()

        self.plugin.action.setChecked(False)
