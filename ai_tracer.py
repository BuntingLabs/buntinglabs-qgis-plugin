# Copyright 2023 Bunting Labs, Inc.

from enum import Enum
from collections import namedtuple

from qgis.PyQt.QtCore import Qt, QSettings, QUrl
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtGui import QColor, QDesktopServices
from qgis.gui import QgsMapToolCapture, QgsRubberBand, QgsVertexMarker, \
    QgsSnapIndicator
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsPoint, QgsWkbTypes, QgsLayerTreeLayer
from PyQt5.QtCore import pyqtSignal

from .tracing_task import AutocompleteTask, HoverTask

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

AutocompleteCacheEntry = namedtuple('AutocompleteCacheEntry', ['uniq_id', 'px', 'py'])

class AutocompleteCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        else:
            # Move the key to the end to show that it was recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    def set(self, key, value):
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

    predictedPointsReceived = pyqtSignal(tuple)

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
        self.predictedPointsReceived.connect(lambda pts: self.updateRubberBand(self.vertices[-1], []))

        self.map_cache = None # MapCacheEntry
        self.autocomplete_cache = AutocompleteCache(250)

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

        # Support for streaming capture technique
        # if self.isStreamingCapture():
        #     last_point = self.vertices[-1]

        #     if pt.distance(last_point) > self.streamingToleranceInPixels:
        #         # We need to add the point, but we ask that the parent class do this
        #         # because we don't have access to mAllowAddingStreamingPoints
        #         super(AIVectorizerTool, self).canvasMoveEvent(e)
        #         self.vertices.append(pt)

        # Check if the shift key is being pressed
        # We have special existing-line-editing mode when shift is hit
        if len(self.vertices) >= 2 and self.map_cache is not None and not (e.modifiers() & Qt.ShiftModifier):

            print('creating HoverTask')
            # -(y2 - self.y_max) / self.dy, (x2 - self.x_min) / self.dx
            pxys = [self.vertices[-1], (pt.x(), pt.y())]
            (x_min, dx, y_max, dy) = (self.map_cache.x_min, self.map_cache.dx, self.map_cache.y_max, self.map_cache.dy)
            pxys = [((px - x_min) / dx, -(py - y_max) / dy) for (px, py) in pxys]

            # Check cache
            hover_cache_entry = AutocompleteCacheEntry(self.map_cache.uniq_id, int(pxys[-1][0]), int(pxys[-1][1]))
            if self.autocomplete_cache.get(hover_cache_entry):
                print('cache HIT')
                self.predicted_points = [QgsPointXY(*pt) for pt in self.autocomplete_cache.get(hover_cache_entry)]
                self.predictedPointsReceived.emit((None,))
            else:
                print('cache MISS')
                ht = HoverTask(self, self.map_cache, pxys)

                ht.messageReceived.connect(lambda e: print('error', e))#self.notifyUserOfMessage(*e))
                def handleGeometryReceived(o, pts):
                    transformed_points = [((jx * dx) + x_min, y_max - (ix * dy)) for (ix, jx) in pts]

                    self.autocomplete_cache.set(hover_cache_entry, transformed_points)

                    o.predicted_points = [QgsPointXY(*pt) for pt in transformed_points]
                    self.predictedPointsReceived.emit((None,))

                # Replace the lambda with the method call
                ht.geometryReceived.connect(lambda pts: handleGeometryReceived(self, pts))

                QgsApplication.taskManager().addTask(
                    ht,
                )

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
            self.predicted_points = []
            last_point = self.vertices[-1]

            # self.scissors_icon.hide()

            # Close it!
            points = [ self.vertices[0], last_point, QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

            self.rb.setFillColor(get_complement(self.digitizingFillColor()))
            self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine or (len(self.vertices) < 2) and not (e.modifiers() & Qt.ShiftModifier):
            self.updateRubberBand(last_point, [pt])
        elif self.mode() == QgsMapToolCapture.CapturePolygon and not (e.modifiers() & Qt.ShiftModifier):
            self.rb.setToGeometry(
                poly_geo,
                None
            )

    def updateRubberBand(self, last_point, appended_points=[]):
        if len(self.predicted_points) > 0:
            # trim predicted points
            if len(appended_points) > 0:
                trimmedPredictedPoints = self.trimVerticesToPoint(self.predicted_points, appended_points[0])
            else:
                trimmedPredictedPoints = self.predicted_points

            points = [last_point] + trimmedPredictedPoints + appended_points

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
        elif e.button() == Qt.LeftButton and e.modifiers() & Qt.ShiftModifier and len(self.predicted_points) > 0:
            # Add points including the predicted_points
            for pt in self.predicted_points:
                self.addVertex(pt)
                self.vertices.append(pt)

            self.predicted_points = []
        elif e.button() == Qt.LeftButton:
            # QgsPointXY with map CRS
            if self.snapIndicator.match().type():
                point = self.snapIndicator.match().point()
            else:
                point = self.toMapCoordinates(e.pos())

            wasDoubleClick = len(self.vertices) >= 1 and point.distance(self.vertices[-1]) == 0

            self.predicted_points = []

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

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )

    def handleCacheEntryCreated(self, uniq_id, dx, dy, x_min, y_max, window_size):
        self.map_cache = MapCacheEntry(uniq_id, dx, dy, x_min, y_max, window_size)

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

            e.accept()
            return

        e.ignore()

    def deactivate(self):
        self.rb.reset()

        # self.scissors_icon.hide()
        self.plugin.action.setChecked(False)