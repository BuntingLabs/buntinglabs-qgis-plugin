# Copyright 2023 Bunting Labs, Inc.

from enum import Enum

from qgis.PyQt.QtCore import Qt, QSettings, QUrl
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtGui import QColor, QDesktopServices
from qgis.gui import QgsMapToolCapture, QgsRubberBand, QgsVertexMarker, \
    QgsSnapIndicator
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsPoint, QgsWkbTypes, QgsLayerTreeLayer

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

        self.scissors_icon = QgsVertexMarker(plugin.iface.mapCanvas())
        self.scissors_icon.setIconType(QgsVertexMarker.ICON_X)
        self.scissors_icon.setColor(get_complement(self.digitizingStrokeColor()))
        self.scissors_icon.setIconSize(18)
        self.scissors_icon.setPenWidth(5)
        self.scissors_icon.setZValue(1000)

        # For snapping
        self.snapIndicator = QgsSnapIndicator(plugin.iface.mapCanvas())
        self.snapper = plugin.iface.mapCanvas().snappingUtils()

        self.streamingToleranceInPixels = int(QSettings().value('qgis/digitizing/stream_tolerance', 2))

        self.predicted_points = []

    # This will only be called in QGIS is older than 3.32, hopefully.
    def supportsTechnique(self, technique):
        # we do not support shape or circular
        return (technique in [
            Qgis.CaptureTechnique.StraightSegments,
            Qgis.CaptureTechnique.Streaming
        ])

    # Wrap currentCaptureTechnique() because it was only added in 3.32.
    def isStreamingCapture(self):
        if hasattr(self, 'currentCaptureTechnique') and hasattr(Qgis, 'CaptureTechnique'):
            if hasattr(Qgis.CaptureTechnique, 'Streaming'):
                return self.currentCaptureTechnique() == Qgis.CaptureTechnique.Streaming
        return False

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
        if self.isStreamingCapture():
            last_point = self.vertices[-1]

            if pt.distance(last_point) > self.streamingToleranceInPixels:
                # We need to add the point, but we ask that the parent class do this
                # because we don't have access to mAllowAddingStreamingPoints
                super(AIVectorizerTool, self).canvasMoveEvent(e)
                self.vertices.append(pt)

        # Check if the shift key is being pressed
        # We have special existing-line-editing mode when shift is hit
        elif e.modifiers() & Qt.ShiftModifier and len(self.vertices) >= 2:

            print('creating HoverTask')
            ht = HoverTask(self, (self.dx, self.dy), (self.x_min, self.y_max), (pt.x(), pt.y()))
            # ht.pointReceived.connect(lambda args: self.handlePointReceived(args))
            # (last_point, poly_geo) = self.shiftClickAdjustment(pt)
            ht.messageReceived.connect(lambda e: print('error', e))#self.notifyUserOfMessage(*e))
            def handleGeometryReceived(o, pts):
                o.predicted_points = [QgsPointXY(*pt) for pt in pts]

            # Replace the lambda with the method call
            ht.geometryReceived.connect(lambda pts: handleGeometryReceived(self, pts))

            # ht.geometryReceived.connect(lambda pts: self.rb.setToGeometry(
            #     QgsGeometry.fromPolylineXY([self.vertices[-1]] + ),
            #     None
            # ))

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
            last_point = self.vertices[-1]

            self.scissors_icon.hide()

            # Close it!
            points = [ self.vertices[0], last_point, QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

            self.rb.setFillColor(self.digitizingFillColor())
            self.rb.setStrokeColor(self.digitizingStrokeColor())

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine or (len(self.vertices) < 2) and not (e.modifiers() & Qt.ShiftModifier):
            points = [last_point] + self.predicted_points + [pt]
            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY(points),
                None
            )
            if len(self.predicted_points) > 0:
                self.rb.setFillColor(get_complement(self.digitizingFillColor()))
                self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))
        elif self.mode() == QgsMapToolCapture.CapturePolygon and not (e.modifiers() & Qt.ShiftModifier):
            self.rb.setToGeometry(
                poly_geo,
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
        elif e.button() == Qt.LeftButton and self.isStreamingCapture():
            # Forces adding a vertex manually
            if self.snapIndicator.match().type():
                point = self.snapIndicator.match().point()
            else:
                point = self.toMapCoordinates(e.pos())

            self.addVertex(point)
            self.vertices.append(point)

            self.startCapturing()
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

                self.autocomplete_task.parameterComputed.connect(lambda params: self.handleParameterComputed(params))

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )

    def handleParameterComputed(self, params):
        (dx, dy, x_min, y_max) = params

        self.dx = dx
        self.dy = dy
        self.x_min = x_min
        self.y_max = y_max

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

        self.scissors_icon.hide()
        self.plugin.action.setChecked(False)