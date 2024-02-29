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

from .tracing_task import AutocompleteTask

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
        self.num_completions = 50

        # List of QgsPointXY that represents the new feature
        # via QgsMapTool.toMapCoordinates(), it's in project CRS
        self.vertices = []

        self.autocomplete_task = None

        # And take control and go full on editing mode
        self.activate()
        # Capturing mode determines whether or not the rubber band
        # will follow the moving cursor, once there's a vertex in the chamber
        self.startCapturing()

        self.shift_state = ShiftClickState.HAS_NOT_CUT

        self.scissors_icon = QgsVertexMarker(plugin.iface.mapCanvas())
        self.scissors_icon.setIconType(QgsVertexMarker.ICON_X)
        self.scissors_icon.setColor(get_complement(self.digitizingStrokeColor()))
        self.scissors_icon.setIconSize(18)
        self.scissors_icon.setPenWidth(5)
        self.scissors_icon.setZValue(1000)

        # For snapping
        self.snapIndicator = QgsSnapIndicator(plugin.iface.mapCanvas())
        self.snapper = plugin.iface.mapCanvas().snappingUtils()

    def initRubberBand(self):
        if self.mode() == QgsMapToolCapture.CaptureLine:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
        elif self.mode() == QgsMapToolCapture.CapturePolygon:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        else:
            # TODO not sure how we could get here
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)

        rb.setFillColor(self.digitizingFillColor())
        rb.setStrokeColor(self.digitizingStrokeColor())
        rb.setWidth(self.digitizingStrokeWidth())
        rb.setLineStyle(Qt.DotLine)

        return rb

    def notifyUserOfError(self, msg, link_url, link_text):
        widget = self.plugin.iface.messageBar().createMessage("Error", "Could not call AI vectorizer: %s." % msg)
        button = QPushButton(widget)

        if link_url is not None and link_text is not None:
            button.setText(link_text)
            button.pressed.connect(lambda: QDesktopServices.openUrl(QUrl(link_url)))
        else:
            button.setText("Open Settings")
            button.pressed.connect(self.plugin.openSettings)

        widget.layout().addWidget(button)
        self.plugin.iface.messageBar().pushWidget(widget, Qgis.Critical, duration=15)

    def handlePointReceived(self, args):
        self.addVertex(QgsPointXY(*args[0]))
        self.vertices.append(QgsPointXY(*args[0]))

    def shiftClickAdjustment(self, pt, trimToPoint=False):

        # Return the geometry prior
        assert len(self.vertices) >= 2
        last_point, last_point_idx = find_closest_projection_point(self.vertices, pt)

        # Geometry for polygon
        points = self.vertices[:last_point_idx+1] + [ last_point, pt, self.vertices[0] ]
        poly_geo = QgsGeometry.fromPolygonXY([points])

        if trimToPoint:
            numToTrim = len(self.vertices)-last_point_idx-1
            for _ in range(numToTrim):
                self.undo()
            self.vertices = self.vertices[:-numToTrim]
            # After trimming, add back our projected point
            self.addVertex(last_point)
            self.vertices.append(last_point)

        return (last_point, poly_geo)

    def isCutting(self, last_point):
        # Returns whether or not we are in cutting mode or draw forwards mode.
        # If this would cut to the end of the line, though, of course we would not
        # cut there, as it's a no-op.
        wouldCutToEnd = last_point.distance(self.vertices[-1]) < 1e-8

        return self.shift_state == ShiftClickState.HAS_NOT_CUT and not wouldCutToEnd

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

        # Check if the shift key is being pressed
        # We have special existing-line-editing mode when shift is hit
        if e.modifiers() & Qt.ShiftModifier and len(self.vertices) >= 2:
            (last_point, poly_geo) = self.shiftClickAdjustment(pt)

            if self.isCutting(last_point):
                # Shift means our last vertex should effectively be the closest point to the line
                self.scissors_icon.setCenter(last_point)
                self.scissors_icon.show()

                # Hide the rubber band
                self.rb.reset()

                return
            else:
                # We've already cut, so now we're drawing lines without autocomplete.
                last_point = self.vertices[-1]
                self.scissors_icon.hide()

                # Use complement color
                self.rb.setFillColor(get_complement(self.digitizingFillColor()))
                self.rb.setStrokeColor(get_complement(self.digitizingStrokeColor()))
        else:
            last_point = self.vertices[-1]

            self.scissors_icon.hide()

            # Close it!
            points = [ self.vertices[0], last_point, QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

            self.rb.setFillColor(self.digitizingFillColor())
            self.rb.setStrokeColor(self.digitizingStrokeColor())

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine or (len(self.vertices) < 2):
            points = [last_point, pt]
            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY(points),
                None
            )
        elif self.mode() == QgsMapToolCapture.CapturePolygon:
            self.rb.setToGeometry(
                poly_geo,
                None
            )

    def canvasPressEvent(self, e):
        pass

    def canvasReleaseEvent(self, e):
        # Either click will cancel an ongoing autocomplete
        if self.autocomplete_task is not None:
            # QgsTasks passed to a task manager end up being owned
            # in C++ land which leads us ... here.
            try:
                self.autocomplete_task.cancel()
            except RuntimeError:
                pass
            finally:
                self.autocomplete_task = None

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
        elif e.button() == Qt.LeftButton:
            # QgsPointXY with map CRS
            if self.snapIndicator.match().type():
                point = self.snapIndicator.match().point()
            else:
                point = self.toMapCoordinates(e.pos())

            if len(self.vertices) >= 2 and (e.modifiers() & Qt.ShiftModifier) and self.shift_state == ShiftClickState.HAS_NOT_CUT:
                self.shiftClickAdjustment(point, trimToPoint=True)
                # Only trim once, then we're completing
                self.shift_state = ShiftClickState.HAS_CUT
                return

            # Left clicking without shift resets us to normal autocomplete mode
            if not (e.modifiers() & Qt.ShiftModifier):
                self.shift_state = ShiftClickState.HAS_NOT_CUT

            wasDoubleClick = len(self.vertices) >= 1 and point.distance(self.vertices[-1]) == 0

            self.addVertex(point)
            self.vertices.append(point)

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            # Create our autocomplete task if we have >=2 vertices
            if len(self.vertices) >= 2 and not (e.modifiers() & Qt.ShiftModifier) and not wasDoubleClick:
                root = QgsProject.instance().layerTreeRoot()
                rlayers = [node.layer() for node in root.children() if isinstance(node, QgsLayerTreeLayer) and isinstance(node.layer(), QgsRasterLayer) and node.itemVisibilityChecked()]

                project_crs = QgsProject.instance().crs()

                self.autocomplete_task = AutocompleteTask(
                    self,
                    vlayer,
                    rlayers,
                    project_crs
                )

                self.autocomplete_task.pointReceived.connect(lambda args: self.handlePointReceived(args))
                self.autocomplete_task.errorReceived.connect(lambda e: self.notifyUserOfError(*e))

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )

    def deactivate(self):
        self.rb.reset()

        self.scissors_icon.hide()
        self.plugin.action.setChecked(False)