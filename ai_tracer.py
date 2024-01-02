# Copyright 2023 Bunting Labs, Inc.

from qgis.PyQt.QtCore import Qt, QSettings
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapToolCapture, QgsRubberBand
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsPoint, QgsWkbTypes

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

# QgsMapToolCapture is a subclass of QgsMapToolEdit that provides
# additional functionality for map tools that capture geometry. It
# is an abstract base class for map tools that capture line and
# polygon geometries. It handles the drawing of rubber bands on the
# map canvas and the capturing of clicks to build the geometry.
class AIVectorizerTool(QgsMapToolCapture):
    def __init__(self, plugin, current_mode):
        # Extend QgsMapToolCapture
        cadDockWidget = plugin.iface.cadDockWidget()
        super(AIVectorizerTool, self).__init__(plugin.iface.mapCanvas(), cadDockWidget, current_mode)

        self.plugin = plugin
        self.rb = self.initRubberBand(current_mode)

        # Options
        self.num_completions = 50

        # List of QgsPointXY that represents the new feature
        self.vertices = []

        self.autocomplete_task = None

        # And take control and go full on editing mode
        self.activate()
        # Capturing mode determines whether or not the rubber band
        # will follow the moving cursor, once there's a vertex in the chamber
        self.startCapturing()

    def initRubberBand(self, mode):
        if mode == QgsMapToolCapture.CaptureLine:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.LineGeometry)
        elif mode == QgsMapToolCapture.CapturePolygon:
            rb = QgsRubberBand(self.plugin.iface.mapCanvas(), QgsWkbTypes.PolygonGeometry)
        else:
            raise ValueError

        rb.setFillColor(self.digitizingFillColor())
        rb.setColor(self.digitizingStrokeColor())
        rb.setWidth(self.digitizingStrokeWidth())
        rb.setLineStyle(Qt.DotLine)

        color = self.digitizingStrokeColor()

        alpha_scale = QSettings().value("Qgis/digitizing/line_color_alpha_scale", 1.0, type=float)
        color.setAlphaF(color.alphaF() * alpha_scale)

        return rb

    def notifyUserOfAuthError(self, e):
        widget = self.plugin.iface.messageBar().createMessage("Error", "Could not call AI vectorizer: %s." % e)
        button = QPushButton(widget)
        button.setText("Edit API key")
        button.pressed.connect(self.plugin.openSettings)
        widget.layout().addWidget(button)
        self.plugin.iface.messageBar().pushWidget(widget, Qgis.Critical, duration=15)

    def handlePointReceived(self, args):
        self.addVertex(QgsPointXY(*args[0]))
        self.vertices.append(QgsPointXY(*args[0]))

    def shiftClickAdjustment(self, pos, trimToPoint=False):
        pt = self.toMapCoordinates(pos)

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

    def canvasMoveEvent(self, e):
        if len(self.vertices) == 0:
            # Nothing to do!
            return

        pt = self.toMapCoordinates(e.pos())

        # Check if the shift key is being pressed
        # We hide the geometry rubber band when the shift button goes down
        if e.modifiers() & Qt.ShiftModifier and len(self.vertices) >= 2:
            # Shift means our last vertex should effectively be the closest point to the line
            (last_point, poly_geo) = self.shiftClickAdjustment(e.pos())

            # Use complement color
            self.rb.setFillColor(get_complement(self.digitizingFillColor()))
            self.rb.setColor(get_complement(self.digitizingStrokeColor()))
        else:
            last_point = self.vertices[-1]

            # Close it!
            points = [ self.vertices[0], last_point, QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

            self.rb.setFillColor(self.digitizingFillColor())
            self.rb.setColor(self.digitizingStrokeColor())

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine or (len(self.vertices) <= 2):
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

        # Right click means we end
        if e.button() == Qt.RightButton:
            vlayer = self.plugin.iface.activeLayer()
            if not isinstance(vlayer, QgsVectorLayer):
                self.plugin.iface.messageBar().pushMessage(
                    "Info",
                    "No active vector layer.",
                    Qgis.Info)
                return

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
            if e.modifiers() & Qt.ShiftModifier:
                self.shiftClickAdjustment(e.pos(), trimToPoint=True)

            # QgsPointXY with map CRS
            point = self.toMapCoordinates(e.pos())

            self.addVertex(point)
            self.vertices.append(point)

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            # Create our autocomplete task if we have >=2 vertices
            if len(self.vertices) >= 2 and not (e.modifiers() & Qt.ShiftModifier):
                root = QgsProject.instance().layerTreeRoot()
                rlayers = [node.layer() for node in root.children() if isinstance(node.layer(), QgsRasterLayer)]
                vlayer = self.plugin.iface.activeLayer()

                project_crs = QgsProject.instance().crs()

                self.autocomplete_task = AutocompleteTask(
                    self,
                    vlayer,
                    rlayers,
                    project_crs
                )

                self.autocomplete_task.pointReceived.connect(lambda args: self.handlePointReceived(args))
                self.autocomplete_task.errorReceived.connect(lambda e: self.notifyUserOfAuthError(e))

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )

    def deactivate(self):
        self.rb.reset()
