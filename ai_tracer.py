# Copyright 2023 Bunting Labs, Inc.

from qgis.PyQt.QtCore import Qt, QSettings
from qgis.PyQt.QtWidgets import QPushButton
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapToolCapture, QgsRubberBand
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer, \
    QgsPoint, QgsWkbTypes

from .tracing_task import AutocompleteTask

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
        last_point_idx = sorted(enumerate(self.vertices), key=lambda v: (v[1].x()-pt.x())**2 + (v[1].y()-pt.y())**2)[0][0]

        # Find the closest point
        last_point = QgsPointXY(self.vertices[last_point_idx].x(), self.vertices[last_point_idx].y())

        # Geometry for polygon
        points = [ QgsPointXY(v.x(), v.y()) for v in self.vertices[:last_point_idx] ]
        # Close it!
        points += [ QgsPointXY(last_point.x(), last_point.y()), QgsPointXY(pt.x(), pt.y()), points[0]]

        poly_geo = QgsGeometry.fromPolygonXY([points])

        if trimToPoint:
            numToTrim = len(self.vertices)-last_point_idx
            for _ in range(numToTrim):
                self.undo()
            self.vertices = self.vertices[:-numToTrim]

        return (last_point, poly_geo)

    def canvasMoveEvent(self, e):
        if len(self.vertices) == 0:
            # Nothing to do!
            return

        pt = self.toMapCoordinates(e.pos())

        # Check if the shift key is being pressed
        # We hide the geometry rubber band when the shift button goes down
        if e.modifiers() & Qt.ShiftModifier:
            # Shift means our last vertex should effectively be the closest point to the line
            (last_point, poly_geo) = self.shiftClickAdjustment(e.pos())
        else:
            last_point = QgsPointXY(self.vertices[-1].x(), self.vertices[-1].y())

            # Close it!
            points = self.vertices + [ QgsPointXY(pt.x(), pt.y()), self.vertices[0]]

            poly_geo = QgsGeometry.fromPolygonXY([points])

        # geometry depends on capture mode
        if self.mode() == QgsMapToolCapture.CaptureLine or (len(self.vertices) <= 2):
            points = [last_point, QgsPointXY(pt.x(), pt.y())]
            self.rb.setToGeometry(
                QgsGeometry.fromPolylineXY(points),
                None
            )
        elif self.mode() == QgsMapToolCapture.CapturePolygon:
            # points = [ QgsPointXY(v.x(), v.y()) for v in self.vertices[:last_point_idx] ]
            # Close it!
            # points += [ QgsPointXY(last_point.x(), last_point.y()), QgsPointXY(pt.x(), pt.y()), points[0]]

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
