# Copyright 2023 Bunting Labs, Inc.

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QPushButton
from qgis.gui import QgsMapToolCapture
from qgis.core import Qgis, QgsFeature, QgsApplication, QgsPointXY, \
    QgsGeometry, QgsPolygon, QgsProject, QgsVectorLayer, QgsRasterLayer

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
        self.current_mode = current_mode

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

    def notifyUserOfAuthError(self, e):
        widget = self.plugin.iface.messageBar().createMessage("Error", "Could not call AI vectorizer: %s." % e)
        button = QPushButton(widget)
        button.setText("Edit API key")
        button.pressed.connect(self.plugin.openSettings)
        widget.layout().addWidget(button)
        self.plugin.iface.messageBar().pushWidget(widget, Qgis.Critical, duration=15)

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

            if self.current_mode == QgsMapToolCapture.CaptureLine:
                g = QgsGeometry(curve.curveToLine())
                f.setGeometry(g)
            elif self.current_mode == QgsMapToolCapture.CapturePolygon:
                poly = QgsPolygon()
                poly.setExteriorRing(curve.curveToLine())

                f.setGeometry(QgsGeometry(poly))
            else:
                raise ValueError

            vlayer.addFeature(f)
            # Don't let vertices cross over
            self.vertices = []

            self.stopCapturing()
        elif e.button() == Qt.LeftButton:
            # QgsPointXY with map CRS
            point = self.toMapCoordinates(e.pos())

            self.addVertex(point)
            self.vertices.append(point)

            # This just sets the capturing property to true so we can
            # repeatedly call it
            self.startCapturing()

            # Create our autocomplete task if we have >=2 vertices
            if len(self.vertices) >= 2:
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

                self.autocomplete_task.pointReceived.connect(lambda args: self.addVertex(QgsPointXY(*args[0])))
                self.autocomplete_task.errorReceived.connect(lambda e: self.notifyUserOfAuthError(e))

                QgsApplication.taskManager().addTask(
                    self.autocomplete_task,
                )
