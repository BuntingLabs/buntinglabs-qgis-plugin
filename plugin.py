# Copyright 2023 Bunting Labs, Inc.

import os

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, \
    QPushButton, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSettings

from qgis.core import Qgis, QgsVectorLayer, QgsWkbTypes
from qgis.gui import QgsMapToolCapture

from .ai_tracer import AIVectorizerTool

class BuntingLabsPlugin:
    # This plugin class contains strictly things that interface with
    # QGIS.

    # The AIVectorizerTool strictly depends on the type of geometry
    # we're tracing, so we re-create it frequently.

    def __init__(self, iface):
        self.iface = iface
        self.iface.currentLayerChanged.connect(self.current_layer_change_event)

        self.settings = QSettings()
        # Set by the text box in save settings
        self.api_key_input = None

        self.tracer = None

    def current_layer_change_event(self, layer):
        # if the current layer is editable, or becomes editable in the future,
        # switch our plugin status
        layer = self.iface.activeLayer()
        if isinstance(layer, QgsVectorLayer):
            self.initTracer()
            self.action.setEnabled(layer is not None and layer.isEditable())

            layer.editingStarted.connect(lambda: self.current_layer_change_event(layer))
            layer.editingStopped.connect(lambda: self.current_layer_change_event(layer))
        else:
            self.action.setEnabled(False)

    def initTracer(self):
        self.tracer = None

        # see what type of layer we are vectorizing
        vlayer = self.iface.activeLayer()
        if isinstance(vlayer, QgsVectorLayer):
            if vlayer.wkbType() in [QgsWkbTypes.LineString, QgsWkbTypes.MultiLineString]:
                current_mode = QgsMapToolCapture.CaptureLine
            elif vlayer.wkbType() == QgsWkbTypes.Polygon:
                current_mode = QgsMapToolCapture.CapturePolygon
            else:
                self.iface.messageBar().pushMessage(
                    "Error",
                    "Unknown vector layer type.",
                    Qgis.Critical)
        self.tracer = AIVectorizerTool(self, current_mode)

    def initGui(self):
        # Initialize the plugin GUI
        icon_path = os.path.join(os.path.dirname(__file__), "vectorizing_icon.png")

        self.action = QAction(QIcon(icon_path), 'Vectorize with AI', self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.toolbarClick)
        self.iface.addToolBarIcon(self.action)

        # Uncheck ourselves if they change tools
        self.iface.mapCanvas().mapToolSet.connect(self.onMapToolChanged)

    def openSettings(self):
        # Create a closeable modal for API key input
        self.api_key_dialog = QDialog(self.iface.mainWindow())
        self.api_key_dialog.setWindowTitle("API Key Settings")

        layout = QVBoxLayout(self.api_key_dialog)
        label = QLabel()
        label.setText("Put your account's API key here to use the AI vectorizer. You can get an API key at <a href='http://buntinglabs.com/dashboard'>your account page</a> after signing up for free.")
        label.setOpenExternalLinks(True)

        self.api_key_input = QLineEdit()
        self.api_key_input.setText(self.settings.value("buntinglabs-qgis-plugin/api_key", "demo"))

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.saveSettings)

        layout.addWidget(label)
        layout.addWidget(self.api_key_input)
        layout.addWidget(save_button)

        self.api_key_dialog.setLayout(layout)
        self.api_key_dialog.exec_()

    def saveSettings(self):
        # Save the API key from the input field to the settings
        api_key = self.api_key_input.text()
        self.settings.setValue("buntinglabs-qgis-plugin/api_key", api_key)

        self.iface.messageBar().pushMessage("Success", "API key changed successfully.", Qgis.Info, duration=10)

        self.api_key_dialog.close()

    def unload(self):
        self.iface.removeToolBarIcon(self.action)

    def onMapToolChanged(self, newTool, _):
        if not isinstance(newTool, AIVectorizerTool):
            self.deactivate()

    def deactivate(self):
        self.action.setChecked(False)
        self.tracer = None

    def toolbarClick(self):
        if self.action.isChecked():
            self.initTracer()
            self.iface.mapCanvas().setMapTool(self.tracer)

            self.action.setChecked(True)
        else:
            # disable
            self.action.setChecked(False)

            self.iface.actionPan().trigger()
