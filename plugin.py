# Copyright 2024 Bunting Labs, Inc.

import os
import random

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, \
    QPushButton, QAction, QHBoxLayout, QCheckBox, QButtonGroup, QRadioButton
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtCore import QSettings, Qt, QSize, QTimer

from qgis.core import Qgis, QgsApplication, QgsVectorLayer, QgsWkbTypes
from qgis.gui import QgsMapToolCapture
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtCore import QUrl

from PyQt5.QtCore import QSize

from .ai_tracer import AIVectorizerTool
from .login_check_task import EmailRegisterTask, ValidateEmailTask
from .onboarding_widget import OnboardingHeaderWidget


# Settings for QGIS
SETTING_API_TOKEN = "buntinglabs-qgis-plugin/api_key"
SETTING_TOS = "buntinglabs-qgis-plugin/terms_of_service_state"

def generate_random_api_key():
    readable_chars = 'ABCDEFGHJKLMNPRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'
    return ''.join(random.choice(readable_chars) for _ in range(12))

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

        icon_path = os.path.join(os.path.dirname(__file__), "vectorizing_icon.png")
        self.action = QAction(QIcon(icon_path), '<b>Vectorize with AI</b><p>Toggle editing on a vector layer then enable to autocomplete new geometries.</p>', None)
        self.tracer = AIVectorizerTool(self)

        # Read the plugin version
        try:
            plugin_metadata = os.path.join(os.path.dirname(__file__), "metadata.txt")
            with open(plugin_metadata, 'r') as f:
                for line in f.readlines():
                    if line.startswith('version='):
                        self.plugin_version = line.split('=')[1].strip()
        except:
            self.plugin_version = 'N/A'

        self.settings_action = None

        # For progress bar, time in ms per number of chunks
        self.expected_time = {
            0: 500,
            1: 1200,
            2: 1900
        }
        # Timer to check after the GUI is created
        self.vis_timer = None

    def update_checkable(self):
        # Before onboarding, it's always checkable.
        if self.settings.value(SETTING_TOS, "") != "y" or self.settings.value(SETTING_API_TOKEN, "") == "":
            self.action.setEnabled(True)
            return

        layer = self.iface.activeLayer()

        if layer is not None and isinstance(layer, QgsVectorLayer) and layer.isEditable():
            self.action.setEnabled(True)
        else:
            self.action.setEnabled(False)

    def current_layer_change_event(self, layer):
        # if the current layer is editable, or becomes editable in the future,
        # switch our plugin status
        if isinstance(layer, QgsVectorLayer):
            layer.editingStarted.connect(self.update_checkable)
            layer.editingStopped.connect(self.update_checkable)

        self.update_checkable()

    # msg_type is Qgis.Critical, Qgis.Info, Qgis.Warning, Qgis.success
    def notifyUserOfMessage(self, msg, msg_type, link_url, link_text, duration):
        widget = self.iface.messageBar().createMessage("AI Vectorizer", msg)
        button = QPushButton(widget)

        if link_url is not None and link_text is not None:
            button.setText(link_text)
            button.pressed.connect(lambda: QDesktopServices.openUrl(QUrl(link_url)))
        else:
            button.setText("Open Settings")
            button.pressed.connect(self.openSettings)

        widget.layout().addWidget(button)
        self.iface.messageBar().pushWidget(widget, msg_type, duration=duration)

    def initGui(self):
        # Initialize the plugin GUI

        # Because we don't pass iface.mainWindow() to the QAction constructor in __init__
        self.iface.mainWindow().addAction(self.action)

        self.action.setCheckable(True)
        self.action.triggered.connect(self.toolbarClick)
        self.iface.addToolBarIcon(self.action)

        # Trigger a current layer change event to get the right action
        self.current_layer_change_event(self.iface.activeLayer())

        # Let the user change settings
        self.settings_action = QAction(
            QIcon(":images/themes/default/console/iconSettingsConsole.svg"),
            'Edit Settings',
            self.iface.mainWindow()
        )

        self.settings_action.triggered.connect(self.openSettings)
        self.iface.addPluginToMenu('Bunting Labs', self.settings_action)

        # Fire timer
        self.vis_timer = QTimer()
        self.vis_timer.singleShot(5000, self.checkPluginToolbarVisibility)

    def checkPluginToolbarVisibility(self):
        # If the user doesn't have the plugins toolbar visible, show a warning.
        if not self.iface.pluginToolBar().isVisible():
            self.notifyUserOfMessage("The plugins toolbar is not visible. Enable it in View > Toolbars > Plugins Toolbar.",
                                     Qgis.Warning,
                                     'https://youtu.be/Wm1pTj55Rys',
                                     'Watch Tutorial',
                                     120)

    # There are five dialogs in our onboarding flow.
    # 1. TOS dialog, requesting consent to our terms.
    # 2. Email dialog, asking for the email of their account.
    # 3. Confirm dialog, where our auto-generated API key can be confirmed.
    # 4. Token dialog, when the API key must be generated from the dashboard.
    # 5. Instructions dialog, onboarding them.
    # Let's create each of these.

    # 1. TOS dialog, which leads to email dialog if accepted.
    def openTOSDialog(self):
        tos_dialog = QDialog(self.iface.mainWindow())
        tos_dialog.setWindowTitle("Bunting Labs AI Vectorizer")
        # Only called after the user clicks "I accept terms of service"
        # in the GUI, so store the user's choice and continue.
        tos_dialog.accepted.connect(lambda: self.settings.setValue(SETTING_TOS, "y"))
        tos_dialog.accepted.connect(lambda: self.openEmailDialog())

        layout = QVBoxLayout(tos_dialog)
        layout.setContentsMargins(40, 20, 40, 20) # Add padding to the layout
        layout.setSpacing(10) # Add spacing between widgets

        layout.addWidget(OnboardingHeaderWidget([
            "Introduction", "Create account", "Verify email"
        ], 0))

        intro_text = QLabel("<p>This plugin uses AI to autocomplete tracing raster maps.</p>")
        intro_text.setWordWrap(True)
        layout.addWidget(intro_text)

        pixmap = QPixmap(os.path.join(os.path.dirname(__file__), 'assets', 'plugin_data_flow.png')).scaled(422, 396, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap_label = QLabel()
        pixmap_label.setPixmap(pixmap)
        pixmap_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(pixmap_label)

        explain_text = QLabel("Because your maps are sent to our servers to run the AI, you must agree to our <a href=\"https://buntinglabs.com/legal/terms\">terms of service</a> to use the plugin.")
        explain_text.setOpenExternalLinks(True)
        explain_text.setWordWrap(True)
        layout.addWidget(explain_text)

        tos_layout = QHBoxLayout()

        reject_button = QPushButton("Abort installation")
        reject_button.clicked.connect(lambda: tos_dialog.reject())
        tos_layout.addWidget(reject_button)

        accept_button = QPushButton("I accept terms of service")
        accept_button.clicked.connect(lambda: tos_dialog.accept())
        accept_button.setDefault(True)
        tos_layout.addWidget(accept_button)

        layout.addLayout(tos_layout)

        version_label = QLabel(f"Bunting Labs AI Vectorizer v{self.plugin_version}")
        version_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(version_label)

        tos_dialog.setLayout(layout)
        tos_dialog.exec_()

    # 2. Email dialog, which triggers emailSubmitted below.
    def openEmailDialog(self):
        email_dialog = QDialog(self.iface.mainWindow())
        email_dialog.setWindowTitle("Bunting Labs AI Vectorizer")
        email_dialog.accepted.connect(self.emailSubmitted)

        layout = QVBoxLayout(email_dialog)
        layout.setContentsMargins(40, 20, 40, 20) # Add padding to the layout
        layout.setSpacing(10) # Add spacing between widgets

        layout.addWidget(OnboardingHeaderWidget([
            "Introduction", "Create account", "Verify email"
        ], 1))

        intro_text = QLabel("To prevent abuse of our servers, we need to verify that you're human.")
        intro_text.setWordWrap(True)
        layout.addWidget(intro_text)

        explain_text = QLabel("You will automatically be on our free trial, with a limit of 150 map chunks for evaluation. No credit card is needed.")
        explain_text.setWordWrap(True)
        layout.addWidget(explain_text)

        email_button_layout = QHBoxLayout()

        # put on self because it's accessed in emailSubmitted
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Your work email here")
        email_button_layout.addWidget(self.email_input)

        start_button = QPushButton("Create account")
        start_button.clicked.connect(lambda: email_dialog.accept())
        email_button_layout.addWidget(start_button)

        layout.addLayout(email_button_layout)

        version_label = QLabel(f"Bunting Labs AI Vectorizer v{self.plugin_version}")
        version_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(version_label)

        email_dialog.setLayout(layout)
        email_dialog.exec_()

    # Triggered in openEmailDialog
    def emailSubmitted(self):
        user_email = self.email_input.text()

        # Randomly generate an api key and submit that to the website
        new_api_key = generate_random_api_key()

        self.register_task = EmailRegisterTask(
            "Registering user's email",
            user_email,
            f"BuntingLabsQGISAIVectorizer/{self.plugin_version}",
            new_api_key
        )
        self.register_task.finishedSignal.connect(lambda status: self.confirmEmailPlease(status, new_api_key))

        QgsApplication.taskManager().addTask(
            self.register_task,
        )

    # This either 4. asks them for a secret token from /dashboard, or
    # 3. asks them to confirm their email.
    def confirmEmailPlease(self, status: str, new_api_key: str):
        # If status was 'created', confirm email.
        # if status was 'error'
        if status == 'failed':
            self.iface.messageBar().pushCritical(
                'AI Vectorizer onboarding failed',
                'Please schedule a call to debug.',
            )
            return
        elif status == 'refresh_token':
            # Ignore randomly generated API key

            token_dialog = QDialog(self.iface.mainWindow())
            token_dialog.setWindowTitle("Bunting Labs AI Vectorizer")
            token_dialog.accepted.connect(self.tokenSubmitted)

            layout = QVBoxLayout(token_dialog)
            layout.setContentsMargins(40, 20, 40, 20) # Add padding to the layout
            layout.setSpacing(10) # Add spacing between widgets

            layout.addWidget(OnboardingHeaderWidget([
                "Introduction", "Create account", "Retrieve Token"
            ], 2))

            intro_text = QLabel("To continue, please copy your secret token from <a href=\"https://buntinglabs.com/dashboard\">your dashboard</a> and paste it below.")
            intro_text.setWordWrap(True)
            intro_text.setOpenExternalLinks(True)
            layout.addWidget(intro_text)

            explain_text = QLabel("This will connect your QGIS plugin to your free account. Keep your token a secret.")
            explain_text.setWordWrap(True)
            layout.addWidget(explain_text)

            secret_button_layout = QHBoxLayout()

            self.token_input = QLineEdit()
            self.token_input.setPlaceholderText("Your secret token here")
            secret_button_layout.addWidget(self.token_input)

            start_button = QPushButton("Save secret token")
            start_button.clicked.connect(lambda: token_dialog.accept())
            secret_button_layout.addWidget(start_button)

            layout.addLayout(secret_button_layout)

            version_label = QLabel(f"Bunting Labs AI Vectorizer v{self.plugin_version}")
            version_label.setAlignment(Qt.AlignCenter)

            layout.addWidget(version_label)

            token_dialog.setLayout(layout)
            token_dialog.exec_()
        else:
            assert status == 'created'

            # Save API key, because setting it was successful
            self.settings.setValue(SETTING_API_TOKEN, new_api_key)

            self.email_confirm_dialog = QDialog(self.iface.mainWindow())
            self.email_confirm_dialog.setWindowTitle("Bunting Labs AI Vectorizer")

            layout = QVBoxLayout(self.email_confirm_dialog)
            layout.setContentsMargins(40, 20, 40, 20) # Add padding to the layout
            layout.setSpacing(10) # Add spacing between widgets

            layout.addWidget(OnboardingHeaderWidget([
                "Introduction", "Create account", "Verify email"
            ], 2))

            intro_text = QLabel("Please open your email inbox, and find the email from <code>login@stytch.com</code> titled “Your account creation request for Bunting Labs”.")
            intro_text.setWordWrap(True)
            layout.addWidget(intro_text)

            pixmap = QPixmap(os.path.join(os.path.dirname(__file__), 'assets', 'confirm_email.png')).scaled(512, 295, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmap_label = QLabel()
            pixmap_label.setPixmap(pixmap)
            pixmap_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(pixmap_label)

            explain_text = QLabel("Click “Continue” in the email to activate your account. You can then return to QGIS and finish installation by clicking the button below.")
            explain_text.setWordWrap(True)
            layout.addWidget(explain_text)

            status_layout = QHBoxLayout()

            self.email_validated_status = QLabel("")
            self.email_validated_status.setAlignment(Qt.AlignCenter)
            self.email_validated_status.setStyleSheet("QLabel {color: darkred; font-size: 18px;}")
            status_layout.addWidget(self.email_validated_status)

            done_button = QPushButton("I've clicked the link")
            done_button.clicked.connect(lambda: self.checkForAccount())
            done_button.setDefault(True)
            status_layout.addWidget(done_button)

            layout.addLayout(status_layout)

            version_label = QLabel(f"Bunting Labs AI Vectorizer v{self.plugin_version}")
            version_label.setAlignment(Qt.AlignCenter)

            layout.addWidget(version_label)

            self.email_confirm_dialog.setLayout(layout)
            self.email_confirm_dialog.exec_()

    def tokenSubmitted(self):
        self.settings.setValue(SETTING_API_TOKEN, self.token_input.text())

        self.check_for_account_task = ValidateEmailTask('validate email', self.settings.value(SETTING_API_TOKEN, ""))
        self.check_for_account_task.finishedSignal.connect(lambda status: status and self.showLastOnboardingTutorial())
        QgsApplication.taskManager().addTask(
            self.check_for_account_task,
        )

    def checkForAccount(self):
        self.email_validated_status.setText("⟳")

        self.check_for_account_task = ValidateEmailTask('validate email', self.settings.value(SETTING_API_TOKEN, ""))
        self.check_for_account_task.finishedSignal.connect(self.updateEmailValidation)

        QgsApplication.taskManager().addTask(
            self.check_for_account_task,
        )

    def updateEmailValidation(self, status):
        if status:
            self.email_validated_status.setText("✓")
            self.email_validated_status.setStyleSheet("QLabel {color: green; font-size: 24px;}")

            self.email_confirm_dialog.accept()
            self.showLastOnboardingTutorial()
        else:
            self.email_validated_status.setText("You haven't clicked the link yet")
            self.email_validated_status.setStyleSheet("QLabel {color: darkred; font-size: 18px;}")

    # 5. the ultimate page
    def showLastOnboardingTutorial(self):
        onboarding_dialog = QDialog(self.iface.mainWindow())
        onboarding_dialog.setWindowTitle("Bunting Labs AI Vectorizer")
        layout = QVBoxLayout()

        split_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("<b>You're in! Here's how you can start using the plugin:</b>"))
        left_layout.addWidget(QLabel("1. Load a raster layer to digitize"))
        left_layout.addWidget(QLabel("2. Create a new vector layer (e.g. Shapefile)"))
        left_layout.addWidget(QLabel("3. Toggle editing mode"))
        left_layout.addWidget(QLabel("4. Click the AI Vectorizer icon in the Plugins toolbar"))
        left_layout.addWidget(QLabel("5. Click two vertices along the feature you want to digitize"))
        left_layout.addWidget(QLabel("6. Move your mouse forwards along the feature to extend the AI tracing forward and load more chunks of the map"))
        left_layout.addWidget(QLabel("7. Left click to accept the AI progress and autocomplete again from that point"))
        left_layout.addWidget(QLabel("8. Hold down <code>shift</code> to manually add vertices"))
        left_layout.addWidget(QLabel("9. Right click to save the feature to your vector layer"))

        gif_label = QLabel()
        gif_movie = QMovie(os.path.join(os.path.dirname(__file__), 'assets', 'instructions.gif'))
        gif_movie.setScaledSize(QSize(int(724/1.5), int(480/1.5)))
        gif_movie.start()
        gif_label.setMovie(gif_movie)

        right_layout.addWidget(gif_label)

        split_layout.addLayout(left_layout)
        split_layout.addLayout(right_layout)

        layout.addLayout(split_layout)

        close_button = QPushButton("Get started")
        close_button.clicked.connect(onboarding_dialog.accept)
        layout.addWidget(close_button)

        onboarding_dialog.setLayout(layout)
        onboarding_dialog.exec_()

    def openSettings(self):
        # Create a closeable modal for API key input
        self.api_key_dialog = QDialog(self.iface.mainWindow())
        self.api_key_dialog.setWindowTitle("AI Vectorizer Settings")
        layout = QVBoxLayout(self.api_key_dialog)

        title_label = QLabel("<b>Terms of Service</b>")
        layout.addWidget(title_label)

        label_with_link = QLabel("You can find our terms of service posted <a href='https://buntinglabs.com/legal/terms'>on our website</a>. Agreement is required to use the AI-enabled autocomplete.")
        label_with_link.setOpenExternalLinks(True)
        label_with_link.setWordWrap(True)
        layout.addWidget(label_with_link)

        tos_checkbox = QCheckBox("I agree to the above terms of service")
        if self.settings.value(SETTING_TOS, "") == "y":
            tos_checkbox.setChecked(True)
        tos_checkbox.stateChanged.connect(lambda: self.settings.setValue(SETTING_TOS, "y" if tos_checkbox.isChecked() else ""))
        layout.addWidget(tos_checkbox)

        title_label = QLabel("<b>Account Secret Key</b>")
        layout.addWidget(title_label)

        label = QLabel("Put your account's secret key here to use the AI vectorizer. You can find your account secret key at <a href='http://buntinglabs.com/dashboard'>your dashboard</a> after signing up for free.")
        label.setOpenExternalLinks(True)
        label.setWordWrap(True)

        sublabel = QLabel("If the secret key has been auto-filled, there's no need to change it, unless your plugin cannot connect.")

        self.api_key_input = QLineEdit()
        self.api_key_input.setText(self.settings.value(SETTING_API_TOKEN, "demo"))

        layout.addWidget(label)
        layout.addWidget(self.api_key_input)
        layout.addWidget(sublabel)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.saveSettings)
        layout.addWidget(save_button)

        self.api_key_dialog.setLayout(layout)
        self.api_key_dialog.exec_()

    def saveSettings(self):
        # Save the API key from the input field to the settings
        self.settings.setValue(SETTING_API_TOKEN, self.api_key_input.text())

        self.iface.messageBar().pushMessage(
            "Bunting Labs AI Vectorizer",
            "Settings saved successfully.",
            Qgis.Info,
            duration=10
        )

        self.api_key_dialog.close()

    def unload(self):
        # Stop and delete the timer to prevent it from running after the plugin is unloaded
        if self.vis_timer is not None:
            if self.vis_timer.isActive():
                self.vis_timer.stop()
            self.vis_timer.deleteLater()

        if self.settings_action is not None:
            self.iface.removePluginMenu('Bunting Labs', self.settings_action)

        self.iface.removeToolBarIcon(self.action)

        self.tracer.deactivate()

    def toolbarClick(self):
        if self.action.isChecked():
            # Depending on how many settings someone has already set,
            # we'll need to revisit the onboarding flow.
            if self.settings.value(SETTING_TOS, "") != "y":
                self.action.setChecked(False)
                self.openTOSDialog()
                return
            elif self.settings.value(SETTING_API_TOKEN, "") == "":
                self.action.setChecked(False)
                self.openEmailDialog()
                return

            self.iface.mapCanvas().setMapTool(self.tracer)

            self.action.setChecked(True)
        else:
            # disable
            self.action.setChecked(False)

            self.iface.actionPan().trigger()
