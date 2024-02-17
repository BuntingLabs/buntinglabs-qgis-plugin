# Copyright 2023 Bunting Labs, Inc.

import os
from typing import List

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class OnboardingHeaderWidget(QWidget):
    def __init__(self, steps: List[str], active_step: int, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)

        logo = QPixmap(os.path.join(os.path.dirname(__file__), 'assets', 'bunting_bird.png'))
        logo_label = QLabel()
        logo_label.setPixmap(logo.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(logo_label)

        for i, step in enumerate(steps):
            step_widget = StepWidget(step, i == active_step)
            self.layout.addWidget(step_widget)
            step_widget.show()

class StepWidget(QWidget):
    def __init__(self, text: str, active: bool, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create a circle label
        self.circle = QLabel(self)
        self.circle.setFixedSize(20, 20)
        self.circle.setStyleSheet("border: 1px solid; border-radius: 10px;")
        if active:
            self.circle.setStyleSheet("border: 1px solid; border-radius: 10px; background-color: #1B6585;")
        self.layout.addWidget(self.circle, 0, Qt.AlignHCenter) # Vertically align the circle above the label
        
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter) # Center align the text
        self.layout.addWidget(self.label) # Add the label to the layout

        font = self.label.font()
        font.setPointSizeF(font.pointSize() * 1.25) # Upgrade font size by 1.25x

        if active:
            font.setBold(True)

        self.label.setFont(font)
