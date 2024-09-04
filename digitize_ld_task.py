# Copyright 2024 Bunting Labs, Inc.

from qgis.PyQt.QtNetwork import QNetworkRequest, QNetworkReply
from qgis.core import QgsTask, QgsMessageLog, Qgis, QgsNetworkAccessManager
from qgis.PyQt.QtCore import pyqtSignal, QUrl, QEventLoop

from PyQt5.QtNetwork import QHttpMultiPart, QHttpPart
from PyQt5.QtCore import QFile, QIODevice

import json

MESSAGE_CATEGORY = 'DigitizeLandDescriptionTask'

class DigitizeLandDescriptionTask(QgsTask):
    """This task uploads a land description and digitizes it into a vector"""

    # Tuple for (error message, Qgis.Critical, error link or None, error button text or None, duration)
    messageReceived = pyqtSignal(tuple)

    coordinatesReceived = pyqtSignal(list)

    def __init__(self, description, api_key, pdf_path, lonlat):
        super().__init__(description, QgsTask.CanCancel)

        self.api_key = api_key
        self.pdf_path = pdf_path
        self.lonlat = lonlat

    def run(self):
        # Exceptions are bad
        try:
            # Upload the pdf using multipart file upload
            url = QUrl('https://qgis-api.buntinglabs.com/deed/v1/vectorize')
            request = QNetworkRequest(url)
            request.setRawHeader(b'Host', b'buntinglabs.com')
            request.setRawHeader(b'X-Api-Key', self.api_key.encode())

            nam = QgsNetworkAccessManager.instance()

            multipart = QHttpMultiPart(QHttpMultiPart.FormDataType)

            # Create file part
            file_part = QHttpPart()
            file_part.setHeader(QNetworkRequest.ContentTypeHeader, "application/pdf")
            file_part.setHeader(QNetworkRequest.ContentDispositionHeader, 'form-data; name="file"; filename="{}"'.format(self.pdf_path.split('/')[-1]))

            file = QFile(self.pdf_path)
            file.open(QIODevice.ReadOnly)
            file_part.setBodyDevice(file)
            file.setParent(multipart)

            multipart.append(file_part)

            pob_part = QHttpPart()
            pob_part.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
            pob_part.setHeader(QNetworkRequest.ContentDispositionHeader, 'form-data; name="pob"; filename="pob.json"')
            pob_part.setBody(json.dumps(self.lonlat).encode())

            multipart.append(pob_part)

            reply = nam.post(request, multipart)
            multipart.setParent(reply)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()

            if reply.error() == QNetworkReply.NoError:
                response_data = reply.readAll().data().decode()
                try:
                    coordinates = json.loads(response_data)
                    self.coordinatesReceived.emit(coordinates)
                    return True
                except json.JSONDecodeError:
                    self.messageReceived.emit((f'Unexpected error while parsing coordinate geometry', Qgis.Critical, None, None, 30))
                    return False
            else:
                error_payload = reply.readAll().data().decode()
                try:
                    error_details = json.loads(error_payload)
                    self.messageReceived.emit((
                        error_details.get('message'),
                        Qgis.Critical,
                        error_details.get('link'),
                        error_details.get('link_text'),
                        30
                    ))
                except json.JSONDecodeError:
                    self.messageReceived.emit((f"Unexpected error while digitizing PDF, {error_payload}", Qgis.Critical, None, None, 30))
                return False
        except Exception as e:
            self.messageReceived.emit((f'Unexpected error while digitizing: {str(e)}', Qgis.Critical, None, None, 30))
            return False

    def finished(self, result):
        pass

    def cancel(self):
        QgsMessageLog.logMessage('Land description digitization task "{}" was canceled'.format(self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()
