# Copyright 2023 Bunting Labs, Inc.

from qgis.PyQt.QtNetwork import QNetworkRequest, QNetworkReply
from qgis.core import QgsTask, QgsMessageLog, Qgis, QgsNetworkAccessManager
from qgis.PyQt.QtCore import pyqtSignal, QUrl, QEventLoop
from PyQt5.QtCore import QByteArray

MESSAGE_CATEGORY = 'EmailRegisterTask'

class EmailRegisterTask(QgsTask):
    """This task registers a user's email"""

    finishedSignal = pyqtSignal(str)
    
    def __init__(self, description, user_email, user_agent, api_key):
        super().__init__(description, QgsTask.CanCancel)
        self.user_email = user_email
        self.user_agent = user_agent
        self.api_key = api_key
        self.exception = None

    def run(self):
        """Perform the email registration"""
        QgsMessageLog.logMessage('Started task "{}"'.format(self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        url = QUrl('https://buntinglabs.com/account/register')
        request = QNetworkRequest(url)
        request.setHeader(QNetworkRequest.UserAgentHeader, self.user_agent)
        request.setRawHeader(b'Host', b'buntinglabs.com')

        data = QByteArray()
        data.append(f"email={QUrl.toPercentEncoding(self.user_email).data().decode()}")
        data.append(f"&api_key={QUrl.toPercentEncoding(self.api_key).data().decode()}")

        try:
            nam = QgsNetworkAccessManager.instance()
            reply = nam.post(request, data)

            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()

            if reply.error() == QNetworkReply.NoError:
                self.finishedSignal.emit('created')
                return True
            else:
                if reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) == 500:
                    if reply.readAll().data().decode() == 'refresh_token':
                        self.finishedSignal.emit('refresh_token')
                        return True
                self.exception = Exception(f'Registration failed with status code: {reply.attribute(QNetworkRequest.HttpStatusCodeAttribute)}')
                self.finishedSignal.emit('failed')
                return False
        except Exception as e:
            self.exception = e
            self.finishedSignal.emit('failed')
            return False
    def finished(self, result):
        """This function is called when the task has completed"""
        if result:
            QgsMessageLog.logMessage('Email registration task "{}" completed successfully'.format(self.description()),
                                     MESSAGE_CATEGORY, Qgis.Success)
        else:
            if self.exception is None:
                QgsMessageLog.logMessage('Email registration task "{}" was not successful but without exception'.format(self.description()),
                                         MESSAGE_CATEGORY, Qgis.Warning)
            else:
                QgsMessageLog.logMessage('Email registration task "{}" Exception: {}'.format(self.description(), self.exception),
                                         MESSAGE_CATEGORY, Qgis.Critical)

    def cancel(self):
        QgsMessageLog.logMessage('Email registration task "{}" was canceled'.format(self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()



class ValidateEmailTask(QgsTask):
    """This task validates the email by making a POST request"""

    finishedSignal = pyqtSignal(bool)

    def __init__(self, description, api_key):
        super().__init__(description, QgsTask.CanCancel)
        self.api_key = api_key
        self.exception = None

    def run(self):
        headers = {
            'host': 'qgis-api.buntinglabs.com',
            'x-api-key': self.api_key,
            'content-type': 'text/plain'
        }

        try:
            url = QUrl("https://qgis-api.buntinglabs.com/chunk/v2")
            request = QNetworkRequest(url)
            for key, value in headers.items():
                request.setRawHeader(key.encode(), value.encode())

            nam = QgsNetworkAccessManager.instance()
            reply = nam.post(request, b'foo')

            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()

            if reply.error() != QNetworkReply.NoError:
                self.exception = Exception(reply.readAll().data().decode('utf-8'))
                self.finishedSignal.emit(False)
                return False

        except Exception as e:
            self.exception = Exception(f'Error: {str(e)}')
            self.finishedSignal.emit(False)
            return False

        self.finishedSignal.emit(True)
        return True

    def finished(self, result):
        """This function is called when the task has completed"""
        if not result and self.exception:
            raise self.exception

    def cancel(self):
        QgsMessageLog.logMessage('API token validation task "{}" was canceled'.format(self.description()),
                                 MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()