# Copyright 2023 Bunting Labs, Inc.

import urllib
from urllib import request, parse

from qgis.core import QgsTask, QgsMessageLog, Qgis
from qgis.PyQt.QtCore import pyqtSignal

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
        url = 'https://buntinglabs.com/account/register'

        try:
            data = parse.urlencode({
                'email': self.user_email,
                'api_key': self.api_key
            }).encode()
            headers = {'User-Agent': self.user_agent, 'Host': 'buntinglabs.com'}
            req = request.Request(url, data=data, headers=headers)  # POST request with custom user agent and host
            with request.urlopen(req) as response:
                if response.getcode() == 200:
                    self.finishedSignal.emit('created')
                    return True
                else:
                    self.exception = Exception('Registration failed with status code: {}'.format(response.getcode()))
                    self.finishedSignal.emit('failed')
                    return False
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 500:
                if e.read().decode() == 'refresh_token':
                    self.finishedSignal.emit('refresh_token')
                    return True
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
                raise self.exception

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
            'x-api-key': self.api_key
        }

        try:
            # Empty post body but we authenticate before anything else
            req = request.Request("https://qgis-api.buntinglabs.com/v1", data=b'foo', headers=headers)
            with request.urlopen(req) as response:
                if response.getcode() != 200:
                    self.exception = Exception(response.read().decode('utf-8'))
                    self.finishedSignal.emit(False)
                    return False
        except BrokenPipeError:
            self.exception = Exception('Got BrokenPipeError when trying to connect to inference server')
            self.finishedSignal.emit(False)
            return False
        except Exception as e:
            response_body = e.read().decode('utf-8')
            print(response_body)

            self.exception = Exception('Response body: {}'.format(response_body))
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