# Copyright 2023 Bunting Labs, Inc.

import os
import http.client
import json
from osgeo import gdal, osr
import numpy as np
import ssl
import math

from qgis.core import QgsTask, QgsMapSettings, QgsMapRendererCustomPainterJob, \
    QgsCoordinateTransform, QgsProject, QgsRectangle, Qgis
from qgis.gui import QgsMapToolCapture
from qgis.PyQt.QtGui import QImage, QPainter, QColor
from qgis.PyQt.QtCore import QSize, pyqtSignal

def rasterUnitsPerPixelEstimate(rlayer, project_crs, vertices):
    # vertices are in project_crs
    assert len(vertices) == 2

    # if same CRS or CRS is "invalid", just do direct calculation
    if rlayer.crs() == project_crs or not rlayer.crs().isValid():
        return rlayer.rasterUnitsPerPixelX()

    # Calculate the Euclidean distance between the last two vertices in the project CRS
    dist_proj_crs = math.sqrt((vertices[-2].x() - vertices[-1].x())**2 + (vertices[-2].y() - vertices[-1].y())**2)

    # Convert points to the CRS of the raster layer
    transform = QgsCoordinateTransform(project_crs, rlayer.crs(), QgsProject.instance())

    point1_rlayer_crs = transform.transform(vertices[-1])
    point2_rlayer_crs = transform.transform(vertices[-2])
    dist_rlayer_crs = math.sqrt((point2_rlayer_crs.x() - point1_rlayer_crs.x())**2 + (point2_rlayer_crs.y() - point1_rlayer_crs.y())**2)

    return (dist_proj_crs / dist_rlayer_crs) * rlayer.rasterUnitsPerPixelX()

def layerDoesIntersect(rlayer, project_crs, vertex):
    # if same CRS or CRS is "invalid", just do direct .contains() check
    if rlayer.crs() == project_crs or not rlayer.crs().isValid():
        return rlayer.extent().contains(vertex)

    transform = QgsCoordinateTransform(project_crs, rlayer.crs(), QgsProject.instance())
    transformed_vertex = transform.transform(vertex)
    return rlayer.extent().contains(transformed_vertex)

class AutocompleteTask(QgsTask):
    # This task can run in the background of QGIS, streaming results
    # back from the inference server.

    pointReceived = pyqtSignal(tuple)
    # Tuple for (error message, Qgis.Critical, error link or None, error button text or None)
    messageReceived = pyqtSignal(tuple)

    # (dx, dy, x_min, y_max)
    parameterComputed = pyqtSignal(tuple)

    def __init__(self, tracing_tool, vlayer, rlayers, project_crs):
        super().__init__(
            'Bunting Labs AI Vectorizer background task for ML inference',
            QgsTask.CanCancel
        )

        self.tracing_tool = tracing_tool
        self.vlayer = vlayer
        self.rlayers = rlayers
        self.project_crs = project_crs

    def run(self):
        # By default, we zoom out 2.5x from the user's perspective.
        proj_crs_units_per_screen_pixel = 2.5 * (self.tracing_tool.plugin.iface.mapCanvas().extent().width() / self.tracing_tool.plugin.iface.mapCanvas().width())

        # The resolution of a raster layer is defined as the ground distance covered by one pixel
        # of the raster. Therefore, a smaller resolution value means a higher resolution raster.
        mapEpsgCode = self.project_crs.postgisSrid()

        # Assuming self.rlayers is a list of QgsRasterLayer objects
        # If the user drags in a raster layer without a CRS, default behavior is to give it "unknown"
        # aka invalid CRS, which (to my knowledge) does not reproject and is equivalent to being in the same CRS.
        intersecting_layers = [ rlayer for rlayer in self.rlayers if layerDoesIntersect(rlayer, self.project_crs, self.tracing_tool.vertices[-1]) ]

        # ( units in project CRS ) / ( 1 raster layer's pixel ), independent of raster CRS based on Euclidean approximation
        rupps = [ rasterUnitsPerPixelEstimate(r, self.project_crs, self.tracing_tool.vertices[-2:]) for r in intersecting_layers ]

        # Use the resolution of the topmost raster layer
        topmost_res_at_pt = rupps[0] if len(intersecting_layers) >= 1 else proj_crs_units_per_screen_pixel

        # Rendering resolution in units per pixel
        dx = max(proj_crs_units_per_screen_pixel, topmost_res_at_pt)
        dy = dx
        # Quadruple the resolution
        # dx *= 4
        # dy *= 4

        if len(self.rlayers) == 0:
            self.messageReceived.emit((
                'No raster layers are loaded. Load a GeoTIFF to use autocomplete.',
                Qgis.Critical, None, None
            ))
            return False

        # First, if they clicked outside of all raster layers, warn them.
        if len(intersecting_layers) == 0:
            self.messageReceived.emit((
                'No raster layers found beneath your autocomplete tool',
                Qgis.Warning, None, None
            ))

        # Size of the rectangle in the CRS coordinates
        window_size = self.tracing_tool.plugin.settings.value("buntinglabs-qgis-plugin/window_size_px", "1200")
        assert window_size in ["1200", "2500"] # Two allowed sizes

        img_width, img_height = int(window_size), int(window_size)
        x_size = img_width * dx
        y_size = img_height * dy

        if x_size <= 0 or y_size <= 0:
            self.messageReceived.emit((
                'Could not render an image from the rasters (this is a plugin bug!).',
                Qgis.Critical,
                'https://github.com/BuntingLabs/buntinglabs-qgis-plugin/issues/new',
                'Report Bug'
            ))
            return False

        # i = y, j = x
        # note that negative i (or y) is up
        x0, y0 = self.tracing_tool.vertices[-2]
        x1, y1 = self.tracing_tool.vertices[-1]
        cx, cy = (x0+x1)/2, (y0+y1)/2

        x_min = cx - x_size / 2
        x_max = cx + x_size / 2
        y_min = cy - y_size / 2
        y_max = cy + y_size / 2

        self.parameterComputed.emit((dx, dy, x_min, y_max))

        # create image
        # Format_RGB888 is 24-bit (8 bits each) for each color channel, unlike
        # Format_RGB32 which by default has 0xff on the alpha channel, and screws
        # up reading it into GDAL!
        img = QImage(QSize(img_width, img_height), QImage.Format_RGB888)

        # white is most canonically background
        color = QColor(255, 255, 255)
        img.fill(color.rgb())

        mapSettings = QgsMapSettings()

        mapSettings.setDestinationCrs(self.project_crs)
        mapSettings.setLayers(self.rlayers)

        rect = QgsRectangle(x_min, y_min, x_max, y_max)
        mapSettings.setExtent(rect)
        mapSettings.setOutputSize(img.size())

        p = QPainter()
        p.begin(img)
        p.setRenderHint(QPainter.Antialiasing)

        render = QgsMapRendererCustomPainterJob(mapSettings, p)
        render.start()
        render.waitForFinished()
        p.end()

        try:
            # Convert QImage to np.array
            ptr = img.bits()
            ptr.setsize(img.height() * img.width() * 3)
            img_np = np.frombuffer(ptr, np.uint8).reshape((img.height(), img.width(), 3))

            # Call the function to convert the image to a geotiff tif and save it as bytes
            tif_data = georeference_img_to_tiff(img_np, mapEpsgCode, x_min, y_max, x_max, y_min)

        except Exception as e:
            self.messageReceived.emit((str(e), Qgis.Critical, None, None))
            return False

        # Get all coordinates
        preceding_coordinates = [ [-(y - y_max)/dy, (x - x_min)/dx] for (x, y) in self.tracing_tool.vertices ]
        vector_payload = json.dumps({
            'coordinates': preceding_coordinates
        })

        options_payload = json.dumps({
            'num_completions': self.tracing_tool.num_completions,
            'qgis_version': Qgis.QGIS_VERSION,
            'plugin_version': self.tracing_tool.plugin.plugin_version,
            'proj_epsg': mapEpsgCode,
            'is_polygon': self.tracing_tool.mode() == QgsMapToolCapture.CapturePolygon
        })

        boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
        body = (
            '--' + boundary,
            'Content-Disposition: form-data; name="image"; filename="rendered.tif"',
            'Content-Type: application/octet-stream',
            '',
            tif_data,
            '--' + boundary,
            'Content-Disposition: form-data; name="vector"; filename="vector.json"',
            'Content-Type: application/json',
            '',
            vector_payload,
            '--' + boundary,
            'Content-Disposition: form-data; name="options"; filename="options.json"',
            'Content-Type: application/json',
            '',
            options_payload,
            '--' + boundary + '--',
            ''
        )
        body = b'\r\n'.join([part.encode() if isinstance(part, str) else part for part in body])

        headers = {
            'Content-Type': 'multipart/form-data; boundary=' + boundary,
            'x-api-key': self.tracing_tool.plugin.settings.value("buntinglabs-qgis-plugin/api_key", "demo")
        }

        try:
            conn = http.client.HTTPSConnection("qgis-api.buntinglabs.com")
            conn.request("POST", "/v1", body, headers)
            res = conn.getresponse()
            if res.status != 200:
                error_payload = res.read().decode('utf-8')

                try:
                    error_details = json.loads(error_payload)
                    self.messageReceived.emit((
                        error_details.get('message'),
                        Qgis.Critical,
                        error_details.get('link'),
                        error_details.get('link_text')
                    ))
                except json.JSONDecodeError:
                    self.messageReceived.emit((error_payload, Qgis.Critical, None, None))

                return False
        except BrokenPipeError:
            self.messageReceived.emit(('Autocomplete server connection was interrupted (BrokenPipeError)', Qgis.Critical, None, None))
            return False
        except ssl.SSLCertVerificationError:
            self.messageReceived.emit(('Autocomplete server failed SSL Certificate Verification', Qgis.Critical, None, None))
            return False
        except Exception as e:
            self.messageReceived.emit((f'Error connecting to autocomplete server: {str(e)}', Qgis.Critical, None, None))
            return False

        buffer = ""
        while True:
            # For some reason, read errors with IncompleteRead?
            try:
                chunk = res.read(16)
                if not chunk:
                    break

                buffer += chunk.decode('utf-8')
            except http.client.IncompleteRead as e:
                buffer += e.partial.decode('utf-8')

            while '\n' in buffer:
                if self.isCanceled():
                    return False

                line, buffer = buffer.split('\n', 1)
                new_point = json.loads(line)

                ix, jx = new_point[0], new_point[1]

                # convert to xy
                xn = (jx * dx) + x_min
                yn = y_max - (ix * dy)

                self.pointReceived.emit(((xn, yn), 1.0))

        return True

    def finished(self, result):
        pass

    def cancel(self):
        super().cancel()



class HoverTask(QgsTask):
    # This task can run in the background of QGIS, streaming results
    # back from the inference server.

    geometryReceived = pyqtSignal(list)
    # Tuple for (error message, Qgis.Critical, error link or None, error button text or None)
    messageReceived = pyqtSignal(tuple)

    def __init__(self, tracing_tool, d_params, xy_params, cursor_pt):
        super().__init__(
            'Bunting Labs AI Vectorizer background task for ML inference',
            QgsTask.CanCancel
        )

        self.tracing_tool = tracing_tool

        (self.dx, self.dy) = d_params
        (self.x_min, self.y_max) = xy_params
        (self.cursor_x, self.cursor_y) = cursor_pt

    def run(self):
        print('running HoverTask')
        # i = y, j = x
        # note that negative i (or y) is up
        x0, y0 = self.tracing_tool.vertices[-2]
        x1, y1 = self.tracing_tool.vertices[-1]
        x2, y2 = self.cursor_x, self.cursor_y

        # px0, py0 = -(y0 - self.y_max) / self.dy, (x0 - self.x_min) / self.dx
        px1, py1 = -(y1 - self.y_max) / self.dy, (x1 - self.x_min) / self.dx
        px2, py2 = -(y2 - self.y_max) / self.dy, (x2 - self.x_min) / self.dx

        headers = {
            'x-api-key': self.tracing_tool.plugin.settings.value("buntinglabs-qgis-plugin/api_key", "demo")
        }

        try:
            print('connecting to staging server...')
            conn = http.client.HTTPSConnection("fly-inference-staging-night-2042.fly.dev")
            conn.request("GET", f"/v2?x1={px1}&y1={py1}&x2={px2}&y2={py2}", headers=headers)
            res = conn.getresponse()
            if res.status != 200:
                error_payload = res.read().decode('utf-8')

                try:
                    error_details = json.loads(error_payload)
                    self.messageReceived.emit((
                        error_details.get('message'),
                        Qgis.Critical,
                        error_details.get('link'),
                        error_details.get('link_text')
                    ))
                except json.JSONDecodeError:
                    self.messageReceived.emit((error_payload, Qgis.Critical, None, None))

                return False
        except BrokenPipeError:
            self.messageReceived.emit(('Autocomplete server connection was interrupted (BrokenPipeError)', Qgis.Critical, None, None))
            return False
        except ssl.SSLCertVerificationError:
            self.messageReceived.emit(('Autocomplete server failed SSL Certificate Verification', Qgis.Critical, None, None))
            return False
        except Exception as e:
            self.messageReceived.emit((f'Error connecting to autocomplete server: {str(e)}', Qgis.Critical, None, None))
            return False

        try:
            response_data = res.read().decode('utf-8')
            print('response data', response_data)
            path_points = json.loads(response_data)

            transformed_points = [((jx * self.dx) + self.x_min, self.y_max - (ix * self.dy)) for (ix, jx) in path_points]

            self.geometryReceived.emit(transformed_points)
        except json.JSONDecodeError as e:
            print('e', e)
            self.messageReceived.emit(('Failed to parse JSON response from server', Qgis.Critical, None, None))
            return False

        return True

    def finished(self, result):
        pass

    def cancel(self):
        super().cancel()


def georeference_img_to_tiff(img_np, epsg, x_min, y_min, x_max, y_max):
    # Open the PNG file
    (rasterYSize, rasterXSize, rasterCount) = img_np.shape

    # Create a new GeoTIFF file in memory
    dst = gdal.GetDriverByName('GTiff').Create('/vsimem/bunting_qgis_tracer.tif', rasterXSize, rasterYSize, rasterCount,
                                               gdal.GDT_Byte, options=["COMPRESS=JPEG", "JPEG_QUALITY=85"])

    # Set the geotransform
    geotransform = [x_min, (x_max-x_min)/rasterXSize, 0, y_min, 0, (y_max-y_min)/rasterYSize]
    dst.SetGeoTransform(geotransform)

    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst.SetProjection(srs.ExportToWkt())

    # Write the array data to the raster bands
    for b in range(rasterCount):
        band = dst.GetRasterBand(b + 1)
        band.WriteArray(img_np[:, :, b])

    # Close the files
    dst = None

    # Return the GeoTIFF-encoded memory contents as a byte array
    f = gdal.VSIFOpenL('/vsimem/bunting_qgis_tracer.tif', 'rb')
    # Because we use the same /vsimem/ URI for each query, double clicking quickly
    # can result in a race condition in georeference_img_to_tiff where it gets .Unlink()'ed
    # before the above open call. This means we get a null pointer here. TODO solve
    # more elegantly, but for now, we'll error out.
    if f is None:
        raise RuntimeError("Autocomplete was used too quickly, please wait a second between requests.")

    gdal.VSIFSeekL(f, 0, os.SEEK_END)
    size = gdal.VSIFTellL(f)
    gdal.VSIFSeekL(f, 0, os.SEEK_SET)
    data = gdal.VSIFReadL(1, size, f)
    gdal.VSIFCloseL(f)

    # Delete the temporary file
    gdal.Unlink('/vsimem/bunting_qgis_tracer.tif')

    return data
