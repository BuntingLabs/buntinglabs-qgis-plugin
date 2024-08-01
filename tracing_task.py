# Copyright 2024 Bunting Labs, Inc.

import os
import json
from osgeo import gdal, osr
import numpy as np
import random
import time
import gzip
import math

from qgis.core import QgsTask, QgsMapSettings, QgsMapRendererCustomPainterJob, \
    QgsCoordinateTransform, QgsProject, QgsNetworkAccessManager, Qgis
from qgis.PyQt.QtGui import QImage, QPainter, QColor
from qgis.PyQt.QtCore import QSize, pyqtSignal, QUrl, QEventLoop, QTimer
from qgis.PyQt.QtNetwork import QNetworkRequest, QNetworkReply

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

class UploadChunkAndSolveTask(QgsTask):
    # This task can run in the background of QGIS

    # Tuple for (error message, Qgis.Critical, error link or None, error button text or None, duration)
    messageReceived = pyqtSignal(tuple)

    graphConstructed = pyqtSignal(tuple)
    metadataReceived = pyqtSignal(tuple) # (chunks_today, chunks_left, pricing_tier, fly_instance_id)

    clearCache = pyqtSignal() # clears chunk-related caching

    def __init__(self, tracing_tool, vlayer, rlayers, project_crs,
                 chunks=[], vertex_px_added=0,
                 should_solve=False, clear_chunk_cache=False):
        super().__init__(
            'AI Vectorizer processing map chunks on server' if should_solve else 'AI Vectorizer preloading map chunks',
            QgsTask.CanCancel
        )

        self.tracing_tool = tracing_tool
        self.vlayer = vlayer
        self.rlayers = rlayers
        self.project_crs = project_crs

        self.chunks = chunks
        self.should_solve = should_solve
        self.clear_chunk_cache = clear_chunk_cache
        self.vertex_px_added = vertex_px_added

        # Store and return later
        self.cur_uuid = self.tracing_tool.currentUuid()

        # For handling race conditions: don't emit graphConstructed
        # if the last vertex has changed since this task started
        self.last_vertex = self.tracing_tool.vertices[-1]

        # For progress bar
        self.start_time = time.time()
        ETs = self.tracing_tool.plugin.expected_time
        self.expected_time = ETs[len(chunks)] if len(chunks) in ETs else 3000

        self.hasNotifiedOfBoot = False

    def run(self):
        self.setProgress(0.0)

        # The resolution of a raster layer is defined as the ground distance covered by one pixel
        # of the raster. Therefore, a smaller resolution value means a higher resolution raster.
        mapCrsWkt = self.project_crs.toWkt()

        mapSettings = QgsMapSettings()
        mapSettings.setDestinationCrs(self.project_crs)
        mapSettings.setLayers(self.rlayers)

        rendered_chunks = []
        for chunk in self.chunks:

            # create image
            # Format_RGB888 is 24-bit (8 bits each) for each color channel, unlike
            # Format_RGB32 which by default has 0xff on the alpha channel, and screws
            # up reading it into GDAL!
            img = QImage(QSize(chunk.chunk_size, chunk.chunk_size), QImage.Format_RGB888)

            # white is most canonically background
            color = QColor(255, 255, 255)
            img.fill(color.rgb())

            rect = chunk.toRectangle()
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
                rect = chunk.toRectangle()
                x_min, y_min, x_max, y_max = rect.xMinimum(), rect.yMinimum(), rect.xMaximum(), rect.yMaximum()

                tif_data = georeference_img_to_tiff(img_np, mapCrsWkt, x_min, y_min, x_max, y_max)
                rendered_chunks.append(tif_data)

            except Exception as e:
                self.messageReceived.emit((str(e), Qgis.Critical, None, None, 15))
                return False

        boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
        # Build the solve body if requested
        body = []
        if self.should_solve:
            # Get all coordinates
            vector_payload = json.dumps({
                'coordinates': [ [y, x] for (x, y) in self.tracing_tool.vertices ]
            })

            # If we don't have enough vertices, then the user right clicked before this task started,
            # which means it's actually out of date and should be ignored.
            if len(self.tracing_tool.vertices) < 2:
                return False

            body.extend([
                '--' + boundary,
                'Content-Disposition: form-data; name="solve_vector"; filename="vector.json"',
                'Content-Type: application/json',
                '',
                vector_payload,
                '--' + boundary + '--' if len(rendered_chunks) == 0 else ''
            ])

        for i, tif_data in enumerate(rendered_chunks):
            body.extend((
                '--' + boundary,
                'Content-Disposition: form-data; name="%s"; filename="rendered.tif"' % str(self.chunks[i]),
                'Content-Type: application/octet-stream',
                '',
                tif_data,
                '--' + boundary + '--' if i == len(rendered_chunks) - 1 else ''
            ))

        body = b'\r\n'.join([part.encode() if isinstance(part, str) else part for part in body])

        headers = {
            'Content-Type': 'multipart/form-data; boundary=' + boundary,
            'Accept-Encoding': 'gzip',
            # If the user puts in a newline character or a space it breaks everything
            'x-api-key': self.tracing_tool.plugin.settings.value("buntinglabs-qgis-plugin/api_key", "demo").strip(),
            'x-clear-chunk-cache': str(self.clear_chunk_cache).lower(),
            'x-qgis-version': Qgis.QGIS_VERSION,
            'x-plugin-version': self.tracing_tool.plugin.plugin_version,
            'x-vertex-px-added': str(self.vertex_px_added)
        }
        if self.tracing_tool.fly_instance_id:
            headers['fly-force-instance-id'] = self.tracing_tool.fly_instance_id

        self.setProgress(10.0)

        try:
            url = QUrl("https://qgis-api.buntinglabs.com/chunk/v2")
            request = QNetworkRequest(url)
            for key, value in headers.items():
                request.setRawHeader(key.encode(), value.encode())

            nam = QgsNetworkAccessManager.instance()
            reply = nam.post(request, body)

            # Use an additional timer to update the progress of the
            # progress bar for this task.
            pb_timer = QTimer()
            pb_timer.setInterval(200) # don't do it too often
            def update_progress():
                elapsed_time = time.time() - self.start_time
                expected_duration = self.expected_time / 1000  # Convert ms to seconds
                # Some foo-y math to make the progress bar look like it's filling up
                progress = min(80.0, 10.0 + 70.0 * (elapsed_time / expected_duration))
                if elapsed_time > expected_duration:
                    remaining = 100.0 - progress
                    progress += remaining * (1 - math.exp(-0.5 * (elapsed_time - expected_duration)))
                self.setProgress(progress)

                # First server boot can take a second: show a helpful message
                if elapsed_time > 3.0 and not any(self.tracing_tool.chunk_cache.values()) and not self.hasNotifiedOfBoot:
                    self.messageReceived.emit(("Server is probably booting, please give it ~10 seconds!", Qgis.Info, None, None, 10))
                    self.hasNotifiedOfBoot = True

            pb_timer.timeout.connect(update_progress)
            pb_timer.start()

            loop = QEventLoop()
            reply.finished.connect(loop.quit)
            loop.exec_()

            pb_timer.stop()

            # Check for 'x-clear-chunks' header
            if reply.rawHeader(b'x-clear-chunks') == b'yes':
                self.clearCache.emit()

            if reply.error() != QNetworkReply.NoError:
                error_payload = reply.readAll().data()
                if reply.rawHeader(b'Content-Encoding') == b'gzip':
                    error_payload = gzip.decompress(error_payload).decode('utf-8')
                else:
                    error_payload = error_payload.decode('utf-8')
                try:
                    error_details = json.loads(error_payload)
                    self.messageReceived.emit((
                        error_details.get('message'),
                        Qgis.Critical,
                        error_details.get('link'),
                        error_details.get('link_text'),
                        15
                    ))
                except json.JSONDecodeError:
                    self.messageReceived.emit((error_payload, Qgis.Critical, None, None, 15))
                return False

            self.setProgress(80.0)

            if self.should_solve:
                content = reply.readAll().data()
                if reply.rawHeader(b'Content-Encoding') == b'gzip':
                    buffer = gzip.decompress(content).decode('utf-8')
                else:
                    buffer = content.decode('utf-8')

                data = json.loads(buffer)

                pts_cost = data['costs']
                pts_paths = data['paths']
                x_min, y_min, dxdy = data['x_min'], data['y_min'], data['dxdy']
                y_max = data['y_max']
                img_height, img_width = data['img_height'], data['img_width']
                opt_points = data['opt_points']

                # Check if the last vertex has changed since this task started
                if self.last_vertex != self.tracing_tool.vertices[-1]:
                    return False

                self.graphConstructed.emit((pts_cost, pts_paths, (x_min, y_min, dxdy, y_max), (img_height, img_width), data['included_chunks'], opt_points, data['trajectory_root'], self.cur_uuid))
                self.metadataReceived.emit((data['chunks_today'], data['chunks_left_today'], data['pricing_tier'], data['fly_instance_id'] if 'fly_instance_id' in data else None))

        except Exception as e:
            self.messageReceived.emit((f'Error connecting to autocomplete server: {str(e)}', Qgis.Critical, None, None, 15))
            return False

        # Update expected time
        ETs = self.tracing_tool.plugin.expected_time
        if len(self.chunks) in ETs:
            ETs[len(self.chunks)] = 0.8 * ETs[len(self.chunks)] + 0.2 * (time.time() - self.start_time)*1000

        return True

    def finished(self, result):
        pass

    def cancel(self):
        super().cancel()

def georeference_img_to_tiff(img_np, crs_wkt, x_min, y_min, x_max, y_max):
    # Open the PNG file
    (rasterYSize, rasterXSize, rasterCount) = img_np.shape

    # Surely this will prevent collisions
    random_hex = ''.join([random.choice('0123456789abcdef') for _ in range(16)])
    vsimem_path = f'/vsimem/bunting_qgis_tracer_{random_hex}.tif'

    # Create a new GeoTIFF file in memory
    dst = gdal.GetDriverByName('GTiff').Create(vsimem_path, rasterXSize, rasterYSize, rasterCount,
                                               gdal.GDT_Byte, options=["COMPRESS=JPEG", "JPEG_QUALITY=85"])

    # Set the geotransform
    geotransform = [x_min, (x_max-x_min)/rasterXSize, 0, y_min, 0, (y_max-y_min)/rasterYSize]
    dst.SetGeoTransform(geotransform)

    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    dst.SetProjection(srs.ExportToWkt())

    # Write the array data to the raster bands
    for b in range(rasterCount):
        band = dst.GetRasterBand(b + 1)
        band.WriteArray(img_np[:, :, b])

    # Close the files
    dst = None

    # Return the GeoTIFF-encoded memory contents as a byte array
    f = gdal.VSIFOpenL(vsimem_path, 'rb')
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
    gdal.Unlink(vsimem_path)

    return data
