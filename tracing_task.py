# Copyright 2023 Bunting Labs, Inc.

import tempfile
import os
import http.client
import json
from osgeo import gdal, osr

from qgis.core import QgsTask, QgsMapSettings, QgsMapRendererCustomPainterJob, \
    QgsCoordinateTransform, QgsProject, QgsRectangle
from qgis.PyQt.QtGui import QImage, QPainter, QColor
from qgis.PyQt.QtCore import QSize, pyqtSignal

class AutocompleteTask(QgsTask):
    # This task can run in the background of QGIS, streaming results
    # back from the inference server.

    pointReceived = pyqtSignal(tuple)
    errorReceived = pyqtSignal(str)

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
        # Initialize resolution variables
        x_res = float('-inf')
        y_res = float('-inf')

        # The resolution of a raster layer is defined as the ground distance covered by one pixel
        # of the raster. Therefore, a smaller resolution value means a higher resolution raster.
        mapEpsgCode = self.project_crs.postgisSrid()

        # Assuming self.rlayers is a list of QgsRasterLayer objects
        primaryrlayer = None
        for rlayer in self.rlayers:
            # Get the extent of the raster layer
            raster_extent = rlayer.extent()

            # Create a QGIS point XY from the last but one point in the path
            point = self.tracing_tool.vertices[-1]

            # Convert point from self.project_crs to rlayer's crs
            point_transform = QgsCoordinateTransform(self.project_crs, rlayer.crs(), QgsProject.instance())
            point = point_transform.transform(point)

            if raster_extent.contains(point):
                # Get the resolution of the raster layer
                x_res_temp = rlayer.rasterUnitsPerPixelX()
                y_res_temp = rlayer.rasterUnitsPerPixelY()

                # Update x_res and y_res if they are None or larger than the current layer's resolution
                if x_res_temp > x_res or y_res_temp > y_res:
                    x_res = x_res_temp
                    y_res = y_res_temp
                    primaryrlayer = rlayer

                print(f"Raster layer {rlayer.name()} intersects with the rectangle.")
                print(f"Resolution in X direction: {x_res}")
                print(f"Resolution in Y direction: {y_res}")

        # Size of the rectangle in the CRS coordinates
        # 1200 is the number of pixels we want to render
        img_width, img_height = 1200, 1200
        x_size = img_width * x_res
        y_size = img_height * y_res

        if x_size <= 0 or y_size <= 0:
            raise ValueError
        assert primaryrlayer is not None

        # i = y, j = x
        # note that negative i (or y) is up
        x0, y0 = self.tracing_tool.vertices[-2]
        x1, y1 = self.tracing_tool.vertices[-1]
        cx, cy = (x0+x1)/2, (y0+y1)/2

        x_min = cx - x_size / 2
        x_max = cx + x_size / 2
        y_min = cy - y_size / 2
        y_max = cy + y_size / 2

        # create image
        img = QImage(QSize(img_width, img_height), QImage.Format_ARGB32_Premultiplied)

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

        # Create temporary files for png and tif images
        with tempfile.NamedTemporaryFile(suffix=".png") as png_temp, \
             tempfile.NamedTemporaryFile(suffix=".tif") as tif_temp:

            # Save the image to the temporary png file
            img.save(png_temp.name, "png")

            # Print png_temp.name's size and filename
            print(f'PNG file size: {os.path.getsize(png_temp.name)} bytes')
            print(f'PNG file name: {png_temp.name}')

            # Call the function to convert the png to tif and save it to the temporary tif file
            georeference_png_to_tiff(png_temp.name, tif_temp.name, mapEpsgCode, x_min, y_max, x_max, y_min)

            # Prepare the image payload
            with open(tif_temp.name, 'rb') as f:
                img_data = f.read()

        i_min, j_min = convert_coords_to_indxs(primaryrlayer, (x_min, y_max))
        i0, j0 = convert_coords_to_indxs(primaryrlayer, (x0, y0))
        i1, j1 = convert_coords_to_indxs(primaryrlayer, (x1, y1))

        vector_payload = json.dumps({
            'coordinates': [[i0-i_min, j0-j_min], [i1-i_min, j1-j_min]]
        })

        options_payload = json.dumps({
            'num_completions': self.tracing_tool.num_completions
        })

        boundary = 'wL36Yn8afVp8Ag7AmP8qZ0SA4n1v9T'
        body = (
            '--' + boundary,
            'Content-Disposition: form-data; name="image"; filename="rendered.tif"',
            'Content-Type: application/octet-stream',
            '',
            img_data,
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
                self.errorReceived.emit(res.read().decode('utf-8'))
                return False
        except BrokenPipeError:
            self.errorReceived.emit('Got BrokenPipeError when trying to connect to inference server')
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
                xn, yn = convert_indxs_to_coords(primaryrlayer, (ix+i_min, jx+j_min))
                self.pointReceived.emit(((xn, yn), 1.0))

        return True

    def finished(self, result):
        pass

    def cancel(self):
        super().cancel()


# Convert map CRS coordinates to the pixels in the image
def convert_coords_to_indxs(rlayer, xy):
    x, y = xy
    provider = rlayer.dataProvider()
    extent = provider.extent()

    dx = rlayer.rasterUnitsPerPixelX()
    dy = rlayer.rasterUnitsPerPixelY()
    top_left_x = extent.xMinimum()
    top_left_y = extent.yMaximum()
    # geo_ref = (top_left_x, top_left_y, dx, dy)
    i = int((y - top_left_y) / dy) * -1
    j = int((x - top_left_x) / dx)
    return i, j

def convert_indxs_to_coords(rlayer, ij):
    i, j = ij
    provider = rlayer.dataProvider()
    extent = provider.extent()

    dx = rlayer.rasterUnitsPerPixelX()
    dy = rlayer.rasterUnitsPerPixelY()
    top_left_x = extent.xMinimum()
    top_left_y = extent.yMaximum()
    # geo_ref = (top_left_x, top_left_y, dx, dy)
    x = j * dx + top_left_x
    y = top_left_y - i * dy
    return x, y

def georeference_png_to_tiff(png_file, tiff_file, epsg, x_min, y_min, x_max, y_max):
    # Open the PNG file
    src = gdal.Open(png_file, gdal.GA_ReadOnly)
    src_img = src.ReadAsArray()

    # Create a new GeoTIFF file
    dst = gdal.GetDriverByName('GTiff').Create(tiff_file, src.RasterXSize, src.RasterYSize, src.RasterCount,
                                               gdal.GDT_Byte, options=["COMPRESS=JPEG"])

    # Set the geotransform
    geotransform = [x_min, (x_max-x_min)/src.RasterXSize, 0, y_min, 0, (y_max-y_min)/src.RasterYSize]
    dst.SetGeoTransform(geotransform)

    # Set the projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dst.SetProjection(srs.ExportToWkt())

    # Write the array data to the raster bands
    for b in range(src.RasterCount):
        band = dst.GetRasterBand(b + 1)
        band.WriteArray(src_img[b])

    # Close the files
    src = None
    dst = None
