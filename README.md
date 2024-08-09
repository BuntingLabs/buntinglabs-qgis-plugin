# AI Vectorizer for QGIS

This QGIS plugin uses machine learning to automatically vectorize lines and polygons from raster maps. It lets you digitize old raster maps more quickly, even for skilled GIS users.

![AI vectorizing a map in QGIS](assets/example.gif)

Use this plugin by clicking the **Vectorize with AI** icon in the QGIS Plugins Toolbar.
Begin editing a vector layer and add two vertices to a new feature to start tracing.
Hover around the map to let the AI autocomplete ahead of you. Use `shift` to manually
add vertices without autocomplete.

### Network Usage

![maps sent to server](assets/plugin_data_flow.png)

Because the plugin uses a hosted AI to autocomplete tracing maps, context around
the cursor (raster layers) is sent to Bunting Labs servers, and the vector output
is returned.

### License

This repository is licensed under the GNU GPLv2.
