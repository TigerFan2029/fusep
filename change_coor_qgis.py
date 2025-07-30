from qgis.core import QgsField
from qgis.PyQt.QtCore import QVariant

# Get the active layer (assuming the layer is selected in QGIS)
layer = iface.activeLayer()

# Ensure the necessary fields exist (StartLat, StartLon, StopLat, StopLon)
field_names = [field.name() for field in layer.fields()]

# Only add the fields if they do not exist
if 'StartLat' not in field_names:
    layer.dataProvider().addAttributes([QgsField('StartLat', QVariant.Double)])
if 'StartLon' not in field_names:
    layer.dataProvider().addAttributes([QgsField('StartLon', QVariant.Double)])
if 'StopLat' not in field_names:
    layer.dataProvider().addAttributes([QgsField('StopLat', QVariant.Double)])
if 'StopLon' not in field_names:
    layer.dataProvider().addAttributes([QgsField('StopLon', QVariant.Double)])

# Update the fields with new coordinates
layer.updateFields()

# Start editing the layer to update the attribute table
layer.startEditing()

# Loop through each feature and update the coordinates
for feature in layer.getFeatures():
    geometry = feature.geometry()

    # Check if the geometry is a MultiLineString
    if geometry.isMultipart():
        # Extract each individual line from the MultiLineString
        for part in geometry.asMultiPolyline():
            # Process each line as a polyline
            start_point = part[0]  # First vertex (start point)
            end_point = part[-1]  # Last vertex (end point)

            # Extract latitude and longitude for the start and end points
            start_lat = start_point[1]  # Latitude of the first point
            start_lon = start_point[0]  # Longitude of the first point
            stop_lat = end_point[1]  # Latitude of the last point
            stop_lon = end_point[0]  # Longitude of the last point

            # Update the fields with the new coordinates
            layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StartLat'), start_lat)
            layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StartLon'), start_lon)
            layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StopLat'), stop_lat)
            layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StopLon'), stop_lon)

    else:
        # Handle single line geometry
        polyline = geometry.asPolyline()
        start_point = polyline[0]  # First vertex (start point)
        end_point = polyline[-1]  # Last vertex (end point)

        # Extract latitude and longitude for the start and end points
        start_lat = start_point[1]  # Latitude of the first point
        start_lon = start_point[0]  # Longitude of the first point
        stop_lat = end_point[1]  # Latitude of the last point
        stop_lon = end_point[0]  # Longitude of the last point

        # Update the fields with the new coordinates
        layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StartLat'), start_lat)
        layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StartLon'), start_lon)
        layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StopLat'), stop_lat)
        layer.changeAttributeValue(feature.id(), layer.fields().indexFromName('StopLon'), stop_lon)

# Commit the changes to the layer
layer.commitChanges()
