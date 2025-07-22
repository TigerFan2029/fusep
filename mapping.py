import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# Define paths
path_data_path = '/Users/tiger/Desktop/FUSEP/165_path.csv'  # Path to your 165_path.csv file
layer_stats_folder = '/Users/tiger/Desktop/FUSEP/predictions_bs100_moredata_12120_245_10_60/'  # Folder containing your layer stats files

# Load the path data (contains radar path information)
path_df = pd.read_csv(path_data_path)

# Ensure 'ProductId' is of the same type (string)
path_df['ProductId'] = path_df['ProductId'].astype(str)

# Prepare a DataFrame with combined lat, lon, and depth (from path_df and layer_stats)
combined_data = []

# Loop through each path and process the corresponding layer stats file
for _, row in path_df.iterrows():
    path_name = row['ProductId']
    path_name = path_name.lstrip('s_').rstrip('_rgram')
    path_lat = row['StartLat']
    path_lon = row['StartLon']
    
    # Construct the full file path for the layer stats file
    layer_stats_file = f"{layer_stats_folder}/{path_name}_layer_stats.csv"
    
    if os.path.exists(layer_stats_file):
        # Load the layer stats file for the current radar path
        layer_stats_df = pd.read_csv(layer_stats_file)
        
        # Add the path info and depth data to the combined data
        for _, block in layer_stats_df.iterrows():
            combined_data.append({
                'Longitude': path_lon,
                'Latitude': path_lat,
                'Depth': block['distance'],
                'Path_Name': path_name
            })

# Convert combined data into a DataFrame
combined_df = pd.DataFrame(combined_data)

# Debug: Print column names and first few rows to verify
print(f"Columns in combined_df: {combined_df.columns}")
print(combined_df.head())

# Create a GeoDataFrame with geometry from the lat, lon coordinates
geometry = [Point(xy) for xy in zip(combined_df['Longitude'], combined_df['Latitude'])]
gdf = gpd.GeoDataFrame(combined_df, geometry=geometry)

# Set the CRS (coordinate reference system) to WGS84 (lat/lon)
gdf.set_crs("EPSG:4326", inplace=True)

# Save the GeoDataFrame as a shapefile
output_shapefile_path = '/Users/tiger/Desktop/FUSEP/depth_data_with_lat_lon.shp'
gdf.to_file(output_shapefile_path)

print(f"Shapefile created at: {output_shapefile_path}")
