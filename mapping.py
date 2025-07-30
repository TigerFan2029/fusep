import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import numpy as np
from geopy.distance import geodesic

path_data_path = '/Users/tiger/Desktop/FUSEP/138_clipped.csv'
layer_stats_folder = '/Users/tiger/Desktop/FUSEP/predictions/'
path_df = pd.read_csv(path_data_path)

path_df['ProductId'] = path_df['ProductId'].astype(str)

def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

combined_data = []

for _, row in path_df.iterrows():
    path_name = row['ProductId']
    path_name = path_name.lstrip('s_').rstrip('_rgram')
    path_lat = row['StartLat']
    path_lon = row['StartLon']
    path_end_lat = row['StopLat']
    path_end_lon = row['StopLon']

    total_distance = calculate_distance(path_lat, path_lon, path_end_lat, path_end_lon)
    
    layer_stats_file = f"{layer_stats_folder}/{path_name}_layer_stats.csv"
    
    if os.path.exists(layer_stats_file):
        layer_stats_df = pd.read_csv(layer_stats_file)
        
        for _, block in layer_stats_df.iterrows():
            # Get depth (distance) and block index
            depth = block['distance']
            
            # Calculate the fraction of the total distance this block represents
            block_fraction = block['block_idx'] / len(layer_stats_df)
            
            # Interpolate along the path by dividing the total distance by the number of blocks
            interp_lat = path_lat + block_fraction * (path_end_lat - path_lat)
            interp_lon = path_lon + block_fraction * (path_end_lon - path_lon)
            
            # Add the interpolated point and depth data to the combined list
            combined_data.append({
                'Longitude': interp_lon,
                'Latitude': interp_lat,
                'Depth': depth,
                'Path_Name': path_name
            })

combined_df = pd.DataFrame(combined_data)

# Print out the table of depth ranges and counts, user selected what to exclude
depth_bins = np.arange(0, combined_df['Depth'].max() + 100, 100)
depth_range_counts = pd.cut(combined_df['Depth'], bins=depth_bins).value_counts().sort_index()

print("Depth Range and Count Table:")
print(depth_range_counts)

threshold = float(input("Enter a depth threshold: "))
combined_df.loc[combined_df['Depth'] > threshold, 'Depth'] = 0

# Create a GeoDataFrame with geometry from the lat, lon coordinates
geometry = [Point(xy) for xy in zip(combined_df['Longitude'], combined_df['Latitude'])]
gdf = gpd.GeoDataFrame(combined_df, geometry=geometry)

gdf.set_crs("+proj=latlong +datum=WGS84 +no_defs +lon_0=0", inplace=True)

output_shapefile_path = '/Users/tiger/Desktop/FUSEP/138_try.shp'
gdf.to_file(output_shapefile_path)

print(f"Shapefile created at: {output_shapefile_path}")
