import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic

#config
RGRAM_DIR    = "/Users/tiger/Desktop/FUSEP/rgram_full"
OUTPUT_DIR   = "/Users/tiger/Desktop/FUSEP/rgram_clipped"
PATH_FILE    = "/Users/tiger/Desktop/FUSEP/138_path.csv"
CLIPPED_PATH_FILE = "/Users/tiger/Desktop/FUSEP/138_clipped.csv"

path_df = pd.read_csv(PATH_FILE)
clipped_df = pd.read_csv(CLIPPED_PATH_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

#alongtrack distance
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def generate_distance_map(path_df):
    distances = []
    for index, row in path_df.iterrows():
        start_lat, start_lon = row['StartLat'], row['StartLon']
        end_lat, end_lon = row['StopLat'], row['StopLon']
        
        distance = calculate_distance(start_lat, start_lon, end_lat, end_lon)
        distances.append(distance)
        
    path_df['PathDistance'] = distances
    return path_df

path_df = generate_distance_map(path_df)

# crop radargram
def crop_radargram(radargram_file, start_lat, start_lon, end_lat, end_lon, path_distance_map, start_dist, stop_dist, product_id):
    path_data = path_distance_map[path_distance_map['ProductId'] == product_id]
    
    if path_data.empty:
        print(f"No matching path data for ProductId: {product_id}")
        return None

    total_distance = path_data['PathDistance'].values[0]

    data = np.loadtxt(radargram_file)
    
    num_columns = data.shape[1]
    step = total_distance / num_columns  # Distance per column
    
    # Calculate the indices for the start and stop distances
    start_idx = int(start_dist / step)
    stop_idx = int(stop_dist / step)
    
    # Crop
    cropped_data = data[:, start_idx:stop_idx]
    
    return cropped_data

for index, row in clipped_df.iterrows():
    product_id = row['ProductId']
    product_id_nos = product_id.lstrip('s_')
    start_lat = row['StartLat']
    start_lon = row['StartLon']
    end_lat = row['StopLat']
    end_lon = row['StopLon']
    start_dist = calculate_distance(start_lat, start_lon, start_lat, start_lon)
    stop_dist = calculate_distance(start_lat, start_lon, end_lat, end_lon)
    
    radargram_file = os.path.join(RGRAM_DIR, f"{product_id_nos}.txt")
    
    if os.path.exists(radargram_file):
        cropped_data = crop_radargram(radargram_file, start_lat, start_lon, end_lat, end_lon, path_df, start_dist, stop_dist, product_id)
 
        if cropped_data is not None:
            output_file = os.path.join(OUTPUT_DIR, f"{product_id_nos}.txt")
        
            if os.path.exists(output_file):
                print(f"Skipping {product_id_nos} as the file already exists.")
                continue
            np.savetxt(output_file, cropped_data)
            print(f"Processed {product_id_nos} and saved cropped radargram. Shape -> {cropped_data.shape}")
    else:
        print(f"Radargram file {radargram_file} not found.")


print("Radargram clipping completed.")
