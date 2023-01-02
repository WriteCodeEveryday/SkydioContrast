import sqlite3
import os
import imageio
import cv2
import concurrent.futures
import queue
from io import BytesIO
from colorthief import ColorThief
from sklearn.cluster import KMeans
from skimage.color import deltaE_ciede2000
from colormath.color_conversions import convert_color
from colormath.color_objects import XYZColor, sRGBColor, LabColor
from webcolors import CSS3_HEX_TO_NAMES, rgb_to_hex

number_of_colors = 128

# Generate a list of all possible CSS3 colors
css3_colors = list(CSS3_HEX_TO_NAMES.keys())

# Convert the colors to XYZ colors
xyz_colors = [convert_color(sRGBColor.new_from_rgb_hex(color), XYZColor) for color in css3_colors]

# Extract the numerical data from the XYZColor objects
xyz_data = [[color.xyz_x, color.xyz_y, color.xyz_z] for color in xyz_colors]

# Use the median cut algorithm to cluster the colors
kmeans = KMeans(n_clusters=number_of_colors)
cluster_indices = kmeans.fit_predict(xyz_data)

# Connect to the database
conn = sqlite3.connect('video_colors.db')

# Create a cursor
cursor = conn.cursor()

def get_frame_data(conn):
    cursor = conn.cursor()
    cursor.execute('''SELECT video_name, video_frame, palette
                      FROM video_colors
                      ORDER BY video_name, video_frame ASC''')
    return cursor.fetchall()

# Connect to the database
conn = sqlite3.connect('video_colors.db')

# Get the contrast colors
frame_data = get_frame_data(conn)

def frame_procesor(frame):
        print("Processing frame " + str(frame[1]))
        
        # Convert the RGB tuples in the palette to hexadecimal format
        palette_hex = frame[2].split(',')
        palette_rgb = [sRGBColor.new_from_rgb_hex(color) for color in palette_hex]

        # Find the color with the highest contrast against the color palette
        highest_contrast_color = None
        highest_contrast = 0

        # Iterate over each cluster and find the color with the highest contrast
        for i, cluster_index in enumerate(cluster_indices):
            # Get the XYZ color for this cluster
            color = xyz_colors[cluster_index]
            
            # Convert the XYZ colors to hexadecimal format
            color_rgb = convert_color(color, sRGBColor)
            
            # Calculate the contrast against all colos in the palette
            contrast_sum = sum([deltaE_ciede2000(color_rgb.get_value_tuple(), palette_color.get_value_tuple()) for palette_color in palette_rgb])
            num_colors = len(palette_rgb)
            contrast = contrast_sum / num_colors

            # Update the color with the highest contrast if necessary
            if contrast > highest_contrast:
                highest_contrast = contrast
                highest_contrast_color = color
     
        contrast_hex = color_rgb.get_rgb_hex()
     
        return (frame[0], frame[1], frame[2], contrast_hex)
    
    
results_queue = queue.Queue()

def on_frame_processing_complete(future):
    result = future.result()
    results_queue.put(result)

# Create a threadpool with 8 threads
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    # Iterate over each video file
    for frame in frame_data:
        # Submit tasks to the threadpool
        future = executor.submit(frame_procesor, frame)
            
        future.add_done_callback(on_frame_processing_complete)   
        if results_queue.qsize() % 100 == 0:
            while not results_queue.empty():
                result = results_queue.get()

                # Store the results in the database
                cursor.execute('''UPDATE video_colors
                   SET contrast = ?
                   WHERE video_name = ? AND video_frame = ? AND palette = ?''', (result[3], result[0], result[1], result[2]))       

            conn.commit()
            print("Pushed results")
    
    # Wait for all tasks to complete
    executor.shutdown()

while not results_queue.empty():
    result = results_queue.get()

    # Store the results in the database
    cursor.execute('''UPDATE video_colors
                   SET contrast = ?
                   WHERE video_name = ? AND video_frame = ? AND palette = ?''', (result[3], result[0], result[1], result[2]))           

conn.commit()
print("Pushed results")

# Close the connection to the database
conn.close()
