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

pallete_colors = 16
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

# Get the list of files in the 'Source' directory
files = os.listdir('Source')

# Filter the list of files to only include MKV, MP4, and WEBM files
video_files = [file for file in files if file.endswith('.mkv') or file.endswith('.mp4') or file.endswith('.webm')]

# Connect to the database
conn = sqlite3.connect('video_colors.db')

# Create a cursor
cursor = conn.cursor()

# Create the table for storing results
cursor.execute('''CREATE TABLE IF NOT EXISTS video_colors (
                    video_name text,
                    video_frame integer,
                    palette text,
                    contrast text
                    )''')

def frame_procesor(video_file, frame, number):
        print("Processing frame " + str(number))
        
        # Encode the frame as a JPEG image
        _, image_data = cv2.imencode('.jpg', frame)
        
        # Create a file-like object from the image data
        image_file = BytesIO(image_data)
        
        # Use ColorThief to generate a color palette from the frame
        color_thief = ColorThief(image_file)
        try:
            palette = color_thief.get_palette(color_count=pallete_colors)
        except: 
            return (video_file, number, 'NONE', 'ERROR')
        
        # Convert the RGB tuples in the palette to hexadecimal format
        palette_hex = [rgb_to_hex(color) for color in palette]
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
        
        return (video_file, number, ','.join(palette_hex), contrast_hex)
    
    
results_queue = queue.Queue()

def on_frame_processing_complete(future):
    result = future.result()
    results_queue.put(result)

# Iterate over each video file
for video_file in video_files:
    # Open the video file
    file = 'Source/'+video_file
    reader = imageio.get_reader(file)
    
    number = 1
    print("Processing " + str(file))
    
    # Create a threadpool with 8 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks to the threadpool
        for frame in reader:
            future = executor.submit(frame_procesor, video_file, frame, number)
            
            future.add_done_callback(on_frame_processing_complete)   
            number += 1
            
            if number % 100 == 0:
                while not results_queue.empty():
                    result = results_queue.get()

                    # Store the results in the database
                    cursor.execute('''INSERT INTO video_colors (video_name, video_frame, palette, contrast)
                        VALUES (?, ?, ?, ?)''', (result[0], result[1], result[2], result[3]))           

                print("Pushed results at frame " + str(number))
                conn.commit()

        # Wait for all tasks to complete
        executor.shutdown()
        
    
    while not results_queue.empty():
        result = results_queue.get()

        # Store the results in the database
        cursor.execute('''INSERT INTO video_colors (video_name, video_frame, palette, contrast)
            VALUES (?, ?, ?, ?)''', (result[0], result[1], result[2], result[3]))
                            
    print("Pushed results at frame " + str(number))
    conn.commit()
    
    print("Completed " + str(file))
    
    # Release the video file
    reader.close()

# Commit the changes to the database
conn.commit()

# Close the connection to the database
conn.close()

def get_contrast_colors(conn):
    cursor = conn.cursor()
    cursor.execute('''SELECT contrast, COUNT(*) as count
                      FROM video_colors
                      GROUP BY contrast
                      ORDER BY count DESC''')
    return cursor.fetchall()

# Connect to the database
conn = sqlite3.connect('video_colors.db')

# Get the contrast colors
contrast_colors = get_contrast_colors(conn)

# Print the contrast colors
for contrast_color, count in contrast_colors:
    print(f'Contrast color: {contrast_color}, Count: {count}')

# Close the connection to the database
conn.close()
