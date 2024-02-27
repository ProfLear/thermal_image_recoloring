#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:59:51 2024

@author: benjaminlear
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_two_closest_colors(color_list, target_color):
    """
    Find the two closest colors in the color_list to the target_color.
    Returns a tuple with the two closest colors.
    """
    color_array = np.array(color_list)
    target_array = np.array(target_color)

    distances = np.sqrt(np.sum((color_array - target_array)**2, axis=1))
    sorted_indices = np.argsort(distances)
    return color_list[sorted_indices[0]], color_list[sorted_indices[1]]

def interpolate_color(color1, color2, target_color, new_color1, new_color2):
    """
    Interpolates the target_color between color1 and color2 and maps it to the space
    between new_color1 and new_color2.
    """
    # Convert to NumPy arrays for vectorized operations
    color1 = np.array(color1)
    color2 = np.array(color2)
    target_color = np.array(target_color)
    new_color1 = np.array(new_color1)
    new_color2 = np.array(new_color2)

    # Calculate distances
    total_distance = np.linalg.norm(color2 - color1)
    target_distance = np.linalg.norm(target_color - color1)

    # Calculate relative distance
    relative_distance = target_distance / total_distance if total_distance else 0

    # Interpolate the new color
    re_color = new_color1 + (new_color2 - new_color1) * relative_distance
    return tuple(re_color.astype(int))

# Read the CSV file and generate old_bar
old_bar = []
with open("./thermalCameraMap.csv", "r") as f:
    for line in f:
        try:
            old_bar.append(tuple(map(int, line.strip().split(","))))
        except ValueError as e:
            print(f"Error parsing line '{line}': {e}")

# Generate new colors using matplotlib's colormap
num_values = len(old_bar)
new_cmap = plt.get_cmap('hot')
new_bar = [tuple(int(x * 255) for x in new_cmap(i / num_values)[:3]) for i in range(num_values)]

# Create a dictionary mapping old RGB values to new RGB values
recolor_map = {old: new for old, new in zip(old_bar, new_bar)}

# Load the original image
original_image = Image.open("./slow_6_1.jpg")

# Ensure image is in RGB mode
if original_image.mode != 'RGB':
    original_image = original_image.convert('RGB')

# Convert image to NumPy array for faster processing
image_data = np.array(original_image)

# Create a new array to store the recolored image data
#now recolor using the new_bar
new_color_data = np.zeros_like(image_data)

# Iterate over each pixel in the image
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        # Get the RGB values of the current pixel
        image_color = tuple(image_data[i, j])
        
        # now, find the color this should be in the new color map
        old_color1, old_color2 = find_two_closest_colors(list(recolor_map.keys()), image_color)
        new_color1 = recolor_map[old_color1]
        new_color2 = recolor_map[old_color2]
        re_color = interpolate_color(old_color1, old_color2, image_color, new_color1, new_color2)
        
        new_color_data[i, j] = re_color
        
# Convert the numpy array back to an Image object
new_color_image = Image.fromarray(new_color_data.astype('uint8'))
new_color_image.show()  # Uncomment to display the image


# recolor


