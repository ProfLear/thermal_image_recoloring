#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:57:11 2024

@author: benjaminlear
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Generate new colors using matplotlib's colormap
num_values = 256
new_cmap = plt.get_cmap('hot')
new_bar = [tuple(int(x * 255) for x in new_cmap(i / num_values)[:3]) for i in range(num_values)]


# Load the original image
original_thermal_image = Image.open("./slow_thermal_grey.jpg")
# Convert image to NumPy array for faster processing
original_thermal_data = np.array(original_thermal_image)

# Create a new array to store the recolored image data
#now recolor using the new_bar
new_thermal_data = np.zeros_like(original_thermal_data)

# Iterate over each pixel in the image
for i in range(original_thermal_data.shape[0]):
    for j in range(original_thermal_data.shape[1]):
        # Get the RGB values of the current pixel
        current_color = original_thermal_data[i, j][0]

        new_thermal_data[i, j] = new_bar[current_color]


# Convert the numpy array back to an Image object
new_thermal_image = Image.fromarray(new_thermal_data.astype('uint8'))
#new_thermal_image.show()  # Uncomment to display the image


# Open the two images
old_visual_image = Image.open("./slow_visual.jpg")


# Blend the images together
result = Image.blend(new_thermal_image, old_visual_image, alpha=0.3)  # alpha determines the transparency of the overlay

# Save the result
result.show()

