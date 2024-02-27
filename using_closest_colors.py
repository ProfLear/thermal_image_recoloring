'''
V 0.1.0
This recolors an image by taking the old color scale list, and then mapping this onto a new color scale using a dictionary. 
Then, for an image, we go through pixel by pixel and for each color find the color in the old color map that is closest to the image color. 
Then, using these colors, we re-color the old image into the new color scale. 
'''

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_closest_color(rgb_list, target_rgb):
    '''
    takes a color and then finds the closest color to it in a list. 
    Parameters
    ----------
    rgb_list : TYPE
        DESCRIPTION.
    target_rgb : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Convert the list and target color to a NumPy array
    rgb_array = np.array(rgb_list)
    target_array = np.array(target_rgb)

    # Calculate the Euclidean distance between the target and each RGB color in the list
    distances = np.sqrt(np.sum((rgb_array - target_array) ** 2, axis=1))

    # Find the index of the smallest distance
    closest_index = np.argmin(distances)

    # Return the closest RGB value
    return rgb_list[closest_index]




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
closest_color_data = np.zeros_like(image_data)

# Iterate over each pixel in the image
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        # Get the RGB values of the current pixel
        current_color = tuple(image_data[i, j])

        # Find the closest color in old_bar
        closest_color = find_closest_color(old_bar, current_color)

        # Replace the pixel with the closest color
        closest_color_data[i, j] = closest_color

# Convert the numpy array back to an Image object
closest_color_image = Image.fromarray(closest_color_data.astype('uint8'))
closest_color_image.show()  # Uncomment to display the image


#now recolor using the new_bar
new_color_data = np.zeros_like(image_data)

# Iterate over each pixel in the image
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        # Get the RGB values of the current pixel
        current_color = tuple(closest_color_data[i, j])

        # Replace the pixel with the closest color
        try:
            new_color_data[i, j] = recolor_map[current_color]
        except:
            pass
# Convert the numpy array back to an Image object
new_color_image = Image.fromarray(new_color_data.astype('uint8'))
new_color_image.show()  # Uncomment to display the image
