import os

import numpy as np
import tensorflow as tf
from PIL import Image

# Load a folder of padded images and save the original images
# Restore resolution from 256x256 to 218x178 by cropping the zero-padded edges
path_input = "./output/gan/output_untrained"
path_output = "./unpadded/gan/output_untrained"

files = os.listdir(path_input)
files = [f for f in files if f.endswith(".jpg")]

# Create a folder to save the original images
os.makedirs(path_output, exist_ok=True)

# Calculate the mask for cropping the zero-padded edges
new_width = int(178 / 218 * 256)  # All images were enlarged by (100 / 218 * 256) = 17%
left_bound = int((256 - new_width) / 2)
right_bound = int((256 + new_width) / 2)


# Loop over all images
for i, file in enumerate(files):
    # Load image
    image = Image.open(os.path.join(path_input, file))
    image = np.array(image)

    # Crop the zero-padded edges to restore the original aspect ratio
    image = image[:, left_bound:right_bound, :]

    # Resize image to 218x178 to restore the original resolution
    image = tf.image.resize(image, (218, 178))
    image = image.numpy()
    image = image.astype(np.uint8)

    # Save image
    image = Image.fromarray(image)
    image.save(os.path.join(path_output, file))
