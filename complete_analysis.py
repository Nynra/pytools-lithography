import os
import cv2
import numpy as np
from pytools_image_processing.utils import show_images
from pytools_lithography.analysis import (
    get_object,
    separate_objects,
    calculate_profile_psd,
    ImagePreProcessor,
)

# Define the path to the image
filename = "Sample_B_0010.jpg"
dir = os.path.join(os.path.dirname(__file__), "dev", "Prachtige_Fotos")
path = os.path.join(dir, filename)

# Check if the folder exists
if not os.path.exists(dir):
    raise FileNotFoundError("The folder in path {} does not exist".format(dir))

# Give me a list of all the files in the folder
accepted_extensions = [".jpg", ".jpeg", ".png"]
files = os.listdir(dir)
print("\nFiles in folder:")
for file in files.copy():
    if not any([file.endswith(ext) for ext in accepted_extensions]):
        files.remove(file)
    else:
        print(file)
print("\n")

# Check if the image exists
if not os.path.exists(path):
    raise FileNotFoundError("The image does in path {} does not exist".format(path))

# Load the image
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Seperate the image into the scalebar and the rest
preprocessor = ImagePreProcessor(image)
lines_image = preprocessor.get_image()
scale, scale_error = preprocessor.get_scale()

# Show the images
# show_images({"lines": lines_image})
# print("Scale: ", scale * image.shape[0])

# Get the objects in the image
objects_masks, angles = separate_objects(lines_image, show_steps=True)
objects = [get_object(lines_image, mask, show_steps=True) for mask in objects_masks]

