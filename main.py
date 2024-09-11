import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from pytools_image_processing.utils import show_images
from tests.utils import load_test_image
from pytools_lithography.analysis.image_processing import ImagePreProcessor

filename = "test_analysis_lines_cropped.jpg"
img = load_test_image(filename)

show_images({
    "Original": img
})




# # Load images
# lines_path = os.path.join(os.path.dirname(__file__), "test_lines_cropped.jpg")
# size_bar_path = os.path.join(os.path.dirname(__file__), "test_size_bar.jpg")

# lines_img = load_image(lines_path)
# size_bar_img = load_image(size_bar_path)

# # Convert the images to grayscale and normalize the intensity
# lines_img = cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY)

# # Get the seperate objects
# masks, angles = separate_objects(lines_img, show_steps=False)
# object_img = get_object(lines_img, masks[0], show_steps=False, dil_iter=50)

# # Fit the step function for each column in the image
# top, bottom, width = extract_profiles(object_img, show_steps=True, accepted_failure=0.50)

