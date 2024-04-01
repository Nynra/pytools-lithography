from pytools_image_processing.utils import (
    load_image,
    show_images,
    rotate_image,
    crop_image,
)
from pytools_image_processing.conversions import rgb_to_grayscale
from pytools_image_processing.analysis import plot_intensity_profile, get_rgb_histogram
from pytools_lithography.sem_analysis import (
    separate_objects,
    get_object,
    fit_block_step,
    extract_profiles,
    calculate_profile_psd,
)
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit


# Load images
lines_path = os.path.join(os.path.dirname(__file__), "test_lines_cropped.jpg")
size_bar_path = os.path.join(os.path.dirname(__file__), "test_size_bar.jpg")

lines_img = load_image(lines_path)
size_bar_img = load_image(size_bar_path)

# Convert the images to grayscale and normalize the intensity
lines_img = cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY)

# Get the seperate objects
masks, angles = separate_objects(lines_img, show_steps=False)
object_img = get_object(lines_img, masks[0], show_steps=False, dil_iter=50)

# Fit the step function for each column in the image
top, bottom, width = extract_profiles(object_img, show_steps=True, accepted_failure=0.50)

# Now we can calculate the PSD
top_psd, top_freqs = calculate_profile_psd(top, show_steps=False, dx=1)

