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
object_img = get_object(lines_img, masks[0], show_steps=True, dil_iter=50)

# Fit the step function for each column in the image
data = object_img[:, 10]


# # Now for each column in the image we want to find the intensity profile
# # We do this by fitting a step function to the profile
# # If the fit was possible we can calculate the PSD of the profile
# fit = np.zeros_like(object_img)
# left_edge = np.zeros(object_img.shape[1])
# right_edge = np.zeros(object_img.shape[1])

# for col in range(object_img.shape[1]):
#     profile = object_img[:, col].transpose()
#     # If an error occurs during fitting plot the profile
#     try:
#         fitted_data, left, right = fit_block_step(profile, show_steps=False, invert_step=True)
#     except (OptimizeWarning, RuntimeError) as e:
#         # plt.plot(profile)
#         # plt.show()
#         continue

#     fit[:, col] = fitted_data.transpose()
#     left_edge[col] = left
#     right_edge[col] = right

# # Make some nice plots for the edges (seperate plots)
# fig = plt.figure(1, 2)
# fig.suptitle("Edge detection")
# plt.subplot(1, 2, 1)
# plt.plot(left_edge)
# plt.title("Left edge")
# plt.subplot(1, 2, 2)
# plt.plot(right_edge)
# plt.title("Right edge")
# plt.show()


# show_images({"test": objects[0]})
# left_edge, right_edge = find_object_edges(img=objects[0], show_steps=True)

# Fit the step function
# calculate_profile_psd(sums, show_steps=False, dx=1)


# function getCropCoordinates(angleInRadians, imageDimensions) {
#     var ang = angleInRadians;
#     var img = imageDimensions;

#     var quadrant = Math.floor(ang / (Math.PI / 2)) & 3;
#     var sign_alpha = (quadrant & 1) === 0 ? ang : Math.PI - ang;
#     var alpha = (sign_alpha % Math.PI + Math.PI) % Math.PI;

#     var bb = {
#         w: img.w * Math.cos(alpha) + img.h * Math.sin(alpha),
#         h: img.w * Math.sin(alpha) + img.h * Math.cos(alpha)
#     };

#     var gamma = img.w < img.h ? Math.atan2(bb.w, bb.h) : Math.atan2(bb.h, bb.w);

#     var delta = Math.PI - alpha - gamma;

#     var length = img.w < img.h ? img.h : img.w;
#     var d = length * Math.cos(alpha);
#     var a = d * Math.sin(alpha) / Math.sin(delta);

#     var y = a * Math.cos(gamma);
#     var x = y * Math.tan(gamma);

#     return {
#         x: x,
#         y: y,
#         w: bb.w - 2 * x,
#         h: bb.h - 2 * y
#     };
# }
