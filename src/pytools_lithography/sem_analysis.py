from pytools_image_processing.morphologic import (
    normalize_image,
)
from pytools_image_processing.analysis import (
    find_components,
)
from pytools_image_processing.utils import show_images, crop_image, get_bounding_rect
from pytools_image_processing.filtering import threshold, adaptive_threshold
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning


def separate_objects(
    image: np.ndarray,
    normalize: bool = True,
    mask_range: tuple[float] = (0.05, 0.8),
    dilate_iterations: int = 30,
    show_steps: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Separate objects in a SEM image.

    This function takes a grayscale image and returns a list of images
    with the separated objects.

    Parameters
    ----------
    image : np.ndarray
        The grayscale image.
    normalize : bool, optional
        If True, normalize the intensity of the image. The default is True.
    correct_rotation : bool, optional
        If True, correct the rotation of the image. The default is True.
    mask_range : tuple[float], optional
        The range of the mask size in percentage of the image size.
        The default is (0.05, 0.5) meaning the mask should be between
        5% and 50% of the image size.
    dilate_iterations : int, optional
        The number of iterations to dilate the mask. The default is 30.
        The mask is dilated to make sure the whole object is included in the
        cropped image.
    show_steps : bool, optional
        If True, show the steps of the conversion. The default is False.

    Returns
    -------
    tuple[list[np.ndarray], list[float]]
        The separated object masks and the angles of the objects.
        The masks are the components found in the image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "image should be a grayscale image not shape {}".format(image.shape)
        )
    if not isinstance(normalize, bool):
        raise ValueError(
            "normalize should be a boolean not type {}".format(type(normalize))
        )
    if not isinstance(mask_range, (tuple, list)):
        raise ValueError(
            "mask_range should be a tuple not type {}".format(type(mask_range))
        )
    if not len(mask_range) == 2:
        raise ValueError(
            "mask_range should be a tuple of length 2 not {}".format(len(mask_range))
        )
    if not isinstance(mask_range[0], (int, float)) or not isinstance(
        mask_range[1], (int, float)
    ):
        raise ValueError(
            "mask_range should be a tuple of integers or floats not {}, {}".format(
                type(mask_range[0]), type(mask_range[1])
            )
        )
    if mask_range[0] < 0 or mask_range[0] > mask_range[1]:
        raise ValueError(
            "mask_range should be a tuple of two positive numbers "
            "where the first is smaller than the second, so not {} and {}".format(
                mask_range[0], mask_range[1]
            )
        )
    if not isinstance(dilate_iterations, int):
        raise ValueError(
            "dilate_iterations should be an integer not type {}".format(
                type(dilate_iterations)
            )
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Normalize the image to increase the contrast
    if normalize:
        norm_image = normalize_image(image)
    else:
        norm_image = image

    # Remove some noise and use a threshold to get the line positions
    norm_image = cv2.GaussianBlur(norm_image, (3, 3), 0)
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # To find the contours we need to define a minimum and maximum size
    # otherwise every pixel cluster will be considered a component
    surface = np.prod(image.shape)
    min_size = int(surface * mask_range[0])
    max_size = int(surface * mask_range[1])

    # Find components
    components, component_img = find_components(
        image=mask,
        min_size=min_size,
        max_size=max_size,
        show_steps=False,
        crop_components=False,
    )
    if len(components) == 0:
        raise ValueError("No components found in the mask!")

    # Now we have the components we can use them to separate the objects
    # in the original image
    angles = []  # Store the angles of the objects
    for component in components:
        # Dilate the mask a bit to make sure we get the whole object
        dil_component = cv2.dilate(
            component, np.ones((3, 3), np.uint8), iterations=dilate_iterations
        )
        coords = get_bounding_rect(dil_component)
        angles.append(coords[-1])

    if show_steps:
        images = {
            "Component {}, a={}".format(i, angles[i]): component
            for i, component in enumerate(components)
        }
        show_images(images)

    return components, angles


import cv2
import numpy as np


def get_object(
    image: np.ndarray, mask: np.ndarray, dil_iter: int = 10,
    show_steps: bool = False
) -> np.ndarray:
    """Get the object from the image using the mask and angle.

    Parameters
    ----------
    image : np.ndarray
        The image with the object.
    mask : np.ndarray
        The mask of the object.
    dil_iter: int, optional
        The number of iterations to dilate the mask. The default is 10.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    np.ndarray
        The object from the image.
    """
    # Use the bounding box of the mask to get the center of rotation
    _, (x, y, w, h, angle) = get_bounding_rect(mask)
    center_point = (x + w // 2, y + h // 2)

    # Rotate the image and the mask
    rot_matrix = cv2.getRotationMatrix2D(center=center_point, angle=angle, scale=1)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (image.shape[1], image.shape[0]))

    # Dilate the mask to make sure the whole object is included
    dilated_mask = cv2.dilate(rotated_mask, np.ones((3, 3), np.uint8), iterations=dil_iter)

    # Get the bounding box of the dilated mask
    rect = cv2.boundingRect(dilated_mask)
    x, y, w, h = rect

    # Crop the image
    cropped_image = rotated_image[y : y + h, x : x + w]

    # Create a mask that only detects black pixels
    mask = cv2.inRange(cropped_image, 0, 0)
    # Find the contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Each of the objects should touch one of the image borders
    # If not the object is not the object we are looking for
    img_x, img_y = cropped_image.shape
    if len(contours) != 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x == 0:
                # Touches the left border so crop from the left
                cropped_image = cropped_image[:, w:]
            elif y == 0:
                # Touches the top border so crop from the top
                cropped_image = cropped_image[h:, :]
            
            if x + w == img_x:
                # Touches the right border so crop from the right
                cropped_image = cropped_image[:, :w]
            elif y + h == img_y:
                # Touches the bottom border so crop from the bottom
                cropped_image = cropped_image[:h, :]

            # # Check in wich corner the object is
            # if x == 0 and y == 0:
            #     # Top left corner
            #     cropped_image = cropped_image[h:, w:]
            # elif x == 0 and y + h == img_y:
            #     # Bottom left corner
            #     cropped_image = cropped_image[:h, w:]
            # elif x + w == img_x and y == 0:
            #     # Top right corner
            #     cropped_image = cropped_image[h:, :w]
            # elif x + w == img_x and y + h == img_y:
            #     # Bottom right corner
            #     cropped_image = cropped_image[:h, :w]
            else:
                continue
        
    if show_steps:
        images = {
            "Original image": image,
            "Rotated mask": rotated_mask,
            "Dilated mask": dilated_mask,
            "Cropped image": cropped_image,
        }
        show_images(images)

    return cropped_image


def condense_line(img: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """Condense the image to one line

    Use Riemann integration (summing pixels over the vertical axis) to
    condense the image to one line. The vertical axis is the longest
    axis in the image

    .. attention::

        The function expects a rotation corrected image. If the image is
        not corrected weird results will be returned.

    Parameters
    ----------
    img : np.ndarray
        The image to condense.

    Returns
    -------
    np.ndarray
        The condensed image.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img should be a numpy array not type {}".format(type(img)))

    # Make sure the largest axis is on the horizontal otherwise transpose
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    # Calculate the riemann sum of the image over the horizontal
    # axis
    sums = -1 * np.sum(img, axis=0)

    # Normalize the sums
    sums = (sums - np.min(sums)) / (np.max(sums) - np.min(sums))

    if show_steps:
        # Plot the image and the condensed line
        plt.subplot(121)
        plt.imshow(img, cmap="gray")
        plt.title("Image")
        plt.subplot(122)
        plt.plot(sums)
        plt.title("Condensed line")
        plt.show()

    return sums


def calculate_profile_psd(
    profile: np.ndarray, dx: float, use_window: bool = True, show_steps: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the Power Line Edge Roughness (LER) PSD.

    Calculate the Power Spectral Density (PSD) of the Line Edge Roughness (LER)
    of a line profile. The profile should be a condensed line profile of the
    line edge. The dx is the distance between the pixels in the profile.

    Parameters
    ----------
    profile : np.ndarray
        The condensed line profile.
    dx : float
        The distance between the pixels in the profile in nm.
    use_window : bool, optional
        If True, use a window function to reduce the edge effects. The default is True.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The PSD of the profile and the frequency.
    """
    if not isinstance(profile, np.ndarray):
        raise ValueError(
            "profile should be a numpy array not type {}".format(type(profile))
        )
    if not isinstance(dx, (int, float)):
        raise ValueError("dx should be a number not type {}".format(type(dx)))
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Calculate the FFT of the profile
    fft = np.fft.fft(profile)
    freq = np.fft.fftfreq(len(profile), dx)

    # Calculate the PSD
    psd = np.abs(fft) ** 2

    # Use a window function to reduce edge effects
    if use_window:
        window = np.hanning(len(profile))
        psd = psd * window

    # Make the spectrum one sided
    psd = psd[: len(psd) // 2]
    freq = freq[: len(freq) // 2]

    # The ler is the square root of the integral of the PSD
    # over the frequency range
    ler = np.sqrt(np.trapz(psd, freq))

    if show_steps:
        plt.plot(freq, psd)
        plt.title("LER PSD")
        plt.xlabel("Wave number (1/nm)")
        plt.ylabel("PSD")

        # Use loglog scale
        plt.xscale("log")
        plt.yscale("log")
        plt.show()

    return psd, freq, ler


# def fit_gaus_step(
#     data: np.ndarray,
#     s: float = None,
#     dx: float = None,
#     bl: float = None,
#     br: float = None,
#     mu: float = None,
#     sigma_l: float = None,
#     sigma_r: float = None,
#     show_steps: bool = False,
# ) -> tuple[np.ndarray, float, float, float, float, float]:
#     """Fit the double gaussian function to the steps.

#     The function is defined as 2 equations:

#     P(x) = bl + (1 - bl) * np.exp(-0.5 * (x - mu)**2 / sigma_l**2) v x < mu
#     P(x) = br + (1 - br) * np.exp(-0.5 * (x - mu)**2 / sigma_r**2) v x >= mu

#     Where the function becomes a normal gaussian curve when bl and br approach zero.

#     To make the fitting more versatile the function is defined with a shift s and a dx
#     These are used as s * P(x - dx).

#     Parameters
#     ----------
#     data: np.ndarray
#         The data to fit.
#     s: float, optional
#         The initial guess height shift of the step. The default is None.
#     bl: float, optional
#         The initial guess baseline on the left side of the step. The default is None.
#     br: float, optional
#         The initial guess baseline on the right side of the step. The default is None.
#     mu: float, optional
#         The initial guess center of the step. The default is None.
#     sigma_l: float, optional
#         The initial guess standard deviation of the left side of the step. The default is None.
#     sigma_r: float, optional
#         The initial guess standard deviation of the right side of the step. The default is None.
#     show_steps: bool, optional
#         If True, show the steps of the fitting. The default is False.

#     Returns
#     -------
#     tuple[np.ndarray, float,float,float,float,float]
#         The fitted data and the parameters of the fit.
#     """
#     if not isinstance(data, np.ndarray):
#         raise ValueError("data should be a numpy array not type {}".format(type(data)))
#     if not isinstance(s, (int, float, type(None))):
#         raise ValueError("s should be a number not type {}".format(type(s)))
#     if not isinstance(dx, (int, float, type(None))):
#         raise ValueError("dx should be a number not type {}".format(type(dx)))
#     if not isinstance(bl, (int, float, type(None))):
#         raise ValueError("bl should be a number not type {}".format(type(bl)))
#     if not isinstance(br, (int, float, type(None))):
#         raise ValueError("br should be a number not type {}".format(type(br)))
#     if not isinstance(mu, (int, float, type(None))):
#         raise ValueError("mu should be a number not type {}".format(type(mu)))
#     if not isinstance(sigma_l, (int, float, type(None))):
#         raise ValueError("sigma_l should be a number not type {}".format(type(sigma_l)))
#     if not isinstance(sigma_r, (int, float, type(None))):
#         raise ValueError("sigma_r should be a number not type {}".format(type(sigma_r)))

#     # If no initial guess is given use some default values
#     if s is None:
#         s = np.max(data)
#     if dx is None:
#         dx = 0
#     if bl is None:
#         bl = 0.1
#     if br is None:
#         br = 0.1
#     if mu is None:
#         mu = len(data) // 2
#     if sigma_l is None:
#         sigma_l = 1
#     if sigma_r is None:
#         sigma_r = 1

#     def left_gaussian(x, bl, mu, sigma_l):
#         return bl + (1 - bl) * np.exp(-0.5 * (x - mu) ** 2 / sigma_l**2)

#     def right_gaussian(x, br, mu, sigma_r):
#         return br + (1 - br) * np.exp(-0.5 * (x - mu) ** 2 / sigma_r**2)

#     # Fit the curves seperately
#     popt_l, _ = curve_fit(left_gaussian, np.arange(mu), data[:mu], p0=[bl, mu, sigma_l])
#     popt_r, _ = curve_fit(
#         right_gaussian, np.arange(mu, len(data)), data[mu:], p0=[br, mu, sigma_r]
#     )

#     # Combine the curves
#     fit_l = left_gaussian(np.arange(mu), *popt_l)
#     fit_r = right_gaussian(np.arange(mu, len(data)), *popt_r)
#     fit = np.concatenate((fit_l, fit_r))

#     if show_steps:
#         plt.plot(data, label="Data")
#         plt.plot(fit, label="Fit")
#         plt.legend()
#         plt.show()

#     return fit, *popt_l, *popt_r


def fit_block_step(
    data: np.ndarray, invert_step: bool = False, show_steps: bool = False
) -> tuple[np.ndarray, float, float]:
    """Fit a block step function to the data.

    Parameters
    ----------
    data : np.ndarray
        The data to fit.
    invert_step : bool, optional
        If True, use an inverted step response, this means the step will
        go from 1 to 0. The default is False.
    show_steps : bool, optional
        If True, show the steps of the fitting. The default is False.

    Returns
    -------
    tuple[np.ndarray, float, float]
        The fitted data and the half height on the left and right side.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data should be a numpy array not type {}".format(type(data)))
    if not isinstance(invert_step, bool):
        raise ValueError(
            "invert_step should be a boolean not type {}".format(type(invert_step))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    def step_up(x, a, b, c):
        """Equation representing a step up using htan."""
        return 0.5 * a * (np.tanh((x - b) / c) + 1)

    def step_down(x, a, b, c):
        """Equation representing a step down using htan."""
        return 0.5 * a * (np.tanh((x - b) / c) - 1)

    # Fit the curves seperately
    if invert_step:
        # Normalize the data between 0 and 1 to make fitting easier
        data = 1- (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        
    
    popt_left, cov_left = curve_fit(step_up, np.arange(len(data)), data, p0=[1, 10, 1])
    popt_right, cov_right = curve_fit(step_down, np.arange(len(data)), data, p0=[1, 10, 1])

    # Check if the fit is acceptable
    if not np.all(np.isfinite(cov_left)) or not np.all(np.isfinite(cov_right)):
        e = OptimizeWarning("The fit is not correct, the covariance matrix contains infinite values")
        print(e)
        raise e
    
    # Fit the curves
    fit_left = step_up(np.arange(len(data)), *popt_left)
    fit_right = step_down(np.arange(len(data)), *popt_right)
    fit = fit_left + fit_right

    # Determine half the height on the left and right side
    left_min = fit_left.min()
    left_max = fit_left.max()
    half_height = (left_max - left_min) / 2
    
    # Find the index with the value closest to half height
    h_left = np.argmin(np.abs(fit_left - half_height))

    right_min = fit_right.min()
    right_max = fit_right.max()
    half_height = (right_max - right_min) / 2
    
    # Find the index with the value closest to half height
    h_right = np.argmin(np.abs(fit_right - half_height))

    if h_right == h_left:
        e = OptimizeWarning("The fit is not correct, left half height position is the same as the right side")
        print(e)
        raise e
    if h_left < int(0.025 * len(data)):
        e =  OptimizeWarning("The fit is not correct, left half height position is at the start of the data")
        print(e)
        raise e
    if h_right == len(data) - int(0.975 * len(data)):
        e = OptimizeWarning("The fit is not correct, right half height position is at the end of the data")
        print(e)
        raise e

    # Normalize the fit
    fit = (fit - np.min(fit)) / (np.max(fit) - np.min(fit))

    if show_steps or h_left == 0:
        print(left_min, left_max, h_left)
        plt.plot(data, label="Data")
        plt.plot(fit, label="Fit")
        plt.legend()
        plt.show()

    return fit, h_left, h_right



def extract_profiles(
    image: np.ndarray, accepted_failure: float = 0.20, show_steps: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the intensity profiles for each column in the image.

    Parameters
    ----------
    image : np.ndarray
        The image to extract the profiles from.
    accepted_failure : float, optional
        The maximum failure rate for the fitting function, by default 0.20.
    show_steps : bool, optional
        If the fitting steps should be shown, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The top edge, bottom edge and width of profile for the object.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "The image must be a numpy array not type {}".format(type(image))
        )
    if not isinstance(accepted_failure, float):
        raise TypeError(
            "The accepted failure must be a float not type {}".format(
                type(accepted_failure)
            )
        )
    if not isinstance(show_steps, bool):
        raise TypeError(
            "The show steps must be a boolean not type {}".format(type(show_steps))
        )

    # Fit the step function
    left, right = [], []
    error_count = 0

    # Make a fit for each column
    for i in range(0, image.shape[1], 1):
        profile = image[:, i].transpose()
        try:
            _, l, r = fit_block_step(profile, show_steps=False, invert_step=True)
        except OptimizeWarning:
            error_count += 1
            continue
        left.append(l)
        right.append(r)

    # Check if the failure rate was acceptable
    if error_count > accepted_failure * object_img.shape[1]:
        print("Too many errors occured during fitting. Aborting.")
        exit()

    left_array = np.array(left)
    right_array = np.array(right)
    width_array = right_array - left_array

    if show_steps:
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(left_array)
        ax[0].set_title("Left edge")
        ax[1].plot(right_array)
        ax[1].set_title("Right edge")
        ax[2].plot(width_array)
        ax[2].set_title("Width")
        plt.show()

    return left_array, right_array, width_array


def find_object_edges(
    img: np.ndarray, show_steps: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Find the edges of the lithography line in a SEM image.

    The function expects only one line to be given in the image. The function
    will return the left and right edge of the line. To find the edges a block
    step is fitted to each column of the image. The left and right edge are
    determined by the half height of the step.

    Parameters
    ----------
    img : np.ndarray
        The image with the lithography line.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The left and right edge of the line.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img should be a numpy array not type {}".format(type(img)))
    if not len(img.shape) == 2:
        raise ValueError(
            "img should be a grayscale image not shape {}".format(img.shape)
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )
    if img.shape[0] < 10:
        raise ValueError("img should have at least 10 rows not {}".format(img.shape[0]))
    if img.shape[1] < 10:
        raise ValueError(
            "img should have at least 10 columns not {}".format(img.shape[1])
        )
    # Make sure the longest axis is on the horizontal
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)

    # Create empty edge arrays
    left_edge = np.zeros(img.shape[0])
    right_edge = np.zeros(img.shape[0])

    print(img.shape)

    for i in range(img.shape[0]):
        col = img[i, :]
        _, left, right = fit_block_step(col, show_steps=True)
        left_edge[i] = left
        right_edge[i] = right

    if show_steps:
        # Draw the edges in the image
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in range(img.shape[0]):
            img[i, int(left_edge[i])] = 255
            img[i, int(right_edge[i])] = 255
        plt.imshow(img)

        # # Also show the graphs
        # plt.subplot(121)
        # plt.plot(left_edge)
        # plt.title("Left edge")
        # plt.subplot(122)
        # plt.plot(right_edge)
        # plt.title("Right edge")
        plt.show()

    return left_edge, right_edge