from pytools_image_processing.morphologic import (
    normalize_image,
)
from pytools_image_processing.analysis import (
    find_components,
)
from pytools_image_processing.utils import show_images, get_bounding_rect
import numpy as np
import cv2
import easyocr as ocr


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


def get_scale_from_bar(
    image: np.ndarray, return_all: bool = False
) -> tuple[float, float]:
    """Determine the number of nm/pixel factor from the scale bar.

    This function expects a cropped scale bar from the SEM in the MPD
    lab. This is in now way a generalized function and should not be used
    for other images. The cropped scale bar is complete bottom part of
    the image with a horizontal cut where the image begins.

    Parameters
    ----------
    image : np.ndarray
        The cropped scale bar.
    return_all : bool, optional
        If True, return all the intermediate steps. The default is False.

    Returns
    -------
    tuple[float, float]
        The scale in nm/pixel and the plus minus error.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "image should be a grayscale image not shape {}".format(image.shape)
        )

    # Create the OCR
    reader = ocr.Reader(["en"], gpu=False)

    # How big of a 'cut-out' do we make from the left corner in the up and right  direction
    SCALE_IMG_DIMENSIONS = 95

    # Crop a copy of the image to the size of the number
    scale_text_img = image.copy()[:, :SCALE_IMG_DIMENSIONS]
    # We use detail = 0 to just get the text, we dont care for the other info
    scale_str = reader.readtext(scale_text_img, detail=0)

    # Check if the scale str contains only numbers
    if not scale_str.isdigit():
        raise ValueError(
            "The scale string should only contain numbers, it now says {}".format(
                scale_str
            )
        )

    scale_int = int(scale_str)

    # Now we have the number, time to find out how long the bar is
    SCALE_BAR_START = 200
    SCALE_BAR_END = 1250
    scale_bar_img = image[:, SCALE_BAR_START:SCALE_BAR_END]

    # Use the Canny edge detection to find the edges
    edges = cv2.Canny(scale_bar_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=100,  # Min number of votes for valid line
        minLineLength=5,  # Min allowed length of line
        maxLineGap=10,  # Max allowed gap between line for joining them
    )
    bar_size = int(lines[0][0][2] - lines[0][0][0])

    # Calculate the scale
    scale = scale_int / bar_size
    pm = (scale_int / (bar_size - 1)) - scale

    if return_all:
        return scale_str, scale_int, scale, pm

    return scale, pm


def get_object(
    image: np.ndarray, mask: np.ndarray, dil_iter: int = 10, show_steps: bool = False
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
    dilated_mask = cv2.dilate(
        rotated_mask, np.ones((3, 3), np.uint8), iterations=dil_iter
    )

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
