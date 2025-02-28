from pytools_image_processing.morphologic import (
    normalize_image,
)
from pytools_image_processing.analysis import (
    find_components,
)
from pytools_image_processing.utils import show_images, get_bounding_rect
import numpy as np
import cv2


def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert the image.

    Parameters
    ----------
    image : np.ndarray
        The image to invert.

    Returns
    -------
    np.ndarray
        The inverted image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    
    if len(image.shape) == 2:
        return 255 - image
    else:
        return cv2.bitwise_not(image)
    

def separate_objects(
    image: np.ndarray,
    normalize: bool = True,
    mask_range: tuple[float] = (0.05, 0.8),
    invert_image: bool = False,
    dilate_iterations: int = 30,
    show_steps: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Separate objects in a SEM image.

    This function takes a grayscale image and returns a list of images
    with the separated objects. The function finds the object by using a
    Gaussian blur and a OTSU threshold. During OTSU thresholding a lot
    of noise will be found so these are removed by defining a minimum
    and maximum size of the components. The components are then dilated
    to make sure the whole object is included in the cropped image.

    .. attention::

        This function assumes that the line profile is higher than the
        substrate around the lines. This means that when only development 
        is done the reverse will be true and the lines will be lower than
        the substrate. This will cause the function to either fail or
        see the space between the lines as the line (in the case of a pre
        cropped image)

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
    invert_image : bool, optional
        If True, invert the image. The default is False. This is useful
        when the lines are lower than the substrate.
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
        # crop_components=False,
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


def mark_objects(
    image: np.ndarray, masks: list[np.ndarray], start_id: int, show_steps: bool = False
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mark the objects to keep track of them.

    During batch processing it becomes harder to keep track of the
    different objects. This function can be used to mark the different
    objects in the image. The objects will me marked with a bounding box
    and a number, starting from 1 in the order of the given masks.

    Parameters
    ----------
    image : np.ndarray
        The image to mark.
    masks : list[np.ndarray]
        The masks of the objects.
    start_id : int
        The number to start counting from.
    show_steps : bool, optional
        If True, show the steps of the calculation. The default is False.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        The marked image and list of ID's.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            "image should be a numpy array not type {}".format(type(image))
        )
    if not len(image.shape) == 2:
        raise ValueError(
            "image should be a grayscale image not shape {}".format(image.shape)
        )
    if not isinstance(masks, list):
        raise ValueError("masks should be a list not type {}".format(type(masks)))
    if not all(isinstance(mask, np.ndarray) for mask in masks):
        raise ValueError(
            "masks should be a list of numpy arrays not {}".format(
                [type(mask) for mask in masks]
            )
        )
    if not all(mask.shape == image.shape for mask in masks):
        raise ValueError(
            "masks should have the same shape as the image not {}".format(
                [mask.shape for mask in masks]
            )
        )
    if not isinstance(start_id, int):
        raise ValueError(
            "start_id should be an integer not type {}".format(type(start_id))
        )
    if not isinstance(show_steps, bool):
        raise ValueError(
            "show_steps should be a boolean not type {}".format(type(show_steps))
        )

    # Create a copy of the image
    marked_image = image.copy()

    # Draw the masks on the image
    ids = []
    for i, mask in enumerate(masks):
        # Get the bounding box of the mask
        box, (x, y, w, h, angle) = get_bounding_rect(mask)

        # Draw the bounding box using the point box so it is rotated
        cv2.polylines(marked_image, [box], isClosed=True, color=255, thickness=2)

        # Draw the number of the object
        id = str(start_id + i + 1)
        cv2.putText(
            marked_image,
            id,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            255,
            thickness=2,
        )
        ids.append(id)


    if show_steps:
        images = {"Marked image": marked_image}
        show_images(images)

    return marked_image, ids


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

    # Rotation causes some annoing black pixels around the object in
    # some cases so we need to remove them. The used method is not
    # perfect but it works for now.
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


  