import cv2
import os
import numpy as np


def get_test_img_path(filename: str) -> str:
    """Get the path to an image in the tests/images directory.

    Parameters
    ----------
    filename : str
        The name of the image file.

    Returns
    -------
    str
        The path to the image file.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string not type {}".format(type(filename)))

        # Check if the file exists
    img_path = os.path.join(os.path.dirname(__file__), "images", filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError("The file {} does not exist".format(img_path))

    return img_path


def load_test_image(filename: str) -> np.ndarray:
    """Load an image from the tests/images directory.

    Parameters
    ----------
    filename : str
        The name of the image file.

    Returns
    -------
    np.ndarray
        The loaded image.
    """
    img_path = get_test_img_path(filename)
    img = cv2.imread(img_path)

    return img
