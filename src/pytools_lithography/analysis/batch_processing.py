import os
import cv2
import numpy as np
from pytools_image_processing.utils import show_images
from .image_processing import (
    get_bounding_rect,
    get_object,
    separate_objects,
    mark_objects,
)

try:
    import easyocr as ocr
except ImportError:
    pass


class ImagePreProcessor:
    """Preprocess the SEM image.

    This class is used to preprocess the SEM image. It can separate the info bar
    from the image and determine the scale of the image.

    .. warning::

        This is by now means a general class and should not be used for
        images that are not from the MPD lab.
    """

    # Initial height guess for the info bar
    _height = 0.10

    def __init__(self, image: np.ndarray) -> ...:
        """Initialize the ImagePreProcessor class.

        Parameters
        ----------
        image : np.ndarray
            The raw SEM image.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(
                "image should be a numpy array not type {}".format(type(image))
            )
        if not len(image.shape) == 2:
            raise ValueError(
                "image should be a grayscale image not shape {}".format(image.shape)
            )
        self._image = image
            # Bounding box of the info bar
        self._x = None
        self._y = None
        self._w = None
        self._h = None

    def get_scale(self, method: str = "dynamic") -> tuple[float, float]:
        """Determine the number of nm/pixel factor from the scale bar.

        This function expects a cropped scale bar from the SEM in the MPD
        lab. This is in now way a generalized function and should not be used
        for other images. The cropped scale bar is complete bottom part of
        the image with a horizontal cut where the image begins.

        Parameters
        ----------
        method : str, optional
            The method to use to determine the scale, this can be dynamic or static.
            When dynamic is chosen the scale is determined by the number on the left
            side of the scale bar. When static is chosen the scale is determined by
            the number on the right side of the scale bar. The default is dynamic.

        Returns
        -------
        tuple[float, float]
            The scale in nm/pixel and the plus minus error.
        """
        if not isinstance(method, str):
            raise ValueError(
                "method should be a string not type {}".format(type(method))
            )
        method = method.lower()
        if not method in ["dynamic", "static"]:
            raise ValueError(
                "method should be either dynamic or static not {}".format(method)
            )
        if not self._check_ready_to_crop():
            self._find_info_bar()

        # Get the scale from the image
        if method == "dynamic":
            scale, pm = self._get_dynamic_scale()
        elif method == "static":
            scale, pm = self._get_static_scale()

        return scale, pm

    def get_info_bar(self) -> np.ndarray:
        """Get the info bar from the image.

        Returns
        -------
        np.ndarray
            The info bar.
        """
        if not self._check_ready_to_crop():
            self._find_info_bar()

        return self._image[self._y : self._y + self._h, self._x : self._x + self._w]

    def get_image(self) -> np.ndarray:
        """Get the image without the info bar.

        Returns
        -------
        np.ndarray
            The image without the info bar.
        """
        if not self._check_ready_to_crop():
            self._find_info_bar()

        return self._image[: self._y, :]

    def _check_ocr_import(self) -> ...:
        """Check if the OCR package can be imported."""
        try:
            import easyocr as ocr
        except ImportError:
            raise ImportError(
                "The easyocr package is required to use this function. "
                "You can install it using 'pip install easyocr' or use the"
                " 'pytools_lithography' package with the 'ocr' extra."
            )

    def _check_ready_to_crop(self) -> bool:
        """Check if the info bar is found.

        Returns
        -------
        bool
            True if the info bar is found, False otherwise.
        """
        if None in [self._x, self._y, self._w, self._h]:
            return False
        return True

    def _find_info_bar(self) -> ...:
        """Find the info bar in the image.

        The info bar is assumed to be a straight black bar at the bottom of the image.
        The function will search for the biggest black contour at the bottom of the image
        and use this as the info bar.

        Raises
        ------
        ValueError
            If no valid info bar is found in the image.
        """
        # Get the image dimensions
        img_x, img_y = self._image.shape

        # Get the height of the scale bar
        scale_height = int(img_y * self._height)

        # Get the scale bar from the bottom of the image
        scale_bar = self._image[-scale_height:, :]

        # Search for the biggest black contour
        mask = cv2.inRange(scale_bar, 0, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise ValueError("No valid info bar found in the image!")

        # Get the bounding box of the biggest contours
        x, y, w, h = cv2.boundingRect(contours[0])

        # Store the bounding box of the info bar
        self._x = x
        self._y = img_y
        self._w = w
        self._h = h

    def _get_dynamic_scale(self) -> tuple[float, float]:
        """Get the scale of the image using the dynamic scale bar.

        The dynamic scale is calculated by the number on the left side of the
        scale bar and the length of the scale bar.

        Returns
        -------
        tuple[float, float]
            The scale in nm/pixel and the plus minus error.
        """
        # Check if the OCR package can be imported
        self._check_ocr_import()

        # Create the OCR
        reader = ocr.Reader(["en"], gpu=False)

        # How big of a 'cut-out' do we make from the left corner in the up and right  direction
        DYNAMIC_SCALE_IMG_DIMENSIONS = 95

        # Crop a copy of the image to the size of the number
        scale_text_img = self._image.copy()[:, :DYNAMIC_SCALE_IMG_DIMENSIONS]

        # We use detail = 0 to just get the text, we dont care for the other info
        scale_str = reader.readtext(scale_text_img, detail=0)[0]

        for char in scale_str:
            if not char.isdigit():
                scale_str = scale_str.replace(char, "")

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
        scale_bar_img = self._image[:, SCALE_BAR_START:SCALE_BAR_END]

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

        return scale, pm

    def _get_static_scale(self) -> tuple[float, float]:
        """Get the scale of the image using the static scale bar.

        The static scale is calculated by the number on the right side of the
        scale bar and the width of the image.

        Returns
        -------
        tuple[float, float]
            The scale in nm/pixel and the plus minus error.
        """
        # Check if the OCR package can be imported
        self._check_ocr_import()

        raise NotImplementedError("This function is not implemented yet")


class ImageProcessor:
    """Process an image.

    The image processor is used to process an image. The processor will extract
    the profiles from the image and calculate the PSD of the profiles. The profiles
    are saved in a dictionary.

    The dictionary will contain the following keys:
    {
        "origin_image": str,
        "object_id": int,
        "analysis_successful": bool,
        "offset_corrected": bool,
        "top_profile": np.ndarray,
        "bottom_profile": np.ndarray,
        "width_profile": np.ndarray,
        "top_psd_roughness": float,
        "bottom_psd_roughness": float,
        "width_psd_roughness": float,
    }

    For each image a json file with the image name will be created. This file will
    contain the dictionary for the objects in the image.
    """

    def __init__(self, filename: str, start_id: int = 0) -> ...:
        """Initialize the ImageProcessor class.

        Parameters
        ----------
        filename : str
            The filename of the image to process.
        start_id : int
            The starting id for the objects in the image. The default is 0.
            The starting ID is usefull when using batch processing to keep track
            of the objects in multiple images.
        """
        if not isinstance(filename, str):
            raise ValueError(
                "filename should be a string not type {}".format(type(filename))
            )
        if not isinstance(start_id, int):
            raise ValueError(
                "start_id should be an integer not type {}".format(type(start_id))
            )
        if start_id < 0:
            raise ValueError("start_id should be a positive integer")
        
        self.check_folder_exists(file_path=filename, raise_exception=True)
        self.check_image_exists(filename, raise_exception=True)

        self._start_id = start_id
        self._filename = filename
        self._raw_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        self._preprocessor = ImagePreProcessor(self._raw_image)
        self._preprocessed = False
        self._processed = False

    # ATTRIBUTES
    @property
    def filename(self) -> str:
        """Get the filename of the image."""
        return self._filename
    
    @property
    def raw_image(self) -> np.ndarray:
        """Get the raw image."""
        return self._raw_image
    
    @property
    def preprocessed(self) -> bool:
        """Get the preprocessed status."""
        return self._preprocessed
    
    @property
    def processed(self) -> bool:
        """Get the processed status."""
        return self._processed
    
    @property
    def lines_image(self) -> np.ndarray:
        """Get the image without the info bar."""
        if not self.preprocessed:
            raise ValueError("The image has not been preprocessed yet")
        return self._lines_image
    
    @property
    def scale(self) -> float:
        """Get the scale of the image."""
        if not self.preprocessed:
            raise ValueError("The image has not been preprocessed yet")
        return self._scale

    # PUBLIC FUNCTIONS
    @staticmethod
    def check_folder_exists(
        folder_name: str = None, file_path: str=None, raise_exception: bool = True
    ) -> bool:
        """Check if the folder exists.

        Parameters
        ----------
        folder_name : str, optional
            The folder name to check. The default is None.
        file_path : str, optional
            The file path to check. The default is None.
        raise_exception : bool, optional
            Raise an exception if the folder does not exist. The default is True.

        Returns
        -------
        bool
            True if the folder exists, False otherwise.
        """
        if not isinstance(folder_name, (type(None), str)):
            raise ValueError(
                "folder_name should be a string not type {}".format(type(folder_name))
            )
        if not isinstance(file_path, (type(None), str)):
            raise ValueError(
                "file_path should be a string not type {}".format(type(file_path))
            )
        if not isinstance(raise_exception, bool):
            raise ValueError(
                "raise_exception should be a boolean not type {}".format(
                    type(raise_exception)
                )
            )
        if folder_name is None and file_path is None:
            raise ValueError("Either folder_name or file_path should be given")

        if folder_name is not None:
            if not os.path.exists(folder_name):
                if raise_exception:
                    raise FileNotFoundError(
                        "The folder {} does not exist".format(folder_name)
                    )
                return False
            return True
        if file_path is not None:
            if not os.path.exists(file_path):
                if raise_exception:
                    raise FileNotFoundError(
                        "The file {} does not exist".format(file_path)
                    )
                return False
            return True

    @staticmethod
    def check_image_exists(filename: str, raise_exception: bool = True) -> bool:
        """Check if the image exists.

        Parameters
        ----------
        filename : str
            The filename to check.
        raise_exception : bool, optional
            Raise an exception if the image does not exist. The default is True.

        Returns
        -------
        bool
            True if the image exists, False otherwise.
        """
        if not isinstance(filename, str):
            raise ValueError(
                "filename should be a string not type {}".format(type(filename))
            )
        if not isinstance(raise_exception, bool):
            raise ValueError(
                "raise_exception should be a boolean not type {}".format(
                    type(raise_exception)
                )
            )
        if not os.path.exists(filename):
            if raise_exception:
                raise FileNotFoundError("The image {} does not exist".format(filename))
            return False
        return True
    
    def preprocess_image(self) -> ...:
        """Preprocess the image.

        The image will be preprocessed by separating the info bar from the image
        and determining the scale of the image.

        Returns
        -------
        np.ndarray
            The image without the info bar.
        """
        if self.preprocessed:
            return
        self._lines_image = self._preprocessor.get_image()
        self._scale = self._preprocessor.get_scale()
        self._preprocessed = True

    def process_image(self, max_fault_tolerance: float=0.25, ignore_errors:bool=True) -> ...:
        """Process the image.

        The image will be processed by extracting the profiles from the image and
        calculating the PSD of the profiles. The profiles are saved in a dictionary.
        """
        if not self.preprocessed:
            self.preprocess_image()

        # Get the profiles
        profiles, angles = self._get_profiles()

    # PRIVATE FUNCTIONS
    def _get_profiles(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the profiles from the image.

        Returns
        -------
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
            The profiles of the top, bottom and width of the objects in the image. Each
            object can be found using the ID as the key.
        """
        img_objects_masks, angles = separate_objects(self.lines_image)
        self._marked_image = mark_objects(self.lines_image, img_objects_masks, start_id=self._start_id)
        line_objects = []

        return profiles, angles
       
    def _get_psd(self, profile: np.ndarray) -> float:
        """Calculate the PSD of a profile.

        Parameters
        ----------
        profile : np.ndarray
            The profile to calculate the PSD of.

        Returns
        -------
        float
            The PSD of the profile.
        """
        raise NotImplementedError("This function is not implemented yet")
    
    def _save_results(self) -> ...:
        """Save the results of the image processing."""
        raise NotImplementedError("This function is not implemented yet")


class BatchProcessor:
    """Process a batch of images.

    The batch processor is used to process a batch of images. The processor
    will extract the profiles from the images and calculate the PSD of the
    profiles. The profiles, PSD and some parameters are saved in a dictionary.

    A dictionary will be generated for each image in the batch. The dictionary
    will contain the following keys:
    {
        "<object_nr>": {
            "origin_image": str,
            "offset_corrected": bool,
            "top_profile": np.ndarray,
            "bottom_profile": np.ndarray,
            "width_profile": np.ndarray,
    }

    An entry will be generated for each object in the image, for each image a dict
    will be generated. The dict will be saved in a json file with the image filename.
    To keep track of which object is which a marked image will also be saved.
    """

    _allowed_image_types = [".png", ".jpg", ".jpeg"]

    def __init__(self, folder_name: str) -> ...:
        """Initialize the batch processor.

        Parameters
        ----------
        folder_name : str
            The folder name where the images are stored.
        """
        if not isinstance(folder_name, str):
            raise ValueError(
                "folder_name should be a string not type {}".format(type(folder_name))
            )
        if not os.path.exists(folder_name):
            raise FileNotFoundError("The folder {} does not exist".format(folder_name))
        if not os.path.isdir(folder_name):
            raise ValueError("{} is not a folder".format(folder_name))
        if not os.listdir(folder_name):
            raise ValueError("The folder {} is empty".format(folder_name))

        self._folder_name = folder_name
        self._image_names = self._get_image_names()

    # ATTRIBUTES
    @property
    def folder_name(self) -> str:
        """Get the folder name."""
        return self._folder_name

    @property
    def image_names(self) -> list[str]:
        """Get the image names in the folder."""
        return self._image_names

    # PUBLIC FUNCTIONS

    # PRIVATE FUNCTIONS
    def _get_image_names(self) -> list[str]:
        """Get the image names in the folder."""
        content = os.listdir(self.folder_name)
        images = []
        for image_name in content:
            extension = os.path.splitext(image_name)[1]
            if extension in self._allowed_image_types:
                images.append(image_name)
        return images
