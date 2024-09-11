import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
import cv2
import easyocr as ocr
from itertools import groupby

def normalize(values: np.ndarray) -> np.ndarray:
    min_value = min(values)
    max_value = max(values)
    normalize_values = (values - min_value) / (max_value - min_value)
    return normalize_values

def get_initial_x_offset_guess(y: np.ndarray) -> tuple[float, float]:
    guess = np.where(y > 0.5, 1, 0)
    guess = np.abs(np.diff(guess))
    guess = np.where(guess == 1)
    guess = (guess[0][0], guess[0][-1])
    return guess

def error_function(x: np.array, amplitude, x_offset, sigma, y_offset) -> np.array:
    diff = x - x_offset
    erf_term = erf(diff / (sigma * np.sqrt(2)))
    y = amplitude * 0.5 * (1 + erf_term) + y_offset
    return np.array(y)


# Constant variables mainly for the scale bar and number, SCALE_BAR_START and SCALE_BAR_END are
# a bit of gueswork just to limit the process time, but should always work.
SCALE_IMG_DIMENSIONS = 125
SCALE_BAR_START = 200
SCALE_BAR_END = 1250

class Analyzer():
    
    def __init__(self, image) -> None:
        self.image = image
        self.height, self.width = image.shape

        self.view = image[0 : self.height- SCALE_IMG_DIMENSIONS, 0:self.width]
        self.info = image[self.height - SCALE_IMG_DIMENSIONS : self.height, 0:self.width]
        self.view_height, self.view_width = self.view.shape
        self.info_height, self.info_width = self.info.shape

    
    def smooth(self,image, window_size: int) -> np.array:
        height, width = image.shape

        flattend_img = pd.Series(image.ravel())
        rolling_mean = flattend_img.rolling(window=window_size).mean().fillna(100)
        smoothed_img = np.array(rolling_mean).reshape(height, width)
        return smoothed_img

    def get_groups(self, bool_array=np.ndarray) -> tuple[list, list]:
        get = lambda group, key, value: len(list(group)) if key == value else 0

        # Create a list to store the lengths of consecutive True values
        true_values = [get(group, key, True) for key, group in groupby(bool_array)]

        # Create a list to store the lengths of consecutive False values
        false_values = [get(group, key, False) for key, group in groupby(bool_array)]
        return (true_values, false_values)
    
    def alternate_combining(self, list1: list, list2: list) -> list:
        result = []
        # make sure the first element in list2 is larger than the first element in list1
        if list1[0] > list2[0]:
            list1, list2 = list2, list1

        # combine the lists by taking elements alternately up to the minimum length
        min_len = min(len(list1), len(list2))
        for i in range(min_len):
            result.append(list1[i])
            result.append(list2[i])

        # extend the result list with any remaining elements in list1 and list2
        result.extend(list1[min_len:])
        result.extend(list2[min_len:])
        return result


    def crop(self, image):
        smoothed_img = self.smooth(image, 10)

        # Horizontal
        mean_list = np.mean(smoothed_img, axis=0)
        bool_array = mean_list > mean_list.mean()
        _, dark_pixels = self.get_groups(bool_array)

        # crop out the dark pixels
        upper = max(0, dark_pixels[0] - 100)
        lower = -max(1, dark_pixels[-1] - 100)
        cropped_image = self.view[:, upper:lower]

        # Vertical
        # divide pixels into light (True) and dark (False) pixels
        mean_list = np.mean(cropped_image, axis=1)
        bool_array = mean_list > mean_list.mean()

        # group light and dark pixel series
        light_pixels, dark_pixels = self.get_groups(bool_array)
        pixel_intensity_series = self.alternate_combining(
            light_pixels, dark_pixels
        )

        # keep only the larges (light) pixel series
        idx = pixel_intensity_series.index(max(pixel_intensity_series))
        before_largest = sum(pixel_intensity_series[:idx])
        after_largest = sum(pixel_intensity_series[idx + 1 :])

        # crop around the larges pixel series
        cropped_image = cropped_image[before_largest:-after_largest, :]

        return cropped_image

    def get_str_from_img(self, image: np.array):
        # Gets images and extracts text, jeej
        
        reader = ocr.Reader(['en'], gpu=True)
        # We use detail = 0 to just get the text, we dont care for the other info
        text_str = reader.readtext(image, detail = 0)
        return text_str

    def get_scale_bar_size(self, image: np.array) -> int:
        scale_img = image[0 : self.info_height, SCALE_BAR_START:SCALE_BAR_END]
        
        edges = cv2.Canny(scale_img, 50, 150, apertureSize = 3)
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=100, # Min number of votes for valid line
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=10 # Max allowed gap between line for joining them
        )
        
        return int(lines[0][0][2] - lines[0][0][0])


    def get_scale(self, return_all = False, return_pm = False):
        # How big of a 'cut-out' do we make from the left corner in the up and right  direction
        SCALE_IMG_DIMENSIONS = 95
        
        scale_img = self.info[0:self.info_height, 0: SCALE_IMG_DIMENSIONS]
        scale_str = self.get_str_from_img(scale_img)[0]
        scale_int = int(scale_str)
        
        scale = scale_int / self.get_scale_bar_size(self.info)
        pm = (scale_int / (self.get_scale_bar_size(self.info) - 1)) - scale
        
        if return_all:
            return scale_str, scale_int, scale
        elif return_pm:
            return scale, pm
        
        return scale

    def get_clean(self):
        image = self.crop(self.view)
        image = self.split(image)
        return image

    def get_line_width(self, group: int = 20):
        image_dict = self.get_clean()
        scale = self.get_scale()

        width_list = []

        for image_slice in image_dict.values():
            # Slice the image into smaller groups and iterate through them.
            for i in range(0, image_slice.shape[0], group):
                # Calculate the width of the sliced image.
                width = self._calculate_width(image_slice[i : i + group, :])

                if width is not None:
                    width_list.append(width * scale)       
        width_list = np.array(width_list)
        # Calculate and return the mean, standard deviation, and number of widths.
        return (width_list.mean(), width_list.std(), len(width_list))



    def split(self, image):
        midpoints = self._get_midpoints(image)
        image_slices = {}
        for idx in range(len(midpoints) - 1):
            slice_start = midpoints[idx]
            slice_end = midpoints[idx + 1] + 20
            image_slice = image[:, slice_start:slice_end]
            image_slices[idx] = image_slice
        return image_slices

    def _get_midpoints(self, image):
        # Make black pixels gray
        image = np.where(image == 0, 100, image)

        intensities = self._get_mean_intensities(image)
        intensity_midpoint = (min(intensities)+max(intensities)) / 2

        in_lower_part = False
        lower_part_start = 0
        midpoints = []
        
        # add midpoints to dict
        for i in range(len(intensities)):
            if intensities[i] < intensity_midpoint and not in_lower_part:
                in_lower_part = True
                lower_part_start = i
            elif intensities[i] >= intensity_midpoint and in_lower_part:
                in_lower_part = False
                midpoint = (lower_part_start + i) // 2
                midpoints.append(midpoint)
          
        extra = 80      
        if intensities[-80] > 0.5:
            extra = 0
        midpoints.append(len(intensities) - extra)
        return midpoints

    def _calculate_width(self, image: np.array) -> float | None:
        try:
            y = normalize(image.mean(axis=0))
            x = np.arange(len(y))
            constants1, constants2 = self._fit_data(x, y)
            width = constants2[1] - constants1[1]
            return width
        except:
            return None
        
    def _fit_data(self,
        x: np.ndarray, y: np.ndarray, covariance = False) -> tuple[np.array, np.array] | tuple[None, None]:
        
        try:
            # split the x and y values so we can fit two funtions,
            x1, y1 = x[0 : len(x) // 2], y[0 : len(y) // 2]
            x2, y2 = x[len(x) // 2 : -1], y[len(x) // 2 : -1]

            # define initial guesses
            x_offset_guess1, x_offset_guess2 = get_initial_x_offset_guess(y)
            y_offset_guess1, y_offset_guess2 = 0.0, -2.0
            amplitude_guess1, amplitude_guess2 = 1.0, -1.0
            sigma_guess = 5
            
            initial_guess1 = [amplitude_guess1, x_offset_guess1, sigma_guess, y_offset_guess1]
            initial_guess2 = [amplitude_guess2, x_offset_guess2, sigma_guess, y_offset_guess2]

            # fit the error function to the data 
            constants1, covariance1  = curve_fit(error_function, x1, y1, p0=initial_guess1)
            constants2, covariance2  = curve_fit(error_function, x2, y2, p0=initial_guess2)
            if covariance:
                return (constants1, constants2, np.sqrt(np.diag(covariance1)), np.sqrt(np.diag(covariance2)))
            
            return (constants1, constants2)
        except:
            return (None, None)

    def _get_mean_intensities(self, image: np.array):
        BOUNDERY = 10

        intensities = image.mean(axis=0)[BOUNDERY:-BOUNDERY]
        intensities_norm = normalize(intensities)
        return intensities_norm


        