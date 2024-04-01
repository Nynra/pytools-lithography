import numpy as np
from .ImageAnalyzer import Analyzer


def normalize(values: np.ndarray) -> np.ndarray:
    min_value = min(values)
    max_value = max(values)
    normalize_values = (values - min_value) / (max_value - min_value)
    return normalize_values

class Image():
    def __init__(self, image: np.array) -> None:
        self.image = image
        self.analyzer = Analyzer(self.image)

    def get_dimensions(self):
        return self.analyzer.height, self.analyzer.width

    def get_scale(self, return_all = False, return_pm = False):
        return self.analyzer.get_scale(return_all, return_pm)
    
    def get_line_width(self):
        return self.analyzer.get_line_width()
    
    def get_cropped_img(self):
        return self.analyzer.crop(self.image)
    
    def get_splitted_img(self):
        return self.analyzer.split(self.get_cropped_img())

    def get_normalized(self, image, axis: int = 0):
        return normalize(image.mean(axis=axis))
    
    def get_normalized_split(self, axis=0):
        n = 100
        image = self.get_splitted_img()[1][n:n+20, :]
        
        return self.get_normalized(image, axis=axis)

        

    

    

   
    