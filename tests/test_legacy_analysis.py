from unittest import TestCase
import numpy as np
from pytools_lithography.analysis.legacy import Analyzer, Image
from utils import get_test_img_path
import cv2


class TestLegacyAnalyzer(TestCase):

    def setUp(self):
        filename = 'legacy_analysis_test.jpg'
        self.image_file = cv2.imread(get_test_img_path(filename), cv2.IMREAD_UNCHANGED)
        self.image = Image(self.image_file)

    def test_get_scale_only(self):
        scale = self.image.get_scale()
        self.assertEqual(round(scale, 4), 0.1176)

    def test_get_scale_and_pm(self):
        scale, pm = self.image.get_scale(return_pm=True)
        self.assertEqual(round(scale, 4), 0.1176)
        self.assertEqual(round(pm, 4), 0.0001)
