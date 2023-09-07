"""
PlotsTest

Test the funtions associated with plotting.
"""

import unittest
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.visualization.visualize import plot_histogram, plot_lineplot

class PlotsTest(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'oov': [0, 1, 2, 3, 4],
            'mean': [0.1, 0.2, 0.3, 0.4, 0.5],
            'std': [0.01, 0.02, 0.03, 0.04, 0.05]
        })

    def test_plot_histogram(self):
        plot_histogram(self.data, ['oov'], 'title', 'xlabel', 'ylabel', bins=10, figsize=(9, 5), show=True, save_path='title.png')
        self.assertTrue(os.path.exists('title.png'))
        os.remove('title.png')

    def test_plot_lineplot(self):
        plot_lineplot(self.data, 'oov', 'mean', 'title', 'xlabel', 'ylabel', figsize=(9, 5), err='std', show=True, save_path='title.png')
        self.assertTrue(os.path.exists('title.png'))
        os.remove('title.png')