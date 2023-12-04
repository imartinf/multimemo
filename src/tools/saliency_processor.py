import os
import cv2

import numpy as np

"""
Saliency processor class.

Credits to sergioesro for the original code.
"""


class SaliencyProcessor:
    def __init__(self, path_to_images, saliency_mode="fine_grained") -> None:
        self.path_to_images = path_to_images
        self.saliency_mode = saliency_mode

    def list_images(self):
        """
        Returns a list of all the images in the dataset.
        """
        return os.listdir(self.path_to_images)

    def get_saliency_map(self, image_name: str) -> np.ndarray:
        """
        Returns the saliency map of the given image between zeros and one.
        """
        image = cv2.imread(os.path.join(self.path_to_images, image_name))
        saliency = self.saliency_fn_selector(self.saliency_mode)
        (success, saliency_map) = saliency.computeSaliency(image)
        if success:
            return saliency_map
        else:
            # Warn and return zeros
            print(f"Error computing saliency map for image {image_name}.")
            return np.zeros(image.shape[:2])

    def apply_saliency_map_over_image(self, img_name: str, saliency_map: np.ndarray, save=False) -> np.ndarray:
        """
        Applies the saliency map over the given image.
        """
        img = cv2.imread(os.path.join(self.path_to_images, img_name))
        for i in range(0, img.shape[2]):
            img[:, :, i] = img[:, :, i] * saliency_map
        if save:
            self.save(img)
        return img

    def save(self, img, path):
        """
        Saves the given image to the given path.
        """
        # Convert to range 0-255 and uint8
        img = img * 255
        img = img.astype(np.uint8)
        cv2.imwrite(path, img)

    def load_map(self, path):
        """
        Loads the saliency map image from the given path.
        Returns it converted to float32 and range 0-1.
        """
        img = cv2.imread(path)
        img = img.astype(np.float32)
        img = img / 255
        return img

    @staticmethod
    def saliency_fn_selector(saliency_mode="fine_grained"):
        """
        Returns the saliency function to use based on the given mode.
        """
        if saliency_mode == "fine_grained":
            return cv2.saliency.StaticSaliencyFineGrained_create()
        elif saliency_mode == "residual":
            return cv2.saliency.StaticSaliencySpectralResidual_create()
        else:
            raise ValueError(f"Unknown saliency mode {saliency_mode}.")
