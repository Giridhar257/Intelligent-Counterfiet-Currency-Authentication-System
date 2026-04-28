import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P=8, R=1)
    hist, _ = np.histogram(lbp.ravel(), bins=128, range=(0, 128))

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    return hist
