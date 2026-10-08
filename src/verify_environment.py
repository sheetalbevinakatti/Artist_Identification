import numpy as np
import pandas as pd
import cv2
import sklearn
import skimage
import PIL
import scipy
import tensorflow as tf

print("=" * 50)
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

print("NumPy       :", np.__version__)
print("Pandas      :", pd.__version__)
print("OpenCV      :", cv2.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Scikit-image:", skimage.__version__)
print("Pillow      :", PIL.__version__)
print("SciPy       :", scipy.__version__)
print("TensorFlow  :", tf.__version__)

print("\nTensorFlow GPUs:")
print(tf.config.list_physical_devices("GPU"))

print("\nEnvironment verification successful!")