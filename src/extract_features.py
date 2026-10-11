import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path


# ============================================================
# GIST FEATURE
# ============================================================

def extract_gist(image):
    """
    GIST-style global image descriptor.

    4 scales × 8 orientations × 4 × 4 spatial regions
    = 512 features
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    # Resize to fixed size
    gray = cv2.resize(gray, (256, 256))

    scales = [3.0, 5.0, 7.0, 9.0]
    orientations = np.linspace(0, np.pi, 8, endpoint=False)

    features = []

    for sigma in scales:
        for theta in orientations:

            kernel = cv2.getGaborKernel(
                (21, 21),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )

            filtered = cv2.filter2D(
                gray,
                cv2.CV_32F,
                kernel
            )

            # Magnitude
            filtered = np.abs(filtered)

            # 4 × 4 spatial regions
            for row in range(4):
                for col in range(4):

                    y1 = row * 64
                    y2 = (row + 1) * 64

                    x1 = col * 64
                    x2 = (col + 1) * 64

                    region = filtered[y1:y2, x1:x2]

                    features.append(np.mean(region))

    return np.array(features, dtype=np.float32)


# ============================================================
# COLOR HISTOGRAM
# ============================================================

def extract_color_histogram(image):
    """
    RGB color histogram.
    32 bins per channel = 96 features.
    """

    features = []

    for channel in range(3):

        hist = cv2.calcHist(
            [image],
            [channel],
            None,
            [32],
            [0, 256]
        )

        hist = cv2.normalize(
            hist,
            hist
        ).flatten()

        features.extend(hist)

    return np.array(features, dtype=np.float32)


# ============================================================
# HU MOMENTS
# ============================================================

def extract_hu_moments(image):
    """
    7 Hu moment features.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    moments = cv2.moments(gray)

    hu = cv2.HuMoments(moments).flatten()

    # Log transform for numerical stability
    hu = -np.sign(hu) * np.log10(
        np.abs(hu) + 1e-10
    )

    return hu.astype(np.float32)


# ============================================================
# HOG
# ============================================================

def extract_hog(image):
    """
    Histogram of Oriented Gradients.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        (128, 128)
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_features(image_path):
    """
    Extract all features from one image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # OpenCV loads BGR → convert to RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Paper preprocessing: 256 × 256 RGB
    image = cv2.resize(
        image,
        (256, 256)
    )

    gist = extract_gist(image)
    color = extract_color_histogram(image)
    hu = extract_hu_moments(image)
    hog_features = extract_hog(image)

    return {
        "gist": gist,
        "color_histogram": color,
        "hu_moments": hu,
        "hog": hog_features
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_dir = Path("data/raw/images")

    images = list(image_dir.glob("*.jpg"))

    if not images:
        print("No images found.")
        exit()

    test_image = images[0]

    print("Feature extraction test")
    print("=" * 50)
    print(f"Image: {test_image.name}")

    features = extract_features(test_image)

    for name, values in features.items():
        print(
            f"{name:20s}: "
            f"{len(values)} values"
        )

    print("=" * 50)
    print("Feature extraction successful!")