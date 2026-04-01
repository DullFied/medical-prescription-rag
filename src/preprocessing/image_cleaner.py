import cv2
import numpy as np


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a prescription image for Tesseract OCR.
    Only call this when Gemini has failed and Tesseract fallback is needed.

    Steps: grayscale → gaussian blur → adaptive threshold
    Returns the processed image as a numpy array.
    """
    print(f"[Preprocess] Preprocessing for Tesseract: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    processed = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    print("[Preprocess] Done — grayscale, blur, threshold applied.")
    return processed