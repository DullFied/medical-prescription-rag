import os
import pytesseract
import numpy as np
from PIL import Image

# Windows: point pytesseract to the Tesseract executable.
# Adjust this path if you installed Tesseract to a different location.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def run_tesseract(image: np.ndarray) -> str:
    """
    Run Tesseract OCR on a preprocessed (numpy) image.
    Returns extracted text string, or empty string on failure.
    """
    try:
        print("[Tesseract OCR] Running Tesseract as fallback...")

        # Convert numpy array to PIL Image for pytesseract
        pil_image = Image.fromarray(image)

        # PSM 6: assume a single block of text — good for prescriptions
        config = "--psm 6"
        text = pytesseract.image_to_string(pil_image, config=config).strip()

        if not text:
            print("[Tesseract OCR] Tesseract returned empty text.")
            return ""

        print(f"[Tesseract OCR] Extracted {len(text)} characters.")
        return text

    except Exception as e:
        print(f"[Tesseract OCR] Error: {e}")
        return ""