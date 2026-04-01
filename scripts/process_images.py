"""
process_images.py

For each image in data/raw_images/:
1. Skip if already processed (JSON already exists)
2. Preprocess with OpenCV
3. Run Gemini Vision OCR
4. Fallback to Tesseract if Gemini fails
5. Save result to data/structured_json/<filename>.json

Handles Gemini rate limits gracefully — if the API quota is hit,
remaining images are skipped and reported. Run the script again later
to pick up where it left off (already-processed images are skipped).

Usage:
    python scripts\process_images.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.image_cleaner import preprocess_image
from src.ocr.gemini_vision import run_gemini_vision
from src.ocr.tesseract_ocr import run_tesseract

RAW_IMAGES_DIR = "data/raw_images"
JSON_OUTPUT_DIR = "data/structured_json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Seconds to wait between Gemini API calls to avoid hitting rate limits.
# Free tier allows ~15 requests/min — 5s gap keeps well within that.
GEMINI_DELAY_SECONDS = 5


def already_processed(filename: str) -> bool:
    """Return True if a JSON for this image already exists."""
    base_name = os.path.splitext(filename)[0]
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
    return os.path.exists(json_path)


def process_single_image(image_path: str) -> dict:
    """Run full OCR pipeline on one image. Returns result dict."""
    filename = os.path.basename(image_path)
    print(f"\n{'='*50}")
    print(f"Processing: {filename}")

    processed_image = preprocess_image(image_path)
    text = run_gemini_vision(image_path)

    if not text:
        print("[Pipeline] Falling back to Tesseract OCR...")
        text = run_tesseract(processed_image)

    if not text:
        print(f"[Pipeline] WARNING: No text extracted from {filename}")
        text = ""

    return {"file": filename, "text": text}


def save_json(result: dict):
    """Save OCR result to JSON file."""
    os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(result["file"])[0]
    output_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Pipeline] Saved: {output_path}")


def main():
    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"[Pipeline] ERROR: Directory not found: {RAW_IMAGES_DIR}")
        sys.exit(1)

    all_images = sorted([
        f for f in os.listdir(RAW_IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not all_images:
        print(f"[Pipeline] No images found in {RAW_IMAGES_DIR}.")
        sys.exit(0)

    # Split into already done vs pending
    pending = [f for f in all_images if not already_processed(f)]
    skipped = [f for f in all_images if already_processed(f)]

    print(f"\n[Pipeline] Total images found : {len(all_images)}")
    print(f"[Pipeline] Already processed  : {len(skipped)} (skipping)")
    print(f"[Pipeline] To process now     : {len(pending)}")

    if not pending:
        print("\n[Pipeline] Nothing new to process.")
        print("[Pipeline] Add more images to data\\raw_images\\ and run again.")
        print("[Pipeline] Or run: python scripts\\build_index.py  to rebuild the index.")
        sys.exit(0)

    succeeded = 0
    failed = []

    for i, filename in enumerate(pending):
        image_path = os.path.join(RAW_IMAGES_DIR, filename)
        try:
            result = process_single_image(image_path)
            save_json(result)
            succeeded += 1

            # Pause between Gemini calls to respect free-tier rate limits
            if i < len(pending) - 1:
                print(f"[Pipeline] Waiting {GEMINI_DELAY_SECONDS}s before next image...")
                time.sleep(GEMINI_DELAY_SECONDS)

        except Exception as e:
            error_msg = str(e)
            print(f"[Pipeline] ERROR on {filename}: {error_msg}")

            # Detect Gemini quota/rate limit errors and stop early
            if "quota" in error_msg.lower() or "rate" in error_msg.lower() or "429" in error_msg:
                print("\n[Pipeline] Gemini rate limit or quota hit.")
                print(f"[Pipeline] Processed {succeeded} image(s) this run.")
                remaining = pending[i:]
                print(f"[Pipeline] {len(remaining)} image(s) still pending: {remaining}")
                print("[Pipeline] Wait a minute and run the script again — already-done images will be skipped.")
                sys.exit(0)

            failed.append(filename)

    # Summary
    print(f"\n{'='*50}")
    print(f"[Pipeline] Run complete.")
    print(f"[Pipeline] Newly processed : {succeeded}")
    print(f"[Pipeline] Failed          : {len(failed)}")
    if failed:
        print(f"[Pipeline] Failed files    : {failed}")
    total_jsons = len(os.listdir(JSON_OUTPUT_DIR)) if os.path.exists(JSON_OUTPUT_DIR) else 0
    print(f"[Pipeline] Total JSONs now : {total_jsons}")
    print(f"\n[Pipeline] Next step: python scripts\\build_index.py")


if __name__ == "__main__":
    main()