import os
import zipfile
import pandas as pd

ZIP_PATH = "data/raw/painter-by-numbers-resized.zip"
METADATA_PATH = "data/processed/artist_metadata.csv"
OUTPUT_DIR = "data/raw/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the 7,462 images required for our project
metadata = pd.read_csv(METADATA_PATH)
required_files = set(
    metadata["new_filename"]
    .dropna()
    .astype(str)
)

print(f"Required images: {len(required_files)}")
print(f"Output folder: {OUTPUT_DIR}")
print("-" * 60)

extracted = 0
already_exists = 0
not_found = []

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    # Get all files inside the ZIP
    zip_files = set(zip_file.namelist())

    for filename in sorted(required_files):

        output_path = os.path.join(OUTPUT_DIR, filename)

        # Don't extract something we already have
        if os.path.exists(output_path):
            already_exists += 1
            continue

        # Check whether the required image exists in the ZIP
        if filename not in zip_files:
            not_found.append(filename)
            continue

        # Extract only this image
        zip_file.extract(filename, OUTPUT_DIR)
        extracted += 1

        if extracted % 500 == 0:
            print(f"Extracted: {extracted}/{len(required_files)}")

print("\n" + "=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)
print(f"Required images : {len(required_files)}")
print(f"Extracted       : {extracted}")
print(f"Already existed : {already_exists}")
print(f"Not found       : {len(not_found)}")

if not_found:
    pd.DataFrame(
        {"filename": not_found}
    ).to_csv(
        "data/processed/not_found_images.csv",
        index=False
    )
    print("\nMissing filenames saved to:")
    print("data/processed/not_found_images.csv")
else:
    print("\nAll required images were found in the ZIP.")