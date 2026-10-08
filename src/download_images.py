import os
import subprocess
import pandas as pd
import time

DATASET = "kovalevvyu/painter-by-numbers-resized"
METADATA = "data/processed/artist_metadata.csv"
OUTPUT_DIR = "data/raw/images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(METADATA)
filenames = df["new_filename"].dropna().unique()
total = len(filenames)
downloaded = 0
skipped = 0
failed = []

print(f"Total images required: {total}")
print(f"Download folder: {OUTPUT_DIR}")
print("-" * 60)

for i, filename in enumerate(filenames, start=1):

    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        skipped += 1
        print(f"[{i}/{total}] Already exists: {filename}")
        continue

    print(f"[{i}/{total}] Downloading: {filename}")

    command = [
        "kaggle",
        "datasets",
        "download",
        DATASET,
        "-f",
        filename,
        "-p",
        OUTPUT_DIR,
        "--quiet",
    ]

    success = False

    for attempt in range(3):

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and os.path.exists(output_path):
                downloaded += 1
                success = True
                print(f"  Downloaded successfully.")
                break

            print(f"  Attempt {attempt + 1} failed.")

        except subprocess.TimeoutExpired:
            print(f"  Attempt {attempt + 1} timed out.")

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(2)

    if not success:
        failed.append(filename)
        print(f"  FAILED: {filename}")

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(f"Required images : {total}")
print(f"Downloaded      : {downloaded}")
print(f"Already existed : {skipped}")
print(f"Failed          : {len(failed)}")

if failed:
    pd.DataFrame({"filename": failed}).to_csv(
        "data/processed/failed_downloads.csv",
        index=False
    )
    print("\nFailed filenames saved to:")
    print("data/processed/failed_downloads.csv")
else:
    print("\nAll images downloaded successfully!")