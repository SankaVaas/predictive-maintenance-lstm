# data/download_data.py
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://data.nasa.gov/api/views/ff5v-kuh6/rows.csv?accessType=DOWNLOAD"

# The dataset is also mirrored reliably here:
MIRROR = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData/"

FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

for fname in FILES:
    dest = RAW_DIR / fname
    if dest.exists():
        print(f"  already exists: {fname}")
        continue
    url = MIRROR + fname
    print(f"  downloading {fname}...")
    urllib.request.urlretrieve(url, dest)

print("\nAll files downloaded to data/raw/")