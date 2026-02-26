#!/usr/bin/env python3
"""
Download more training images with extended Pexels URLs
"""

import os
import requests
import time
from pathlib import Path

# Extended list of apple images from Pexels
APPLE_URLS = [
    "https://images.pexels.com/photos/1132047/pexels-photo-1132047.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1300975/pexels-photo-1300975.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/39803/pexels-photo-39803.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/8966/pexels-photo.jpg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/142520/pexels-photo-142520.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510394/pexels-photo-1510394.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510389/pexels-photo-1510389.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/53362/apples-fruit-red-healthy-53362.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7195173/pexels-photo-7195173.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/209549/pexels-photo-209549.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/616833/pexels-photo-616833.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/347926/pexels-photo-347926.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/566888/pexels-photo-566888.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7195142/pexels-photo-7195142.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7110288/pexels-photo-7110288.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5340919/pexels-photo-5340919.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4919799/pexels-photo-4919799.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5340914/pexels-photo-5340914.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1132048/pexels-photo-1132048.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510390/pexels-photo-1510390.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7195159/pexels-photo-7195159.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4894387/pexels-photo-4894387.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7110312/pexels-photo-7110312.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1389104/pexels-photo-1389104.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5340905/pexels-photo-5340905.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/775818/pexels-photo-775818.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4226856/pexels-photo-4226856.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7110305/pexels-photo-7110305.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/616834/pexels-photo-616834.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/572916/pexels-photo-572916.jpeg?auto=compress&cs=tinysrgb&w=200",
]

# Extended list of banana images from Pexels
BANANA_URLS = [
    "https://images.pexels.com/photos/1093038/pexels-photo-1093038.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/3440682/pexels-photo-3440682.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198095/pexels-photo-4198095.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2238309/pexels-photo-2238309.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/3440691/pexels-photo-3440691.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050315/pexels-photo-4050315.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198091/pexels-photo-4198091.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2217032/pexels-photo-2217032.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050300/pexels-photo-4050300.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198093/pexels-photo-4198093.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051261/pexels-photo-4051261.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/327098/pexels-photo-327098.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050290/pexels-photo-4050290.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2316465/pexels-photo-2316465.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051268/pexels-photo-4051268.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5966621/pexels-photo-5966621.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050294/pexels-photo-4050294.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198094/pexels-photo-4198094.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198096/pexels-photo-4198096.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050311/pexels-photo-4050311.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051262/pexels-photo-4051262.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4198092/pexels-photo-4198092.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051271/pexels-photo-4051271.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050306/pexels-photo-4050306.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051267/pexels-photo-4051267.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050297/pexels-photo-4050297.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050312/pexels-photo-4050312.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051269/pexels-photo-4051269.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4050288/pexels-photo-4050288.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/4051263/pexels-photo-4051263.jpeg?auto=compress&cs=tinysrgb&w=200",
]

def download_image(url, filepath):
    """Download an image from URL to filepath"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"✓ Downloaded: {filepath.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filepath.name}: {e}")
        return False

def download_dataset():
    """Download all training images"""
    print("🍎🍌 Downloading extended training images...\n")

    # Create directories
    apple_dir = Path("data/apples")
    banana_dir = Path("data/bananas")

    # Get existing files count
    existing_apples = len(list(apple_dir.glob('*.jpg')))
    existing_bananas = len(list(banana_dir.glob('*.jpg')))

    print(f"Existing: {existing_apples} apples, {existing_bananas} bananas\n")

    # Download apples
    print("Downloading more apples...")
    apple_count = 0
    for i, url in enumerate(APPLE_URLS, existing_apples + 1):
        filepath = apple_dir / f"apple_{i}.jpg"
        if download_image(url, filepath):
            apple_count += 1
        time.sleep(0.5)

    # Download bananas
    print("\nDownloading more bananas...")
    banana_count = 0
    for i, url in enumerate(BANANA_URLS, existing_bananas + 1):
        filepath = banana_dir / f"banana_{i}.jpg"
        if download_image(url, filepath):
            banana_count += 1
        time.sleep(0.5)

    print(f"\n✅ Download complete!")
    print(f"   New apples: {apple_count}")
    print(f"   New bananas: {banana_count}")
    print(f"\n📊 Total dataset:")
    print(f"   Total apples: {len(list(apple_dir.glob('*.jpg')))}")
    print(f"   Total bananas: {len(list(banana_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    download_dataset()
