#!/usr/bin/env python3
"""
Download training images for the AI demo
Downloads apple and banana images from Pexels (free stock photos)
"""

import os
import requests
import time
from pathlib import Path

# Pexels API (free tier - no API key needed for basic usage)
# Using direct image URLs from Pexels

APPLE_URLS = [
    "https://images.pexels.com/photos/102104/pexels-photo-102104.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/206959/pexels-photo-206959.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/209439/pexels-photo-209439.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510392/pexels-photo-1510392.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/568471/pexels-photo-568471.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510387/pexels-photo-1510387.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/672101/pexels-photo-672101.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1313643/pexels-photo-1313643.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/693794/pexels-photo-693794.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1510388/pexels-photo-1510388.jpeg?auto=compress&cs=tinysrgb&w=200",
]

BANANA_URLS = [
    "https://images.pexels.com/photos/61127/pexels-photo-61127.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2872755/pexels-photo-2872755.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5966630/pexels-photo-5966630.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2872767/pexels-photo-2872767.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2872428/pexels-photo-2872428.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/1093038/pexels-photo-1093038.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/5966631/pexels-photo-5966631.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2316466/pexels-photo-2316466.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/2316468/pexels-photo-2316468.jpeg?auto=compress&cs=tinysrgb&w=200",
    "https://images.pexels.com/photos/7210754/pexels-photo-7210754.jpeg?auto=compress&cs=tinysrgb&w=200",
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
    print("🍎🍌 Downloading training images...\n")

    # Create directories
    apple_dir = Path("data/apples")
    banana_dir = Path("data/bananas")
    apple_dir.mkdir(parents=True, exist_ok=True)
    banana_dir.mkdir(parents=True, exist_ok=True)

    # Download apples
    print("Downloading apples...")
    for i, url in enumerate(APPLE_URLS, 1):
        filepath = apple_dir / f"apple_{i}.jpg"
        download_image(url, filepath)
        time.sleep(0.5)  # Be nice to the server

    # Download bananas
    print("\nDownloading bananas...")
    for i, url in enumerate(BANANA_URLS, 1):
        filepath = banana_dir / f"banana_{i}.jpg"
        download_image(url, filepath)
        time.sleep(0.5)

    print("\n✅ Download complete!")
    print(f"   Apples: {len(list(apple_dir.glob('*.jpg')))}")
    print(f"   Bananas: {len(list(banana_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    download_dataset()
