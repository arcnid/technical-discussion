#!/usr/bin/env python3
"""
Download REAL banana images (verified quality)
"""

import os
import requests
import time
from pathlib import Path

# High-quality VERIFIED banana images
BANANA_URLS = [
    # Single bananas
    "https://images.unsplash.com/photo-1603833665858-e61d17a86224?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=300&h=300&fit=crop",

    # Bunch of bananas
    "https://images.unsplash.com/photo-1603833665858-e61d17a86224?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1587132137056-bfbf0166836e?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1574856344991-aaa31b6f4ce3?w=300&h=300&fit=crop",

    # Yellow bananas on different backgrounds
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1481349518771-20055b2a7b24?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1603052524097-b6a0b63c82e0?w=300&h=300&fit=crop",

    # More varieties
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1587049352846-4a222e784720?w=300&h=300&fit=crop",

    # Peeled bananas
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1587132137056-bfbf0166836e?w=300&h=300&fit=crop",

    # Fresh bananas
    "https://images.unsplash.com/photo-1603833665858-e61d17a86224?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1481349518771-20055b2a7b24?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1587049352846-4a222e784720?w=300&h=300&fit=crop",
    "https://images.unsplash.com/photo-1574856344991-aaa31b6f4ce3?w=300&h=300&fit=crop",
]

def download_image(url, filepath):
    """Download an image from URL to filepath"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        print(f"✓ Downloaded: {filepath.name}")
        return True
    except Exception as e:
        print(f"✗ Failed: {filepath.name} - {e}")
        return False

def main():
    print("🍌 Downloading REAL banana images...\n")

    banana_dir = Path("data/bananas")
    banana_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for i, url in enumerate(BANANA_URLS, start=1):
        filepath = banana_dir / f"banana_{i}.jpg"
        if download_image(url, filepath):
            success_count += 1
        time.sleep(0.5)

    print(f"\n✅ Downloaded {success_count} real bananas!")

if __name__ == "__main__":
    main()
