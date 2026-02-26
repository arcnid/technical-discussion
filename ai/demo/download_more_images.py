#!/usr/bin/env python3
"""
Download more training images using Unsplash
Unsplash provides free random images
"""

import os
import requests
import time
from pathlib import Path

def download_from_unsplash(query, output_dir, start_num, count=40):
    """Download images from Unsplash"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {count} {query} images from Unsplash...")

    successful = 0
    for i in range(count):
        try:
            # Unsplash Source API - random image for query
            # Adding a random seed to get different images
            url = f"https://source.unsplash.com/200x200/?{query}&sig={start_num + i}"

            response = requests.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()

            # Save image
            filename = f"{query}_{start_num + i}.jpg"
            filepath = output_dir / filename

            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"✓ Downloaded: {filename}")
            successful += 1

            # Be nice to the server
            time.sleep(1)

        except Exception as e:
            print(f"✗ Failed to download {query}_{start_num + i}: {e}")

    return successful

def main():
    print("🍎🍌 Downloading additional training images from Unsplash...\n")

    # Download more apples (starting from 11 since we have 10)
    apple_count = download_from_unsplash("apple", "data/apples", 11, 40)

    print()

    # Download more bananas
    banana_count = download_from_unsplash("banana", "data/bananas", 11, 40)

    print(f"\n✅ Download complete!")
    print(f"   New apples: {apple_count}")
    print(f"   New bananas: {banana_count}")

    # Count total
    apple_dir = Path("data/apples")
    banana_dir = Path("data/bananas")
    print(f"\n📊 Total dataset:")
    print(f"   Total apples: {len(list(apple_dir.glob('*.jpg')))}")
    print(f"   Total bananas: {len(list(banana_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    main()
