#!/usr/bin/env python3
"""
Download high-quality apple images - different varieties
"""

import os
import requests
import time
from pathlib import Path

# High-quality apple images from Unsplash (free to use)
# Different varieties: Red Delicious, Granny Smith, Gala, Cosmic Crisp, Honeycrisp
APPLE_URLS = [
    # Red apples
    "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=300&h=300&fit=crop",  # Red apple
    "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=300&h=300&fit=crop",  # Red apple close-up
    "https://images.unsplash.com/photo-1584306670957-acf935f5033c?w=300&h=300&fit=crop",  # Red apple with leaf
    "https://images.unsplash.com/photo-1619546952812-520e98064a52?w=300&h=300&fit=crop",  # Multiple red apples
    "https://images.unsplash.com/photo-1590005354167-6da97870c757?w=300&h=300&fit=crop",  # Single red apple

    # Granny Smith (green apples)
    "https://images.unsplash.com/photo-1579613832111-ac7dfcc7723f?w=300&h=300&fit=crop",  # Green apple
    "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=300&h=300&fit=crop",  # Green apples
    "https://images.unsplash.com/photo-1576179635662-9d1983e97e1e?w=300&h=300&fit=crop",  # Green apple close

    # Yellow/Golden apples
    "https://images.unsplash.com/photo-1567722500458-7f7e8b4c4b5c?w=300&h=300&fit=crop",  # Yellow apple
    "https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?w=300&h=300&fit=crop",  # Golden delicious

    # Mixed varieties
    "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?w=300&h=300&fit=crop",  # Apple variety
    "https://images.unsplash.com/photo-1568702846914-96b305d2aaeb?w=300&h=300&fit=crop",  # Gala apple
    "https://images.unsplash.com/photo-1559181567-c3190ca9959b?w=300&h=300&fit=crop",  # Apple on tree
    "https://images.unsplash.com/photo-1587049352846-4a222e784720?w=300&h=300&fit=crop",  # Fresh apple
    "https://images.unsplash.com/photo-1591952991274-23b1e187cedb?w=300&h=300&fit=crop",  # Apple basket

    # More red varieties
    "https://images.unsplash.com/photo-1609501676725-7186f017a4b7?w=300&h=300&fit=crop",  # Red delicious
    "https://images.unsplash.com/photo-1594621604779-1f9e79f9c3fb?w=300&h=300&fit=crop",  # Honeycrisp
    "https://images.unsplash.com/photo-1576677215335-384b25c72f6d?w=300&h=300&fit=crop",  # Cosmic crisp
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
        print(f"✗ Failed to download {filepath.name}: {e}")
        return False

def main():
    """Download all apple images"""
    print("🍎 Downloading quality apple images (different varieties)...\n")

    apple_dir = Path("data/apples")
    apple_dir.mkdir(parents=True, exist_ok=True)

    # Start numbering from 40 to avoid conflicts
    success_count = 0
    for i, url in enumerate(APPLE_URLS, start=40):
        filepath = apple_dir / f"apple_{i}.jpg"
        if download_image(url, filepath):
            success_count += 1
        time.sleep(0.5)  # Be respectful to servers

    print(f"\n✅ Download complete!")
    print(f"   Downloaded: {success_count} new apples")
    print(f"   Total apples: {len(list(apple_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    main()
