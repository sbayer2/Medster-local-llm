#!/usr/bin/env python3
"""Quick test of vision analysis with a single DICOM image."""

import sys
sys.path.insert(0, 'src')

from medster.tools.analysis.primitives import scan_dicom_directory, get_dicom_metadata_from_path
from medster.utils.image_utils import dicom_to_base64_png
from medster.model import call_llm
from medster import config
from pathlib import Path

# Set vision model
config.set_selected_model("qwen3-vl:8b")

print("🔍 Scanning DICOM directory...")
dicom_files = scan_dicom_directory()
print(f"✓ Found {len(dicom_files)} DICOM files")

if not dicom_files:
    print("❌ No DICOM files found")
    sys.exit(1)

# Get first file
first_file = dicom_files[0]
print(f"\n📁 Selected: {first_file}")

# Get metadata
print("\n📊 Loading metadata...")
metadata = get_dicom_metadata_from_path(first_file)
print(f"   Modality: {metadata.get('modality', 'Unknown')}")
print(f"   Body Part: {metadata.get('body_part', 'Unknown')}")
print(f"   Dimensions: {metadata.get('dimensions', 'Unknown')}")
print(f"   Study Description: {metadata.get('study_description', 'Unknown')}")

# Load image directly from path (converts DICOM to base64 PNG)
print("\n🖼️  Converting DICOM to base64 PNG...")
try:
    image_base64 = dicom_to_base64_png(Path(first_file), target_size=(256, 256), quality=85)
    print(f"✓ Image converted ({len(image_base64)} bytes of base64 data)")
except Exception as e:
    print(f"❌ Failed to load image: {e}")
    sys.exit(1)

# Analyze with vision model
print("\n🔬 Analyzing with qwen3-vl:8b vision model...")
prompt = f"""Analyze this medical image (256x256 pixels, base64-encoded PNG).

Metadata:
- Modality: {metadata.get('modality', 'Unknown')}
- Body Part: {metadata.get('body_part', 'Unknown')}
- Study: {metadata.get('study_description', 'Unknown')}

Briefly describe what you see in the image. Focus on:
1. Image type (X-ray, MRI, CT, etc.)
2. Anatomical region visible
3. Any notable findings or features"""

response = call_llm(
    prompt=prompt,
    images=[image_base64],
    model="qwen3-vl:8b"
)

analysis = response.content if hasattr(response, 'content') else str(response)

print("\n" + "="*70)
print("VISION ANALYSIS RESULTS")
print("="*70)
print(analysis)
print("="*70)
