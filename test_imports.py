#!/usr/bin/env python3

"""
Test script to verify that diffusers imports work without flash_attn compatibility issues
"""

import os
import sys

def test_imports():
    print("Testing diffusers imports...")
    
    try:
        print("1. Testing basic torch import...")
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA version: {torch.version.cuda}")
        
        print("\n2. Testing diffusers import...")
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.utils import export_to_video
        print("   ✓ Successfully imported AutoencoderKLWan, WanPipeline, export_to_video")
        
        print("\n3. Testing pipeline import...")
        from pipelines import generate_video_wan, get_memory_usage
        print("   ✓ Successfully imported pipeline functions")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 