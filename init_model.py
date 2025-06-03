#!/usr/bin/env python3
"""
Model initialization script for pre-downloading and testing the video generation model.
This can be run during Docker build or as a warm-up script.
"""

import torch
from diffusers import AutoencoderKLWan, WanPipeline
import os
import time

def init_model():
    """Initialize and cache the model"""
    print("🚀 Starting model initialization...")
    start_time = time.time()
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available, using CPU")
        
        # Model configuration
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        print(f"📦 Loading model: {model_id}")
        
        # Load VAE
        print("🔄 Loading VAE...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        print("✅ VAE loaded successfully")
        
        # Load main pipeline
        print("🔄 Loading main pipeline...")
        pipe = WanPipeline.from_pretrained(
            model_id, 
            vae=vae, 
            torch_dtype=torch.bfloat16
        )
        
        if torch.cuda.is_available():
            pipe.to("cuda")
            print("✅ Pipeline moved to CUDA")
        
        load_time = time.time() - start_time
        print(f"🎉 Model initialization completed in {load_time:.2f} seconds")
        
        # Optional: Run a quick test generation
        if os.getenv("TEST_GENERATION", "false").lower() == "true":
            print("🧪 Running test generation...")
            test_generation(pipe)
        
        return pipe, vae
        
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        raise

def test_generation(pipe):
    """Run a quick test generation to ensure everything works"""
    try:
        print("🔄 Generating test video...")
        output = pipe(
            prompt="A beautiful sunset over the ocean",
            negative_prompt="blurry, low quality",
            height=360,  # Smaller for faster test
            width=640,
            num_frames=25,  # ~1 second at 24fps
            guidance_scale=5.0
        ).frames[0]
        print("✅ Test generation successful!")
        
    except Exception as e:
        print(f"⚠️  Test generation failed: {e}")

if __name__ == "__main__":
    init_model() 