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
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  CUDA not available during build - this is normal for Docker build process")
            print("🔧 Model weights will be downloaded and cached for runtime use")
        
        # Model configuration
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        print(f"📦 Loading model: {model_id}")
        
        # Load VAE
        print("🔄 Loading VAE...")
        try:
            # Try to load with fp16 variant first
            vae = AutoencoderKLWan.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=torch.bfloat16,
                variant="fp16",
                use_safetensors=True
            )
            print("✅ VAE loaded with fp16 variant")
        except Exception as e:
            print(f"⚠️ Failed to load VAE with fp16 variant: {e}")
            print("🔄 Trying fallback: loading without variant...")
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    model_id, 
                    subfolder="vae", 
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                )
                print("✅ VAE loaded without variant")
            except Exception as e2:
                print(f"⚠️ Failed to load VAE with safetensors: {e2}")
                print("🔄 Trying final fallback: loading with default settings...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_id, 
                    subfolder="vae", 
                    torch_dtype=torch.bfloat16
                )
                print("✅ VAE loaded with default settings")
        
        # Load main pipeline
        print("🔄 Loading main pipeline...")
        try:
            # Try to load with fp16 variant first
            pipe = WanPipeline.from_pretrained(
                model_id, 
                vae=vae, 
                torch_dtype=torch.bfloat16,
                variant="fp16",
                use_safetensors=True
            )
            print("✅ Pipeline loaded with fp16 variant")
        except Exception as e:
            print(f"⚠️ Failed to load pipeline with fp16 variant: {e}")
            print("🔄 Trying fallback: loading without variant...")
            try:
                pipe = WanPipeline.from_pretrained(
                    model_id, 
                    vae=vae, 
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True
                )
                print("✅ Pipeline loaded without variant")
            except Exception as e2:
                print(f"⚠️ Failed to load pipeline with safetensors: {e2}")
                print("🔄 Trying final fallback: loading with default settings...")
                pipe = WanPipeline.from_pretrained(
                    model_id, 
                    vae=vae, 
                    torch_dtype=torch.bfloat16
                )
                print("✅ Pipeline loaded with default settings")
        
        # Only move to CUDA if available (runtime, not build time)
        if cuda_available:
            pipe.to("cuda")
            print("✅ Pipeline moved to CUDA")
        else:
            print("📋 Pipeline kept on CPU (will move to GPU at runtime)")
        
        load_time = time.time() - start_time
        print(f"🎉 Model initialization completed in {load_time:.2f} seconds")
        
        # Optional: Run a quick test generation (only if CUDA available)
        if os.getenv("TEST_GENERATION", "false").lower() == "true" and cuda_available:
            print("🧪 Running test generation...")
            test_generation(pipe)
        elif os.getenv("TEST_GENERATION", "false").lower() == "true":
            print("⏭️  Skipping test generation - no GPU available during build")
        
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