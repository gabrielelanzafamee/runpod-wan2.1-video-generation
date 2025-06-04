import torch
import os

# Handle potential xformers/flash_attn compatibility issues
try:
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    XFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Import error with xformers/flash_attn: {e}")
    print("Attempting to import without xformers optimization...")
    
    # Disable xformers before importing diffusers
    os.environ["XFORMERS_DISABLED"] = "1"
    
    try:
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.utils import export_to_video
        XFORMERS_AVAILABLE = False
        print("Successfully imported diffusers without xformers")
    except ImportError as fallback_error:
        print(f"Failed to import even without xformers: {fallback_error}")
        raise

# Global model instances - will be loaded once on first use
_pipe = None
_vae = None

def get_pipeline():
    """Lazy load the pipeline - loads once and reuses across calls"""
    global _pipe, _vae
    
    if _pipe is None:
        print("Loading model pipeline...")
        
        # Use the larger model since we have plenty of VRAM
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        
        # Load with optimized settings
        print("Loading VAE...")
        _vae = AutoencoderKLWan.from_pretrained(
            model_id, 
            subfolder="vae", 
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True
        )
        
        print("Loading main pipeline...")
        _pipe = WanPipeline.from_pretrained(
            model_id, 
            vae=_vae, 
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,  # Disable safety checker for speed
            requires_safety_checker=False
        )
        
        # Move to GPU with optimizations
        _pipe = _pipe.to("cuda", dtype=torch.bfloat16)
        
        # Memory and performance optimizations for 80GB VRAM
        # With 80GB, we can afford to keep everything in memory
        _pipe.enable_model_cpu_offload(gpu_id=0)  # Smart offloading
        
        # Enable memory efficient attention (Flash Attention if available)
        if XFORMERS_AVAILABLE:
            try:
                _pipe.enable_xformers_memory_efficient_attention()
                print("✓ Enabled xFormers memory efficient attention")
            except ImportError:
                print("⚠ xFormers not available, using default attention")
            
        # Enable attention slicing for memory efficiency
        _pipe.enable_attention_slicing("auto")
        
        # Enable VAE slicing to reduce memory usage during encoding/decoding
        _pipe.enable_vae_slicing()
        
        # Compile the UNet for faster inference (requires PyTorch 2.0+)
        try:
            print("Compiling UNet for faster inference...")
            _pipe.unet = torch.compile(_pipe.unet, mode="reduce-overhead", fullgraph=True)
            print("✓ UNet compilation successful")
        except Exception as e:
            print(f"⚠ UNet compilation failed: {e}")
            
        # Compile VAE decoder for faster video export
        try:
            print("Compiling VAE decoder...")
            _pipe.vae.decoder = torch.compile(_pipe.vae.decoder, mode="reduce-overhead")
            print("✓ VAE decoder compilation successful")
        except Exception as e:
            print(f"⚠ VAE decoder compilation failed: {e}")
        
        print("Model pipeline loaded and optimized successfully!")
    
    return _pipe

def generate_video_wan(prompt, negative_prompt, expected_height=720, expected_width=1280, seconds=5.0, video_path="output.mp4"):
    """Optimized video generation with better performance settings"""
    # Get the pipeline (loads once, reuses afterwards)
    pipe = get_pipeline()
    
    fps = 24
    num_frames = int(fps * seconds) + 1
    
    print(f"Generating video with {num_frames} frames at {expected_width}x{expected_height}")
    
    # Optimize inference parameters for speed vs quality balance
    with torch.inference_mode():  # Faster than torch.no_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):  # Mixed precision
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=expected_height,
                width=expected_width,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=20,  # Reduced from default for speed
                generator=torch.Generator(device="cuda").manual_seed(42),  # For reproducibility
                max_sequence_length=256,  # Optimize sequence length
                output_type="pt"  # Keep tensors on GPU until export
            ).frames[0]
    
    print(f"Exporting video to {video_path}")
    
    # Export with optimized settings
    export_to_video(output, video_path, fps=fps)
    
    print(f"Video exported to {video_path}")
    
    # Clear CUDA cache periodically to prevent memory fragmentation
    if torch.cuda.memory_allocated() > 0.8 * torch.cuda.max_memory_allocated():
        torch.cuda.empty_cache()
    
    return video_path

def get_memory_usage():
    """Utility function to monitor VRAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2), 
            "max_allocated_gb": round(max_allocated, 2)
        }
    return None

