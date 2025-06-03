import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

# Global model instances - will be loaded once on first use
_pipe = None
_vae = None

def get_pipeline():
    """Lazy load the pipeline - loads once and reuses across calls"""
    global _pipe, _vae
    
    if _pipe is None:
        print("Loading model pipeline...")
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        _vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        _pipe = WanPipeline.from_pretrained(model_id, vae=_vae, torch_dtype=torch.bfloat16)
        _pipe.to("cuda")
        print("Model pipeline loaded successfully!")
    
    return _pipe

def generate_video_wan(prompt, negative_prompt, expected_height = 720 , expected_width = 1280, seconds = 5.0, video_path = "output.mp4"):
    # Get the pipeline (loads once, reuses afterwards)
    pipe = get_pipeline()
    
    fps = 24
    num_frames = int(fps * seconds) + 1
    
    print(f"Generating video with {num_frames} frames")
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=expected_height,
        width=expected_width,
        num_frames=num_frames,
        guidance_scale=5.0
    ).frames[0]
    
    print(f"Exporting video to {video_path}")
    
    export_to_video(output, video_path, fps=fps)
    
    print(f"Video exported to {video_path}")
    
    return video_path

