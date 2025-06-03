# Optimization configuration for 80GB VRAM inference
import os

class InferenceConfig:
    """Configuration for optimized video generation inference"""
    
    # Model settings
    MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"  # Use the larger 14B model with 80GB VRAM
    TORCH_DTYPE = "bfloat16"  # Most efficient for modern GPUs
    
    # Performance settings
    ENABLE_XFORMERS = True
    ENABLE_TORCH_COMPILE = True
    ENABLE_ATTENTION_SLICING = True
    ENABLE_VAE_SLICING = True
    ENABLE_MODEL_CPU_OFFLOAD = True
    
    # CUDA optimizations
    ALLOW_TF32 = True
    CUDNN_BENCHMARK = True
    CUDNN_DETERMINISTIC = False
    
    # Inference parameters
    DEFAULT_NUM_INFERENCE_STEPS = 20  # Reduced for speed (default is usually 50)
    DEFAULT_GUIDANCE_SCALE = 5.0
    MAX_SEQUENCE_LENGTH = 256
    
    # Memory management
    MEMORY_EFFICIENT_ATTENTION = True
    GRADIENT_CHECKPOINTING = False  # Disable since we're not training
    CLEAR_CUDA_CACHE_THRESHOLD = 0.8  # Clear cache when 80% of max memory is used
    
    # Video generation defaults
    DEFAULT_FPS = 24
    DEFAULT_WIDTH = 1280
    DEFAULT_HEIGHT = 720
    DEFAULT_SECONDS = 5.0
    
    # Batch processing (for multiple video generation)
    MAX_BATCH_SIZE = 1  # Conservative for video generation
    
    @staticmethod
    def get_cuda_memory_config():
        """Get CUDA memory configuration for 80GB VRAM"""
        return {
            "max_split_size_mb": 512,
            "expandable_segments": True,
            "memory_fraction": 0.95  # Use 95% of available VRAM
        }
    
    @staticmethod
    def get_torch_compile_config():
        """Get optimal torch.compile configuration"""
        return {
            "mode": "reduce-overhead",  # Best for inference
            "fullgraph": True,
            "dynamic": False
        } 