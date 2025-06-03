# 🎬 Wan2 Video Generation with Avatar Integration

A serverless video generation service powered by the Wan2.1 AI model, designed for RunPod deployment with Supabase storage integration. This project enables high-quality text-to-video generation with customizable parameters and cloud storage capabilities.

[![RunPod](https://api.runpod.io/badge/gabrielelanzafamee/runpod-wan2.1-video-generation)](https://www.runpod.io/console/hub/gabrielelanzafamee/runpod-wan2.1-video-generation)

## ✨ Features

- **High-Quality Video Generation**: Uses Wan-AI/Wan2.1-T2V-14B-Diffusers model for professional 720p video output
- **Serverless Architecture**: Optimized for RunPod serverless deployment with minimal cold start times
- **Cloud Storage**: Automatic upload to Supabase storage with signed URL generation
- **Flexible Parameters**: Customizable video dimensions, duration, and generation settings
- **Docker Optimized**: Pre-built container with model caching for fast deployment
- **FastAPI Interface**: RESTful API for easy integration

## 🚀 Quick Start

### Prerequisites

- Docker with GPU support
- NVIDIA GPU with CUDA support
- RunPod account (for serverless deployment)
- Supabase account (for storage)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/gabrielelanzafamee/video_gen_ltx_avatar.git
   cd video_gen_ltx_avatar
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials
   ```

3. **Build and run locally**
   ```bash
   docker build -t video-gen-wan2 .
   docker run --gpus all -p 8000:8000 video-gen-wan2
   ```

4. **Test the API**
   ```bash
   curl -X POST "http://localhost:8000/generate-video" \
        -H "Content-Type: application/json" \
        -d '{
          "prompt": "A beautiful sunset over the ocean",
          "negative_prompt": "blurry, low quality"
        }'
   ```

## 🏗️ Architecture

### Model Loading Strategy

The project implements an optimized model loading strategy for serverless environments:

- **Build-time Caching**: Models are downloaded during Docker build to eliminate cold start delays
- **Lazy Loading**: Runtime model initialization only occurs on first request
- **Memory Optimization**: Global model instances are reused across function calls

### File Structure

```
├── handler.py         # RunPod serverless handler
├── app.py             # FastAPI development server
├── pipelines.py       # Video generation pipeline
├── init_model.py      # Model initialization script
├── Dockerfile         # Optimized container build
├── requirements.txt   # Python dependencies
└── .dockerignore      # Build optimization
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
BUCKET_ID=your_storage_bucket_name

# Optional: Model Testing
TEST_GENERATION=false
```

### API Parameters

#### RunPod Handler Input
```json
{
  "prompt": "Your video description",
  "negative_prompt": "What to avoid in the video",
  "width": 1280,
  "height": 720,
  "seconds": 5
}
```

#### FastAPI Endpoint Input
```json
{
  "prompt": "Your video description",
  "negative_prompt": "What to avoid in the video"
}
```

## 🚢 Deployment

### Docker Build Notes

**Important**: During Docker build, you'll see "CUDA not available" - this is **normal and expected**. The build process runs on CPU and only downloads/caches the model weights. GPU access is only available when the container runs on RunPod's GPU instances.

The build process will show:
```
⚠️ CUDA not available during build - this is normal for Docker build process
🔧 Model weights will be downloaded and cached for runtime use
```

### RunPod Serverless Deployment

1. **Build and push your Docker image**
   ```bash
   docker build -t your-registry/video-gen-wan2:latest .
   docker push your-registry/video-gen-wan2:latest
   ```

2. **Create RunPod Serverless Endpoint**
   - Go to RunPod Console → Serverless
   - Create new endpoint
   - Use your Docker image URL
   - Configure GPU requirements (recommend RTX 4090 or better)

3. **Set environment variables in RunPod**
   - Add your Supabase credentials
   - Configure storage bucket settings

### Local Testing

For development and testing:

```bash
# Run FastAPI server
python app.py

# Or use Docker
docker run --gpus all -p 8000:8000 video-gen-wan2 python app.py
```

## 📊 Performance

### Model Specifications

- **Model**: Wan-AI/Wan2.1-T2V-14B-Diffusers (14B parameters)
- **GPU Memory**: ~12-16GB VRAM required
- **Generation Time**: ~30-60 seconds for 5-second video (720p)
- **Output Format**: MP4 with 24 FPS at 720p resolution
- **Optimized for**: High-quality 720p video generation

### Optimization Features

- **Pre-cached Models**: Zero download time during inference
- **Efficient Memory Usage**: Optimized tensor operations for 14B parameter model
- **Batch Processing**: Single model instance handles multiple requests
- **Storage Integration**: Direct cloud upload without local storage

## 🛠️ Development

### Adding New Features

1. **Custom Pipeline**: Modify `pipelines.py` to add new generation parameters
2. **Storage Backends**: Extend `handler.py` for different cloud providers
3. **API Endpoints**: Add new routes in `app.py` for additional functionality

### Model Customization

To use a different model:

1. Update `model_id` in `pipelines.py` and `init_model.py`
2. Adjust model-specific parameters
3. Rebuild Docker image

### Testing

```bash
# Test model initialization
python init_model.py

# Test with sample generation
TEST_GENERATION=true python init_model.py

# API testing
python app.py
```

## 📋 Requirements

### System Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM
- **CUDA**: Version 12.4+ recommended
- **Docker**: With NVIDIA Container Toolkit
- **Python**: 3.11+

### Key Dependencies

- `diffusers`: Hugging Face diffusion models
- `torch`: PyTorch deep learning framework
- `runpod`: Serverless runtime
- `supabase`: Cloud storage client
- `fastapi`: Web framework for development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan AI](https://huggingface.co/Wan-AI) for the Wan2.1 video generation model
- [RunPod](https://runpod.io) for serverless GPU infrastructure
- [Supabase](https://supabase.com) for cloud storage solutions
- [Hugging Face](https://huggingface.co) for the diffusers library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/gabrielelanzafamee/video_gen_ltx_avatar/issues)
- **Discussions**: [GitHub Discussions](https://github.com/gabrielelanzafamee/video_gen_ltx_avatar/discussions)
- **Documentation**: Check the code comments and this README

---

**⚡ Ready to generate amazing 720p videos with Wan2.1? Deploy on RunPod and start creating!** 