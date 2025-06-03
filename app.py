import uvicorn
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import uuid
from pipelines import generate_video_wan

app = FastAPI()

def utils_generate_random_path():
    return f"output_{uuid.uuid4()}.mp4"

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str
    
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.post("/generate-video")
async def generate_video_endpoint(data: VideoGenerationRequest):
    try:
        video_path = generate_video_wan(data.prompt, data.negative_prompt, video_path=utils_generate_random_path())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")
    
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    
    return Response(content=video_bytes, media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)