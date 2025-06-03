import os

from dotenv import load_dotenv
load_dotenv()

import runpod
from pipelines import generate_video_wan, get_memory_usage
import uuid
from supabase import create_client
import time

# supabase client
try:
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )
except Exception as e:
    print(f"Error creating supabase client: {e}")
    supabase = None

def utils_generate_random_path():
    return f"output_{uuid.uuid4()}.mp4"

async def handler(job):
    input = job["input"]
    
    # Log initial memory usage
    memory_stats = get_memory_usage()
    if memory_stats:
        print(f"Initial VRAM usage: {memory_stats}")
    
    start_time = time.time()
    
    try:
        prompt = input["prompt"]
        negative_prompt = input["negative_prompt"]
        width = int(input["width"])
        height = int(input["height"])
        seconds = int(input["seconds"])
    except Exception as e:
        print(f"Error parsing input: {e}")
        raise Exception("Error parsing input")
    
    
    if supabase is None:
        raise Exception("Supabase client not created")
    
    # upload video to bucket
    bucket_id = os.getenv("BUCKET_ID", "default")
    file_name = utils_generate_random_path()
    video_path = f"/tmp/{file_name}"
    
    # build video
    try:
        print(f"Starting video generation...")
        generation_start = time.time()
        
        video_path = generate_video_wan(prompt, negative_prompt, video_path=video_path, expected_width=width, expected_height=height, seconds=seconds)
        
        generation_time = time.time() - generation_start
        print(f"Video generation completed in {generation_time:.2f} seconds")
        
        # Log memory usage after generation
        memory_stats = get_memory_usage()
        if memory_stats:
            print(f"Post-generation VRAM usage: {memory_stats}")
            
    except Exception as e:
        print(f"Error generating video: {e}")
        raise Exception("Error generating video")
    
    supabase_storage_full_path = f"Avatars/{file_name}"
    
    # upload video to bucket
    upload_start = time.time()
    with open(video_path, "rb") as video_file:
        supabase.storage.from_(bucket_id).upload(
            path=supabase_storage_full_path,
            file=video_file,
            file_options={
                "upsert": True,
                "cache-control": "3600",
                "content-type": "video/mp4"
            }
        )
    upload_time = time.time() - upload_start
    print(f"Video uploaded in {upload_time:.2f} seconds")
        
    # get public url
    public_url = supabase.storage.from_(bucket_id).create_signed_url(supabase_storage_full_path, expires_in=60 * 60 * 24, options={
        "download": True
    })
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    return { 
        "video_url": public_url,
        "processing_time": round(total_time, 2),
        "generation_time": round(generation_time, 2) if 'generation_time' in locals() else None,
        "memory_stats": get_memory_usage()
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})