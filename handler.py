import os

from dotenv import load_dotenv
load_dotenv()

import runpod
from pipelines import generate_video_wan
import uuid
from supabase import create_client

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
        video_path = generate_video_wan(prompt, negative_prompt, video_path=video_path, expected_width=width, expected_height=height, seconds=seconds)
    except Exception as e:
        print(f"Error generating video: {e}")
        raise Exception("Error generating video")
    
    supabase_storage_full_path = f"Avatars/{file_name}"
    
    # upload video to bucket
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
        
    # get public url
    public_url = supabase.storage.from_(bucket_id).create_signed_url(supabase_storage_full_path, expires_in=60 * 60 * 24, options={
        "download": True
    })
    
    return { "video_url": public_url }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})