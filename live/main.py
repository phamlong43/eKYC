import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import schedule_service
from save_data import extract_video_embeddings

app2 = FastAPI(
    title="Python Backend API",
    description="Backend service để expose Python functions/models",
    version="1.0.0"
)

app2.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_video(input_path, output_path):

    try:
        # Lệnh FFmpeg để chuyển đổi video
        command = [
            'ffmpeg',
            '-i', input_path,  # File input
            '-c:v', 'libx264',  # Sử dụng codec H.264
            '-preset', 'medium',
            '-crf', '23',  # Chất lượng nén
            '-c:a', 'aac',  # Codec âm thanh
            '-b:a', '128k',  # Bitrate âm thanh
            output_path
        ]

        # Thực thi lệnh
        result = subprocess.run(command, capture_output=True, text=True)

        # Kiểm tra kết quả
        if result.returncode == 0:
            return output_path
        else:
            print("Lỗi chuyển đổi video:")
            print(result.stderr)
            return None

    except Exception as e:
        print(f"Lỗi khi chuyển đổi video: {e}")
        return None


@app2.post("/api/upload/video")
async def upload_video(
        video: UploadFile = File(...),
        request: str = Form(None)
):
    print("Received request")
    print(request)
    parsed_data = json.loads(request)
    user_id = parsed_data.get('userId')
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await video.read())
            temp_video_path = temp_video.name

        converted_video_path = temp_video_path.replace('.mp4', '_converted.mp4')

        # Chuyển đổi video
        converted_path = convert_video(temp_video_path, converted_video_path)

        if converted_path is None:
            return JSONResponse(content={
                "success": False,
                "error": "Không thể chuyển đổi video"
            }, status_code=400)

        video_embeddings = extract_video_embeddings(converted_path,user_id)
        os.unlink(converted_path)
        os.unlink(temp_video_path)
        if video_embeddings is not None:
            return JSONResponse(content={
                "success": True,
                "embeddings": video_embeddings.tolist()
            })
        else:
            return JSONResponse(content={
                "success": False,
                "error": "Không thể trích xuất embedding từ video"
            }, status_code=400)

    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": str(e)
        }, status_code=500)
@app2.get("/health")
async def health_check():
    return {"status": "healthy"}


class ImageResponse(BaseModel):
    path: str
    filename: str


@app2.get("/api/images/{year}/{month}/{day}", response_model=List[ImageResponse])
async def get_images(year: str, month: str, day: str):
    # Construct the directory path
    date_path = f"{day.zfill(2)}-{month.zfill(2)}-{year}"
    directory = Path(f"images/{date_path}")

    if not directory.exists():
        raise HTTPException(status_code=404, detail="No images found for this date")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    images = []

    for file in directory.iterdir():
        if file.suffix.lower() in valid_extensions:
            images.append(ImageResponse(
                path=f"/api/image/{date_path}/{file.name}",
                filename=file.name
            ))

    return images


@app2.get("/api/image/{date_path}/{filename}")
async def get_image(date_path: str, filename: str):
    file_path = Path(f"images/{date_path}/{filename}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)

class VideoInfo(BaseModel):
    filename: str
    path: str
    size: int
    date: str


@app2.get("/api/videos", response_model=List[VideoInfo])
async def get_videos():
    video_dir = Path("videos")

    if not video_dir.exists():
        raise HTTPException(status_code=404, detail="No videos found")

    videos = []
    for video_file in video_dir.glob("*.mp4"):
        # Parse date from filename (d-m-y.mp4)
        date_str = video_file.stem  # Remove .mp4 extension
        videos.append(VideoInfo(
            filename=video_file.name,
            path=f"/api/video/{video_file.name}",
            size=video_file.stat().st_size,
            date=date_str
        ))

    # Sort by date (newest first)
    return sorted(videos, key=lambda x: x.date, reverse=True)


@app2.get("/api/video/{filename}")
async def get_video(filename: str):
    video_path = Path("videos") / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=filename
    )

class TimeSchedule(BaseModel):
    start: Optional[str]=''
    late: Optional[str]=''
    end: Optional[str]

@app2.get("/schedule")
async def get_schedule():
    return schedule_service.get_schedule()

@app2.post("/schedule")
async def create_schedule(schedule: TimeSchedule):
    return schedule_service.create_schedule(schedule)

@app2.put("/schedule")
async def update_schedule(schedule: TimeSchedule):
    return schedule_service.update_schedule(schedule)

@app2.put("/schedule/time/end")
async def delete_schedule(schedule: TimeSchedule):
    return schedule_service.update_end_check_time(schedule)

@app2.get("/check/time")
async def get_check_time():
    state = schedule_service.check_attendance_status()
    return state["status"]

if __name__ == "__main__":
    uvicorn.run(app2, host="0.0.0.0", port=8000)