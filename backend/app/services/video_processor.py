"""Video processing service using FFmpeg."""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

from backend.app.config import settings


class VideoProcessor:
    """Handles video processing operations."""

    @staticmethod
    async def get_video_info(video_path: Path) -> dict:
        """Get video metadata using FFprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {stderr.decode()}")

        data = json.loads(stdout.decode())

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("No video stream found")

        # Parse FPS
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            fps = num / den if den else 30.0
        else:
            fps = float(fps_str)

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "duration": float(data.get("format", {}).get("duration", 0)),
            "total_frames": int(video_stream.get("nb_frames", 0)) or int(fps * float(data.get("format", {}).get("duration", 0))),
            "codec": video_stream.get("codec_name", "unknown"),
        }

    @staticmethod
    async def create_proxy_video(
        input_path: Path,
        output_path: Path,
        scale: float = 0.5,
        crf: int = 28,
    ) -> Path:
        """Create a lower-resolution proxy video for smooth playback."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", f"scale=iw*{scale}:ih*{scale}",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        if process.returncode != 0:
            logger.warning(f"Proxy creation failed, using original video")
            return input_path

        return output_path

    @staticmethod
    async def extract_frames(
        video_path: Path,
        output_dir: Path,
        mode: str = "seconds",
        interval: float = 0.5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> list[dict]:
        """
        Extract frames from video at specified intervals.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            mode: 'seconds' or 'frames'
            interval: Interval between frames (seconds or frame count)
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)

        Returns:
            List of extracted frame info dictionaries
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video info
        info = await VideoProcessor.get_video_info(video_path)
        fps = info["fps"]
        duration = info["duration"]

        # Calculate frame numbers to extract
        if mode == "seconds":
            frame_interval = int(fps * interval)
        else:
            frame_interval = int(interval)

        frame_interval = max(1, frame_interval)

        start_frame = int((start_time or 0) * fps)
        end_frame = int((end_time or duration) * fps)
        total_frames = info["total_frames"]
        end_frame = min(end_frame, total_frames)

        frames_to_extract = list(range(start_frame, end_frame, frame_interval))

        logger.info(
            f"Extracting {len(frames_to_extract)} frames from {video_path.name} "
            f"(interval: {interval} {mode})"
        )

        # Use OpenCV for frame extraction (more precise than FFmpeg for specific frames)
        cap = cv2.VideoCapture(str(video_path))
        extracted_frames = []

        try:
            for frame_num in frames_to_extract:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()

                if not ret:
                    continue

                timestamp = frame_num / fps
                frame_filename = f"frame_{frame_num:08d}.jpg"
                frame_path = output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                extracted_frames.append({
                    "frame_number": frame_num,
                    "timestamp": timestamp,
                    "image_path": str(frame_path),
                })

        finally:
            cap.release()

        logger.info(f"Extracted {len(extracted_frames)} frames")
        return extracted_frames

    @staticmethod
    def get_frame(video_path: Path, frame_number: int) -> Optional[bytes]:
        """Get a single frame from video as JPEG bytes."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                return None

            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes()
        finally:
            cap.release()

    @staticmethod
    async def extract_single_frame(
        video_path: Path,
        frame_number: int,
        output_path: Path,
    ) -> Path:
        """Extract a single frame to a file."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_number}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return output_path
        finally:
            cap.release()

