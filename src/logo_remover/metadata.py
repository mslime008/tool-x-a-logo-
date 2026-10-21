from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import time
from threading import Event


class MediaError(RuntimeError):
    pass


def binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(f"Không tìm thấy {name}. Hãy cài FFmpeg và thêm vào PATH.")
    return found


def run_json(args: list[str], cancel: Event | None = None) -> dict:
    proc = subprocess.Popen([binary("ffprobe"), "-v", "error", *args, "-of", "json"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 120
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise MediaError("Đã hủy đọc mốc thời gian.")
            if time.monotonic() >= deadline:
                raise MediaError("ffprobe vượt thời gian chờ 120 giây.")
            try:
                stdout, stderr = proc.communicate(timeout=.15)
                break
            except subprocess.TimeoutExpired:
                continue
        if proc.returncode:
            raise MediaError(stderr.decode(errors="replace")[-2000:])
        return json.loads(stdout)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    video_index: int
    width: int
    height: int
    duration: float
    fps: str
    time_base: str
    start_time: float
    rotation: int
    sar: str
    pix_fmt: str
    color_space: str
    color_transfer: str
    color_primaries: str
    color_range: str
    audio_codec: str | None
    hdr: bool

    @property
    def rate(self) -> float:
        try:
            return float(Fraction(self.fps)) or 30.0
        except (ValueError, ZeroDivisionError):
            return 30.0


def parse_metadata(data: dict, path: str) -> VideoMetadata:
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MediaError("Tệp không có luồng video hợp lệ.")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    rotation = int(float(video.get("tags", {}).get("rotate", 0)))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    transfer = video.get("color_transfer", "unknown")
    fmt = video.get("pix_fmt", "unknown")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0 or int(video.get("width", 0)) <= 0 or int(video.get("height", 0)) <= 0:
        raise MediaError("Không đọc được kích thước hoặc thời lượng video.")
    return VideoMetadata(
        path=str(Path(path).resolve()), video_index=int(video.get("index", 0)),
        width=int(video["width"]), height=int(video["height"]),
        duration=duration, fps=video.get("avg_frame_rate", "30/1"),
        time_base=video.get("time_base", "1/1000"), start_time=float(video.get("start_time") or 0),
        rotation=rotation, sar=video.get("sample_aspect_ratio", "1:1"), pix_fmt=fmt,
        color_space=video.get("color_space", "unknown"), color_transfer=transfer,
        color_primaries=video.get("color_primaries", "unknown"),
        color_range=video.get("color_range", "unknown"), audio_codec=audio.get("codec_name"),
        hdr=transfer in {"smpte2084", "arib-std-b67"})


def probe(path: str | Path, *, safe: bool = False) -> VideoMetadata:
    p = Path(path).resolve()
    if p.suffix.lower() not in {".mp4", ".mov"}:
        raise MediaError("Chỉ hỗ trợ đầu vào MP4 và MOV.")
    options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", "mov", "-enable_drefs", "0"] if safe else []
    if safe:
        options += ["-show_entries", "stream=index,codec_name,codec_type,width,height,duration,avg_frame_rate,time_base,start_time,sample_aspect_ratio,pix_fmt,color_space,color_transfer,color_primaries,color_range:stream_tags=rotate:stream_disposition=attached_pic:stream_side_data=rotation:format=duration,format_name"]
    return parse_metadata(run_json([*options, "-show_streams", "-show_format", str(p)]), str(p))


def frame_times(meta: VideoMetadata, cancel: Event | None = None, *, safe: bool = False) -> tuple[list[float], list[float]]:
    options = (["-protocol_whitelist", "file,pipe", "-format_whitelist", "mov", "-enable_drefs", "0",
                "-read_intervals", "%+#10000"] if safe else [])
    data = run_json([*options, "-select_streams", str(meta.video_index), "-show_frames", "-show_entries",
                     "frame=best_effort_timestamp_time,pkt_duration_time,duration_time", meta.path], cancel)
    frames = data.get("frames", [])
    if not frames:
        raise MediaError("Không đọc được mốc thời gian khung hình.")
    if any("best_effort_timestamp_time" not in f for f in frames):
        raise MediaError("Video thiếu PTS; không thể đảm bảo đồng bộ thời gian.")
    origin = float(frames[0]["best_effort_timestamp_time"])
    times = [float(f["best_effort_timestamp_time"]) - origin for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise MediaError("PTS không tăng; cần sửa timestamp nguồn trước khi xử lý.")
    durations = [b - a for a, b in zip(times, times[1:])]
    last = frames[-1]
    durations.append(float(last.get("duration_time") or last.get("pkt_duration_time")
                           or max(1 / meta.rate, meta.duration - times[-1])))
    return times, durations
