"""Disk-backed, sequential CPU pipeline; timestamps retained through concat demuxer."""
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
import cv2
import numpy as np
from .inpainting import inpaint
from .masks import Mask
from .metadata import MediaError, VideoMetadata, binary, frame_times


class Cancelled(RuntimeError):
    pass


@dataclass
class ExportSettings:
    codec: str = "h264"
    crf: int = 18
    preset: str = "medium"
    method: str = "TELEA"
    radius: int = 3

    def validate(self) -> None:
        if self.codec not in {"h264", "h265"} or not 0 <= self.crf <= 51:
            raise ValueError("Codec hoặc CRF không hợp lệ.")
        if self.preset not in {"fast", "medium", "slow"}:
            raise ValueError("Preset không hợp lệ.")
        if self.method not in {"TELEA", "NS"} or not 1 <= self.radius <= 20:
            raise ValueError("Thiết lập inpainting không hợp lệ.")


def _checked(command: list[str], cancel: threading.Event, log: Path) -> None:
    with log.open("wb") as handle:
        proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=handle)
        try:
            while True:
                if cancel.is_set():
                    raise Cancelled("Đã hủy tác vụ.")
                try:
                    code = proc.wait(timeout=0.15)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if code:
                raise MediaError(log.read_text(errors="replace")[-3000:])
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def read_frame(meta: VideoMetadata, seconds: float) -> np.ndarray:
    result = subprocess.run([binary("ffmpeg"), "-v", "error", "-threads", "2", "-filter_threads", "1", "-noautorotate", "-ss",
                             str(max(0, min(seconds, max(0, meta.duration - 1 / meta.rate)))),
                             "-i", meta.path, "-map", f"0:{meta.video_index}", "-frames:v", "1", "-threads", "1",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                            capture_output=True, timeout=60)
    size = meta.width * meta.height * 3
    if result.returncode or len(result.stdout) != size:
        raise MediaError(result.stderr.decode(errors="replace")[-1000:] or "Không đọc được khung hình.")
    return np.frombuffer(result.stdout, np.uint8).reshape(meta.height, meta.width, 3).copy()


def export_video(meta: VideoMetadata, masks: list[Mask], output: Path,
                 settings: ExportSettings, cancel: threading.Event | None = None,
                 progress: Callable[[int, str], None] = lambda *_: None,
                 *, frame_processor: Callable | None = None, safe_media: bool = False) -> Path:
    cancel = cancel or threading.Event()
    settings.validate()
    for mask in masks:
        mask.validate()
    output = Path(output).resolve()
    if output.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("Đầu ra phải là MP4 hoặc MOV.")
    if output == Path(meta.path).resolve() or output.exists():
        raise FileExistsError("Không ghi đè tệp gốc hoặc tệp đã tồn tại.")
    if meta.hdr:
        raise MediaError("Nguồn HDR: pipeline BGR 8 bit chưa bảo toàn màu HDR; hãy chuyển SDR có kiểm soát trước.")
    if meta.width % 2 or meta.height % 2:
        raise MediaError("Xuất yuv420p cần kích thước chẵn; không tự thay đổi kích thước nguồn.")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        times, durations = frame_times(meta, cancel, safe=safe_media)
    except MediaError:
        if cancel.is_set():
            raise Cancelled("Đã hủy tác vụ.") from None
        raise
    required = meta.width * meta.height * 3 * len(times) + 128 * 1024 ** 2
    if shutil.disk_usage(output.parent).free < required:
        raise MediaError(f"Thiếu dung lượng tạm; cần khoảng {required / 1024**3:.1f} GiB trống ở thư mục đầu ra.")
    ffmpeg = binary("ffmpeg")
    input_options = ["-protocol_whitelist", "file,pipe", "-format_whitelist", "mov", "-enable_drefs", "0"] if safe_media else []
    with tempfile.TemporaryDirectory(prefix="logo-remover-", dir=output.parent) as temp:
        root = Path(temp)
        log = root / "ffmpeg.log"
        done = threading.Event()
        with log.open("wb") as stderr:
            decoder = subprocess.Popen([ffmpeg, "-v", "error", "-threads", "2", "-filter_threads", "1", "-noautorotate", *input_options, "-i", meta.path,
                                        "-map", f"0:{meta.video_index}", "-fps_mode", "passthrough", "-threads", "1",
                                        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                                       stdout=subprocess.PIPE, stderr=stderr)
            def interrupt_decoder():
                while not done.wait(0.1):
                    if cancel.is_set():
                        if decoder.poll() is None:
                            decoder.kill()
                        return
            watcher = threading.Thread(target=interrupt_decoder, daemon=True)
            watcher.start()
            try:
                size = meta.width * meta.height * 3
                with (root / "frames.ffconcat").open("w", encoding="utf-8", newline="\n") as manifest:
                    manifest.write("ffconcat version 1.0\n")
                    for index, (seconds, duration) in enumerate(zip(times, durations)):
                        if cancel.is_set():
                            raise Cancelled("Đã hủy tác vụ.")
                        raw = decoder.stdout.read(size)
                        if len(raw) != size:
                            if cancel.is_set():
                                raise Cancelled("Đã hủy tác vụ.")
                            raise MediaError("Số khung giải mã không khớp PTS. " + log.read_text(errors="replace")[-1000:])
                        frame = np.frombuffer(raw, np.uint8).reshape(meta.height, meta.width, 3)
                        repaired = (frame_processor(frame, masks, seconds) if frame_processor is not None
                                    else inpaint(frame, masks, seconds, settings.method, settings.radius))
                        name = f"frame-{index:08d}.png"
                        if not cv2.imwrite(str(root / name), repaired, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                            raise MediaError("Không ghi được khung hình tạm.")
                        manifest.write(f"file {name}\noption framerate 1000000\nduration {duration:.9f}\n")
                        progress(int(75 * (index + 1) / len(times)), "Đang tái tạo khung hình")
                    # A terminal duplicate supplies the last duration to the concat demuxer.
                    manifest.write(f"file {name}\noption framerate 1000000\n")
                if decoder.stdout.read(1):
                    raise MediaError("Số khung giải mã lớn hơn danh sách PTS.")
                if decoder.wait(timeout=30):
                    raise MediaError(log.read_text(errors="replace")[-2000:])
            finally:
                done.set()
                if decoder.poll() is None:
                    decoder.kill()
                decoder.wait()
                decoder.stdout.close()
                watcher.join(timeout=1)
        if cancel.is_set():
            raise Cancelled("Đã hủy tác vụ.")
        progress(78, "Đang mã hóa bằng CPU")
        intermediate = root / "encoded.mp4"
        command = [ffmpeg, "-v", "error", "-nostdin", "-n", "-filter_threads", "1", "-threads", "1", "-safe", "0", "-f", "concat", "-i",
                   str(root / "frames.ffconcat"), "-an", "-frames:v", str(len(times)),
                   "-fps_mode", "vfr", "-enc_time_base", "1:1000000", "-c:v",
                   "libx264" if settings.codec == "h264" else "libx265", "-crf", str(settings.crf),
                   "-preset", settings.preset, "-threads", "2", "-bf", "0",
                   "-pix_fmt", "yuv420p", "-bsf:v",
                   f"setts=duration=if(eq(N\\,{len(times)-1})\\,{durations[-1]:.9f}/TB\\,DURATION)"]
        if settings.codec == "h265":
            command += ["-x265-params", "pools=2:frame-threads=2:fps=" + meta.fps, "-tag:v", "hvc1"]
        else:
            command += ["-x264-params", "fps=" + meta.fps]
        filters = []
        if meta.sar not in {"N/A", "0:1", "unknown"}:
            filters.append("setsar=" + meta.sar.replace(":", "/"))
        matrix = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601",
                  "smpte240m": "smpte240m", "fcc": "fcc", "bt2020nc": "bt2020"}.get(meta.color_space)
        scale = []
        if matrix:
            scale.append("out_color_matrix=" + matrix)
        if meta.color_range in {"pc", "tv"}:
            scale.append("out_range=" + meta.color_range)
        if scale:
            filters.append("scale=" + ":".join(scale))
        params = []
        for name, value in [("colorspace", meta.color_space), ("color_trc", meta.color_transfer),
                            ("color_primaries", meta.color_primaries)]:
            if value not in {"unknown", "unspecified", "reserved", "N/A"}:
                params.append(name + "=" + value)
        if meta.color_range in {"pc", "tv"}:
            params.append("range=" + ("full" if meta.color_range == "pc" else "limited"))
        if params:
            filters.append("setparams=" + ":".join(params))
        if filters:
            command += ["-vf", ",".join(filters)]
        command += ["-video_track_timescale", "1000000", str(intermediate)]
        _checked(command, cancel, log)
        progress(93, "Đang ghép âm thanh và metadata")
        staged = root / ("result" + output.suffix.lower())
        # Keep relative stream offsets: video starts at zero, shift source audio by video start.
        command = [ffmpeg, "-v", "error", "-nostdin", "-n", "-copyts", "-display_rotation", str(meta.rotation),
                   "-noautorotate", "-i", str(intermediate),
                   "-itsoffset", str(-meta.start_time), "-noautorotate", *input_options, "-i", meta.path,
                   "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_chapters", "-1",
                   "-c:v", "copy"]
        compatible = {"aac", "mp3", "alac", "ac3", "eac3"}
        audio_copy = meta.audio_codec in compatible or output.suffix.lower() == ".mov"
        command += ["-c:a", "copy" if audio_copy else "aac"]
        if not audio_copy and meta.audio_codec:
            progress(94, "Âm thanh không tương thích MP4: chuyển AAC 192 kb/s")
            command += ["-b:a", "192k"]
        command += ["-metadata:s:v:0", f"rotate={meta.rotation}", "-movflags", "+faststart",
                    "-video_track_timescale", "1000000", str(staged)]
        _checked(command, cancel, log)
        if cancel.is_set():
            raise Cancelled("Đã hủy tác vụ.")
        # Hard-link publication fails atomically if a competing job created the same output.
        os.link(staged, output)
        progress(100, "Hoàn tất")
    return output
