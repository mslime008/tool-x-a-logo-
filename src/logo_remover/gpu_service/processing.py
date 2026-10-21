"""Bounded temporal windows, disk-backed overlap blending, original pixels outside masks."""
from pathlib import Path
import shutil
import subprocess
import threading
import cv2
import numpy as np
from .config import ModelConfig
from .models import TemporalAdapter
from ..export import Cancelled, ExportSettings, export_video
from ..masks import Mask, active_mask, rasterize
from ..metadata import VideoMetadata, binary, frame_times


def temporal_windows(start: int, end: int, window: int, overlap: int):
    if not 0 <= start < end or not 1 <= overlap < window:
        raise ValueError("Invalid temporal window")
    cursor = start
    while True:
        stop = min(end, cursor + window)
        yield cursor, stop
        if stop == end:
            break
        cursor += window - overlap


def blend_weights(count: int) -> np.ndarray:
    return np.minimum(np.arange(1, count + 1), np.arange(count, 0, -1)).astype(np.float32)


def decode_frames(meta: VideoMetadata, root: Path, count: int, cancel: threading.Event):
    """Sequential raw decode; ffmpeg process belongs to worker's killable process group."""
    with (root / "decode.log").open("wb") as log:
        process = subprocess.Popen([binary("ffmpeg"), "-v", "error", "-threads", "2", "-filter_threads", "1",
            "-noautorotate", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov", "-enable_drefs", "0", "-i", meta.path, "-map", f"0:{meta.video_index}", "-fps_mode", "passthrough",
            "-frames:v", str(count + 1), "-threads", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
            stdout=subprocess.PIPE, stderr=log)
        try:
            size = meta.width * meta.height * 3
            for index in range(count):
                if cancel.is_set():
                    raise Cancelled("Cancelled")
                raw = process.stdout.read(size)
                if len(raw) != size:
                    raise ValueError("Decoded frame count differs from timestamps")
                frame = np.frombuffer(raw, np.uint8).reshape(meta.height, meta.width, 3)
                if not cv2.imwrite(str(root / f"source-{index:06d}.png"), frame,
                                   [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError("Cannot write decoded frame")
            if process.stdout.read(1):
                raise ValueError("Unexpected additional source frames")
            if process.wait(timeout=30):
                raise ValueError("FFmpeg decode failed")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


class TemporalProcessor:
    def __init__(self, root: Path, meta: VideoMetadata, masks: list[Mask], times: list[float],
                 config: ModelConfig, adapter: TemporalAdapter, cancel: threading.Event, progress):
        self.root, self.meta, self.masks, self.times = root, meta, masks, times
        self.config, self.adapter, self.cancel, self.progress = config, adapter, cancel, progress
        self.weights = np.zeros(len(times), np.float32)
        self.index = 0
        # A fixed source-space ROI across the job avoids window-to-window spatial shifts.
        union = np.zeros((meta.height, meta.width), np.uint8)
        for mask in masks:
            union |= rasterize(mask, meta.width, meta.height)
        x, y, w, h = cv2.boundingRect(union)
        margin = config.context_pixels
        self.x0, self.y0 = max(0, x - margin), max(0, y - margin)
        self.x1, self.y1 = min(meta.width, x + w + margin), min(meta.height, y + h + margin)
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        self.pad_w, self.pad_h = (-self.w) % config.alignment, (-self.h) % config.alignment
        if (self.w + self.pad_w) * (self.h + self.pad_h) > config.max_pixels:
            raise ValueError("Mask ROI exceeds GPU model pixel budget; choose a smaller region or larger GPU budget")

    def frame(self, index):
        frame = cv2.imread(str(self.root / f"source-{index:06d}.png"))
        if frame is None:
            raise ValueError("Missing decoded frame")
        return frame

    def scenes(self):
        boundaries, previous = [0], None
        for i in range(len(self.times)):
            frame = cv2.resize(self.frame(i), (96, 54), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if previous is not None and cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA) > self.config.scene_cut_threshold:
                boundaries.append(i)
            previous = hist
        boundaries.append(len(self.times))
        return list(zip(boundaries, boundaries[1:]))

    def run(self):
        windows = [pair for a, b in self.scenes()
                   for pair in temporal_windows(a, b, self.config.window, self.config.overlap)]
        for number, (start, end) in enumerate(windows):
            if self.cancel.is_set():
                raise Cancelled("Cancelled")
            frames, masks = [], []
            for index in range(start, end):
                frame = self.frame(index)[self.y0:self.y1, self.x0:self.x1]
                binary, _ = active_mask(self.masks, self.meta.width, self.meta.height, self.times[index])
                binary = binary[self.y0:self.y1, self.x0:self.x1]
                frames.append(cv2.copyMakeBorder(frame, 0, self.pad_h, 0, self.pad_w, cv2.BORDER_REFLECT_101))
                masks.append(cv2.copyMakeBorder(binary, 0, self.pad_h, 0, self.pad_w, cv2.BORDER_CONSTANT))
            if any(np.any(mask) for mask in masks):
                # Fixed-T exported models also support short tails and one-frame scenes.
                # Repeated boundary context is discarded after inference, never exported.
                while len(frames) < self.config.window:
                    frames.append(frames[-1])
                    masks.append(masks[-1])
                batch = np.stack(frames)
                prediction = self.adapter.infer(batch, np.stack(masks))
                if not isinstance(prediction, np.ndarray) or prediction.dtype != np.uint8 or prediction.shape != batch.shape:
                    raise ValueError("Adapter violated the uint8 temporal output contract")
                for offset, weight in enumerate(blend_weights(end - start)):
                    index = start + offset
                    if not np.any(masks[offset]):
                        continue
                    path = self.root / f"prediction-{index:06d}.npy"
                    shape = (self.h, self.w, 3)
                    accum = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+",
                                                    dtype=np.float32, shape=shape)
                    if self.weights[index] == 0:
                        accum[:] = 0
                    accum[:] += prediction[offset, :self.h, :self.w].astype(np.float32) * weight
                    accum.flush()
                    del accum
                    self.weights[index] += weight
            self.progress(int(15 + 55 * (number + 1) / len(windows)), "Temporal GPU inference")

    def __call__(self, original, masks, seconds):
        index = self.index
        self.index += 1
        if index >= len(self.times) or abs(seconds - self.times[index]) > 1e-6:
            raise ValueError("Temporal result does not match source timestamps")
        binary, alpha = active_mask(masks, self.meta.width, self.meta.height, seconds)
        if not np.any(binary):
            return original
        if self.weights[index] <= 0:
            raise ValueError("No prediction for an active frame")
        accum = np.load(self.root / f"prediction-{index:06d}.npy", mmap_mode="r")
        repaired = accum / self.weights[index]
        result = original.copy()
        target = result[self.y0:self.y1, self.x0:self.x1]
        weight = alpha[self.y0:self.y1, self.x0:self.x1, None]
        blended = np.rint(target * (1 - weight) + repaired * weight).clip(0, 255).astype(np.uint8)
        selected = binary[self.y0:self.y1, self.x0:self.x1] != 0
        target[selected] = blended[selected]
        return result


def process_video(meta: VideoMetadata, masks: list[Mask], settings: ExportSettings, output: Path,
                  config: ModelConfig, adapter: TemporalAdapter, work: Path,
                  cancel: threading.Event, progress, max_frames: int = 1800, max_work_bytes: int = 40 * 1024**3):
    times, _ = frame_times(meta, cancel, safe=True)
    if len(times) > max_frames:
        raise ValueError("Source exceeds maximum frame count")
    # Worst-case raw PNG + float ROI accumulators + export's second PNG set.
    estimate = len(times) * (meta.width * meta.height * 6 + config.max_pixels * 12) + 256 * 1024**2
    if estimate > max_work_bytes:
        raise OSError("Temporal processing exceeds configured workspace byte budget")
    if shutil.disk_usage(work).free < estimate:
        raise OSError("Insufficient temporary disk for temporal processing")
    processor = TemporalProcessor(work, meta, masks, times, config, adapter, cancel, progress)
    decode_frames(meta, work, len(times), cancel)
    progress(15, "Decoded source frames")
    processor.run()
    # Reuse the proven local FFmpeg PTS, SDR metadata and source-audio mux path unchanged.
    export_video(meta, masks, output, settings, cancel,
                 lambda n, text: progress(70 + int(n * .29), text), frame_processor=processor, safe_media=True)
