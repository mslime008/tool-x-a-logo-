"""Media time is seconds relative to the first displayed source frame."""
from bisect import bisect_right


def frame_index(timestamps: list[float], seconds: float) -> int:
    return max(0, min(len(timestamps) - 1, bisect_right(timestamps, seconds) - 1))


def active(start: float, end: float, seconds: float) -> bool:
    return start <= seconds < end
