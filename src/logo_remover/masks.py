from dataclasses import asdict, dataclass, field
import cv2
import numpy as np
from .timeline import active


@dataclass
class Mask:
    kind: str = "rectangle"
    points: list[tuple[float, float]] = field(default_factory=list)
    start: float = 0.0
    end: float = 999999.0
    padding: int = 3
    feather: int = 2
    brush_size: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        minimum = {"rectangle": 2, "polygon": 3, "brush": 1}
        if self.kind not in minimum or len(self.points) < minimum.get(self.kind, 1):
            raise ValueError("Mask chưa đủ điểm hoặc sai loại.")
        if not np.isfinite([self.start, self.end]).all() or not 0 <= self.start < self.end:
            raise ValueError("Khoảng thời gian mask không hợp lệ.")
        if not 0 <= self.padding <= 100 or not 0 <= self.feather <= 50 or not 1 <= self.brush_size <= 500:
            raise ValueError("Kích thước mask vượt giới hạn.")
        if not np.isfinite(np.asarray(self.points)).all():
            raise ValueError("Tọa độ mask không hợp lệ.")


def rasterize(mask: Mask, width: int, height: int) -> np.ndarray:
    mask.validate()
    image = np.zeros((height, width), np.uint8)
    points = np.rint(mask.points).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    if mask.kind == "rectangle":
        a, b = points[:2]
        cv2.rectangle(image, tuple(np.minimum(a, b)), tuple(np.maximum(a, b)), 255, -1)
    elif mask.kind == "polygon":
        cv2.fillPoly(image, [points], 255)
    else:
        for a, b in zip(points, points[1:]):
            cv2.line(image, tuple(a), tuple(b), 255, mask.brush_size)
        for p in points:
            cv2.circle(image, tuple(p), max(1, mask.brush_size // 2), 255, -1)
    if mask.padding:
        size = 2 * mask.padding + 1
        image = cv2.dilate(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return image


def active_mask(masks: list[Mask], width: int, height: int, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    union = np.zeros((height, width), np.uint8)
    alpha = np.zeros((height, width), np.float32)
    for mask in masks:
        if not active(mask.start, mask.end, seconds):
            continue
        binary = rasterize(mask, width, height)
        union |= binary
        if mask.feather:
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
            weight = np.minimum(distance / (mask.feather + 1), 1.0)
        else:
            weight = binary.astype(np.float32) / 255
        np.maximum(alpha, weight, out=alpha)
    return union, alpha
