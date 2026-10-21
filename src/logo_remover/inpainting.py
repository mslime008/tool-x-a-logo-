import cv2
import numpy as np
from .masks import Mask, active_mask


def inpaint(frame: np.ndarray, masks: list[Mask], seconds: float,
            method: str = "TELEA", radius: float = 3) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Cần khung hình BGR 8 bit.")
    if method not in {"TELEA", "NS"} or not 1 <= radius <= 20:
        raise ValueError("Thuật toán hoặc bán kính không hợp lệ.")
    height, width = frame.shape[:2]
    mask, alpha = active_mask(masks, width, height, seconds)
    if not np.any(mask):
        return frame
    if np.all(mask):
        raise ValueError("Mask phủ toàn bộ khung hình; không còn vùng ảnh để tái tạo.")
    x, y, w, h = cv2.boundingRect(mask)
    margin = max(32, int(radius * 3))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
    crop = frame[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    repaired = cv2.inpaint(crop, local_mask, radius,
                          cv2.INPAINT_TELEA if method == "TELEA" else cv2.INPAINT_NS)
    result = frame.copy()
    weights = alpha[y0:y1, x0:x1, None]
    blended = np.rint(crop * (1 - weights) + repaired * weights).clip(0, 255).astype(np.uint8)
    region = result[y0:y1, x0:x1]
    region[local_mask != 0] = blended[local_mask != 0]
    return result
