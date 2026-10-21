import cv2
import numpy as np
import pytest
from logo_remover.inpainting import inpaint
from logo_remover.masks import Mask, active_mask, rasterize
from logo_remover.timeline import frame_index


@pytest.mark.parametrize("kind,points", [
    ("rectangle", [(20, 20), (35, 35)]),
    ("polygon", [(20, 20), (40, 20), (30, 40)]),
    ("brush", [(20, 20), (35, 35)])])
def test_shapes_and_padding(kind, points):
    mask = Mask(kind, points, padding=0, feather=0, brush_size=5)
    raw = rasterize(mask, 80, 60)
    assert raw.dtype == np.uint8 and raw[25, 25] == 255
    mask.padding = 4
    padded = rasterize(mask, 80, 60)
    assert np.count_nonzero(padded) > np.count_nonzero(raw)
    assert np.all(padded[raw != 0] == 255)


@pytest.mark.parametrize("method", ["TELEA", "NS"])
def test_inpainting_changes_logo_only_inside_active_region(method):
    frame = np.full((80, 100, 3), 90, np.uint8)
    frame[25:40, 30:50] = 240
    mask = Mask(points=[(29, 24), (50, 40)], start=1, end=2, padding=2, feather=2)
    union, alpha = active_mask([mask], 100, 80, 1.5)
    result = inpaint(frame, [mask], 1.5, method)
    assert np.array_equal(result[union == 0], frame[union == 0])
    assert result[30, 35, 0] < 150
    assert np.all(alpha[union == 0] == 0)
    assert inpaint(frame, [mask], 0.5, method) is frame
    assert inpaint(frame, [mask], 2, method) is frame


def test_multiple_masks_time_and_union():
    masks = [Mask(points=[(3, 3), (9, 9)], start=0, end=1, padding=0),
             Mask(points=[(20, 20), (29, 29)], start=1, end=3, padding=0)]
    union, _ = active_mask(masks, 40, 40, 1)
    assert union[5, 5] == 0 and union[25, 25] == 255
    assert frame_index([0, .04, .10], .08) == 1


def test_reversed_clipped_rectangle_and_validation():
    raw = rasterize(Mask(points=[(12, 12), (-5, -5)], padding=0), 20, 20)
    assert raw[0, 0] == 255 and raw[12, 12] == 255
    with pytest.raises(ValueError):
        rasterize(Mask("polygon", [(0, 0)]), 20, 20)
    with pytest.raises(ValueError):
        Mask(points=[(0, 0), (2, 2)], start=2, end=1).validate()
    with pytest.raises(ValueError):
        inpaint(np.zeros((20, 20, 3), np.uint8), [Mask(points=[(0, 0), (19, 19)])], 0)
