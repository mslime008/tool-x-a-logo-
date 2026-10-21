import pytest
from logo_remover.metadata import MediaError, parse_metadata


def fixture():
    return {"streams": [
        {"codec_type": "video", "width": 3840, "height": 2160, "avg_frame_rate": "30000/1001",
         "time_base": "1/90000", "duration": "8.008", "sample_aspect_ratio": "1:1",
         "start_time": "0.125", "pix_fmt": "yuv420p10le", "color_transfer": "smpte2084",
         "color_primaries": "bt2020", "color_space": "bt2020nc",
         "side_data_list": [{"rotation": 90}]},
        {"codec_type": "audio", "codec_name": "aac"}]}


def test_hdr_rotation_rational_fps_audio():
    meta = parse_metadata(fixture(), "source.mov")
    assert meta.hdr and meta.rotation == 90
    assert meta.width == 3840 and meta.height == 2160
    assert meta.rate == pytest.approx(29.97002997)
    assert meta.start_time == .125 and meta.audio_codec == "aac"


def test_missing_video():
    with pytest.raises(MediaError):
        parse_metadata({"streams": []}, "invalid.mp4")


def test_duration_fallback_and_sdr():
    data = fixture()
    del data["streams"][0]["duration"]
    data["streams"][0]["color_transfer"] = "bt709"
    data["format"] = {"duration": "10.5"}
    meta = parse_metadata(data, "input.mp4")
    assert meta.duration == 10.5 and not meta.hdr
