import hashlib
from pathlib import Path
import shutil
import subprocess
import threading
import cv2
import numpy as np
import pytest
from logo_remover.export import Cancelled, ExportSettings, export_video, read_frame
from logo_remover.masks import Mask
from logo_remover.metadata import frame_times, probe, run_json

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)], check=True,
                   capture_output=True, timeout=120)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("moving-logo")
    path = root / "source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=8",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
           "-vf", "drawbox=x=225:y=15:w=70:h=28:color=white:t=fill,drawbox=x=235:y=22:w=50:h=14:color=red:t=fill,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
           "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "16",
           "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
           "-c:a", "aac", "-shortest", path)
    return path


def audio_hash(path):
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
                            "-c", "copy", "-f", "hash", "-hash", "sha256", "-"],
                            check=True, capture_output=True)
    return result.stdout


@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_synthetic_eight_second_export(synthetic, tmp_path, codec):
    source_hash = hashlib.sha256(synthetic.read_bytes()).hexdigest()
    meta = probe(synthetic)
    mask = Mask(points=[(223, 13), (297, 45)], end=4, padding=2, feather=1)
    output = tmp_path / f"result-{codec}.mp4"
    updates = []
    export_video(meta, [mask], output, ExportSettings(codec=codec, preset="fast"),
                 progress=lambda n, text: updates.append(n))
    actual = probe(output)
    assert (actual.width, actual.height) == (320, 180)
    assert actual.audio_codec == "aac"
    assert actual.color_primaries == "bt709" and actual.color_transfer == "bt709"
    source_times, source_durations = frame_times(meta)
    result_times, result_durations = frame_times(actual)
    assert len(result_times) == len(source_times) == 96
    assert np.allclose(source_times, result_times, atol=.00001)
    assert abs(actual.duration - meta.duration) < .09
    assert updates[-1] == 100
    assert audio_hash(output) == audio_hash(synthetic)
    assert hashlib.sha256(synthetic.read_bytes()).hexdigest() == source_hash
    before, after = read_frame(meta, 1), read_frame(actual, 1)
    assert np.mean(cv2.absdiff(before[20:40, 230:290], after[20:40, 230:290])) > 10
    # Outside mask is encoded with lossy compression; test an error bound, not bit equality.
    assert np.mean(cv2.absdiff(before[70:], after[70:])) < 12
    assert not list(tmp_path.glob("logo-remover-*"))
    with pytest.raises(FileExistsError):
        export_video(meta, [mask], output, ExportSettings())


def test_cancel_and_never_overwrite(synthetic, tmp_path):
    event = threading.Event()
    output = tmp_path / "cancelled.mp4"
    def cancel_after_frame(n, text):
        event.set()
    with pytest.raises(Cancelled):
        export_video(probe(synthetic), [], output, ExportSettings(), event, cancel_after_frame)
    assert not output.exists() and not list(tmp_path.glob("logo-remover-*"))
    with pytest.raises(FileExistsError):
        export_video(probe(synthetic), [], synthetic, ExportSettings())


def test_vfr_rotation_sar_and_audio_offset(tmp_path):
    source = tmp_path / "vfr.mp4"
    unrotated = tmp_path / "vfr-base.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x96:rate=20:duration=2",
           "-f", "lavfi", "-i", "sine=duration=2:sample_rate=48000",
           "-vf", "select='if(lt(t,1),1,not(mod(n,2)))',setsar=4/3",
           "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "2", "-c:a", "aac", unrotated)
    ffmpeg("-display_rotation", "90", "-i", unrotated, "-c", "copy", source)
    meta = probe(source)
    output = tmp_path / "rotated-output.mp4"
    export_video(meta, [], output, ExportSettings(preset="fast"))
    result = probe(output)
    t1, _ = frame_times(meta)
    t2, _ = frame_times(result)
    assert len(t1) == len(t2)
    assert len(set(round(b-a, 3) for a, b in zip(t1, t1[1:]))) > 1
    assert np.allclose(t1, t2, atol=.00001)
    assert result.rotation == meta.rotation == 90
    assert result.sar == meta.sar == "4:3"
    assert audio_hash(source) == audio_hash(output)


@pytest.mark.parametrize("resolution", [(1920, 1080), (3840, 2160)])
def test_hd_single_frame_bounded_pipeline(tmp_path, resolution):
    source = tmp_path / "4k.mp4"
    ffmpeg("-f", "lavfi", "-i", f"color=c=gray:size={resolution[0]}x{resolution[1]}:rate=1:duration=1",
           "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", source)
    meta = probe(source)
    output = tmp_path / "4k-result.mp4"
    export_video(meta, [Mask(points=[(100, 100), (180, 140)], end=1)], output,
                 ExportSettings(preset="fast"))
    result = probe(output)
    assert (result.width, result.height) == resolution
    assert len(frame_times(result)[0]) == 1


def test_source_with_delayed_audio_and_nonzero_start(tmp_path):
    source = tmp_path / "offset.mov"
    ffmpeg("-itsoffset", "0.5", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=2",
           "-itsoffset", "0.9", "-f", "lavfi", "-i", "sine=duration=1:sample_rate=48000",
           "-copyts", "-fps_mode", "passthrough", "-c:v", "libx264", "-threads", "2",
           "-c:a", "aac", source)
    output = tmp_path / "offset-result.mp4"
    meta = probe(source)
    assert meta.start_time >= .4
    export_video(meta, [], output, ExportSettings(preset="fast"))
    original = run_json(["-show_streams", str(source)])["streams"]
    actual = run_json(["-show_streams", str(output)])["streams"]
    def delta(streams):
        audio = next(s for s in streams if s["codec_type"] == "audio")
        video = next(s for s in streams if s["codec_type"] == "video")
        return float(audio["start_time"]) - float(video["start_time"])
    assert delta(actual) == pytest.approx(delta(original), abs=.002)
    assert audio_hash(source) == audio_hash(output)


def test_mov_pcm_audio_fallback(tmp_path):
    source = tmp_path / "pcm.mov"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x96:rate=5:duration=1",
           "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-threads", "2",
           "-c:a", "pcm_s16le", source)
    output = tmp_path / "aac-result.mp4"
    updates = []
    export_video(probe(source), [], output, ExportSettings(preset="fast"),
                 progress=lambda n, text: updates.append(text))
    assert probe(output).audio_codec == "aac"
    assert any("AAC" in text for text in updates)


def test_hdr_and_odd_dimensions_rejected(synthetic, tmp_path):
    from dataclasses import replace
    meta = probe(synthetic)
    with pytest.raises(Exception, match="HDR"):
        export_video(replace(meta, hdr=True), [], tmp_path / "hdr.mp4", ExportSettings())
    with pytest.raises(Exception, match="chẵn"):
        export_video(replace(meta, width=319), [], tmp_path / "odd.mp4", ExportSettings())
    assert not list(tmp_path.iterdir())
