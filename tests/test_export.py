"""화자 분리 발화 구간 처리 및 오디오 내보내기 단위 테스트."""

import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from speaker_split.diarize import Segment, process_segments_and_overlaps
from speaker_split.export import (
    apply_fade,
    export_json,
    export_rttm,
    export_tracks,
    load_audio,
)


def test_process_segments_no_overlap():
    raw_segments = [
        Segment("SPEAKER_00", 0.0, 2.0),
        Segment("SPEAKER_01", 2.5, 4.0),
        Segment("SPEAKER_00", 4.2, 5.0),
    ]
    annotated, intervals = process_segments_and_overlaps(
        raw_segments, keep_overlap=False, min_duration=0.25
    )

    assert len(annotated) == 3
    assert not any(s.overlap for s in annotated)
    assert len(intervals["SPEAKER_00"]) == 2
    assert len(intervals["SPEAKER_01"]) == 1
    assert intervals["SPEAKER_00"][0] == (0.0, 2.0)
    assert intervals["SPEAKER_01"][0] == (2.5, 4.0)


def test_process_segments_with_overlap_excluded():
    # SPEAKER_00: 1.0 ~ 4.0, SPEAKER_01: 3.0 ~ 5.0
    # Overlap: 3.0 ~ 4.0
    raw_segments = [
        Segment("SPEAKER_00", 1.0, 4.0),
        Segment("SPEAKER_01", 3.0, 5.0),
    ]
    annotated, intervals = process_segments_and_overlaps(
        raw_segments, keep_overlap=False, min_duration=0.25
    )

    assert len(annotated) == 2
    assert all(s.overlap for s in annotated)
    # SPEAKER_00 exclusive: 1.0 ~ 3.0
    # SPEAKER_01 exclusive: 4.0 ~ 5.0
    assert intervals["SPEAKER_00"] == [(1.0, 3.0)]
    assert intervals["SPEAKER_01"] == [(4.0, 5.0)]


def test_process_segments_with_overlap_kept():
    raw_segments = [
        Segment("SPEAKER_00", 1.0, 4.0),
        Segment("SPEAKER_01", 3.0, 5.0),
    ]
    annotated, intervals = process_segments_and_overlaps(
        raw_segments, keep_overlap=True, min_duration=0.25
    )

    assert intervals["SPEAKER_00"] == [(1.0, 4.0)]
    assert intervals["SPEAKER_01"] == [(3.0, 5.0)]


def test_process_segments_min_duration_filter():
    raw_segments = [
        Segment("SPEAKER_00", 0.0, 0.1),  # 0.1초 (< 0.25)
        Segment("SPEAKER_00", 1.0, 3.0),  # 2.0초
    ]
    annotated, intervals = process_segments_and_overlaps(
        raw_segments, keep_overlap=False, min_duration=0.25
    )

    assert intervals["SPEAKER_00"] == [(1.0, 3.0)]


def test_apply_fade():
    sr = 16000
    chunk = np.ones((1600, 2), dtype=np.float32)  # 0.1초
    fade_samples = 160  # 10ms
    faded = apply_fade(chunk, fade_samples)

    assert faded.shape == chunk.shape
    assert np.isclose(faded[0, 0], 0.0, atol=1e-3)
    assert np.isclose(faded[-1, 0], 0.0, atol=1e-3)
    assert np.isclose(faded[800, 0], 1.0, atol=1e-3)


def test_export_tracks_and_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 16000
        duration_sec = 6.0
        total_samples = int(sr * duration_sec)

        # 2채널 더미 오디오 신호 생성 (1ch: 440Hz, 2ch: 880Hz)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        audio = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1).astype(np.float32)

        input_audio_path = tmp_path / "test_dialogue.wav"
        sf.write(str(input_audio_path), audio, sr)

        # 로드 검증
        loaded_audio, loaded_sr = load_audio(input_audio_path)
        assert loaded_sr == sr
        assert loaded_audio.shape == audio.shape

        speaker_intervals = {
            "SPEAKER_00": [(0.0, 2.0), (4.0, 5.0)],
            "SPEAKER_01": [(2.0, 3.5)],
        }

        segments = [
            Segment("SPEAKER_00", 0.0, 2.0, False),
            Segment("SPEAKER_01", 2.0, 3.5, False),
            Segment("SPEAKER_00", 4.0, 5.0, False),
        ]

        # mode=both 내보내기
        outputs = export_tracks(
            audio_data=loaded_audio,
            sample_rate=sr,
            speaker_intervals=speaker_intervals,
            output_dir=tmp_path,
            base_stem="test_dialogue",
            mode="both",
            fade_ms=10.0,
        )

        assert "SPEAKER_00" in outputs
        assert "SPEAKER_01" in outputs
        assert outputs["SPEAKER_00"]["pad"] == "test_dialogue_SPEAKER_00.wav"
        assert outputs["SPEAKER_00"]["concat"] == "test_dialogue_SPEAKER_00_concat.wav"
        assert outputs["SPEAKER_01"]["pad"] == "test_dialogue_SPEAKER_01.wav"
        assert outputs["SPEAKER_01"]["concat"] == "test_dialogue_SPEAKER_01_concat.wav"

        # 파일 실제 존재 및 길이 확인
        # pad 파일 확인
        pad_data, pad_sr = sf.read(str(tmp_path / outputs["SPEAKER_00"]["pad"]))
        assert pad_sr == sr
        assert len(pad_data) == total_samples  # 원본 길이 유지
        # 3.0초 시점은 SPEAKER_00이 말하지 않은 구간이므로 0이어야 함
        mid_idx = int(3.0 * sr)
        assert np.allclose(pad_data[mid_idx], 0.0)

        # concat 파일 확인
        concat_data, concat_sr = sf.read(str(tmp_path / outputs["SPEAKER_00"]["concat"]))
        expected_samples = int((2.0 + 1.0) * sr)  # 3.0초 분량
        assert abs(len(concat_data) - expected_samples) <= 2

        # RTTM 저장 검증
        rttm_path = export_rttm(segments, tmp_path, "test_dialogue")
        assert rttm_path.is_file()
        with open(rttm_path, "r", encoding="utf-8") as f:
            rttm_lines = f.readlines()
        assert len(rttm_lines) == 3
        assert "SPEAKER test_dialogue 1 0.000 2.000 <NA> <NA> SPEAKER_00 <NA> <NA>" in rttm_lines[0]

        # JSON 저장 검증
        json_path = export_json(
            input_path=input_audio_path,
            sample_rate=sr,
            duration_sec=duration_sec,
            mode="both",
            keep_overlap=False,
            speakers=["SPEAKER_00", "SPEAKER_01"],
            segments=segments,
            outputs_map=outputs,
            output_dir=tmp_path,
            base_stem="test_dialogue",
        )
        assert json_path.is_file()
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["input"] == "test_dialogue.wav"
        assert meta["sample_rate"] == sr
        assert meta["mode"] == "both"
        assert meta["keep_overlap"] is False
        assert meta["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
        assert len(meta["segments"]) == 3
        assert meta["outputs"]["SPEAKER_00"]["pad"] == "test_dialogue_SPEAKER_00.wav"
