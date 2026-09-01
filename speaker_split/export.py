"""오디오 트랙 분할 및 결과(WAV, RTTM, JSON) 내보내기 모듈.

- 원본 오디오 로드 (다양한 포맷 지원)
- 화자별 pad 모드 WAV 생성 (원본 길이 유지, 타 화자/무음 구간 0 처리)
- 화자별 concat 모드 WAV 생성 (발화 구간만 순차 연결, 경계 fade 적용)
- RTTM 및 JSON 메타데이터 저장
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf

from speaker_split.diarize import Segment


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """
    오디오 파일을 로드하여 (waveform_2d, sample_rate) 튜플을 반환합니다.
    waveform_2d는 shape (num_samples, num_channels)의 float32 numpy array입니다.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"[오류] 입력 오디오 파일을 찾을 수 없습니다: {audio_path}")

    # 1. soundfile로 직접 로드 시도 (WAV, FLAC, OGG 등)
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        pass

    # 2. torchaudio로 로드 시도
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(audio_path))
        # waveform shape: (channels, samples) -> transpose to (samples, channels)
        data = waveform.cpu().numpy().T.astype(np.float32)
        return data, sr
    except Exception:
        pass

    # 3. FFmpeg subprocess를 이용한 PCM 파이프 디코딩 (m4a, mp3, 기타 모든 포맷 대응)
    try:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(audio_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-"
        ]
        # 먼저 오디오의 채널 수와 샘플레이트 정보 확인을 위한 probe 시도 (또는 기본 44100/2ch로 변환하지 않고 원본 스트림 추출)
        # 안정적인 변환: 44.1kHz stereo로 표준 추출
        cmd_fixed = [
            "ffmpeg",
            "-v", "error",
            "-i", str(audio_path),
            "-ar", "44100",
            "-ac", "2",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-"
        ]
        proc = subprocess.Popen(cmd_fixed, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_out, raw_err = proc.communicate()
        if proc.returncode == 0 and len(raw_out) > 0:
            arr = np.frombuffer(raw_out, dtype=np.float32)
            arr = arr.reshape(-1, 2)
            return arr, 44100
    except Exception:
        pass

    raise RuntimeError(
        f"[오류] 오디오 파일을 읽을 수 없습니다: {audio_path}\n"
        f"파일이 손상되었거나 지원되지 않는 코덱일 수 있습니다. FFmpeg 설치 여부를 확인하세요."
    )


def apply_fade(chunk: np.ndarray, fade_samples: int) -> np.ndarray:
    """오디오 청크의 시작과 끝에 선형 페이드 인/아웃을 적용합니다."""
    if fade_samples <= 0 or len(chunk) <= 2:
        return chunk
    
    actual_fade = min(fade_samples, len(chunk) // 2)
    if actual_fade <= 0:
        return chunk

    weights_in = np.linspace(0.0, 1.0, actual_fade, endpoint=True, dtype=np.float32)[:, None]
    weights_out = weights_in[::-1]

    chunk = chunk.copy()
    chunk[:actual_fade] *= weights_in
    chunk[-actual_fade:] *= weights_out
    return chunk


def export_tracks(
    audio_data: np.ndarray,
    sample_rate: int,
    speaker_intervals: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
    base_stem: str,
    mode: str = "pad",
    fade_ms: float = 10.0,
) -> Dict[str, Dict[str, str]]:
    """
    화자별 발화 구간을 기반으로 WAV 오디오 파일들을 내보냅니다.

    Args:
        audio_data: 원본 오디오 배열 (num_samples, num_channels)
        sample_rate: 샘플레이트
        speaker_intervals: 화자별 유효 발화 시간 구간 {speaker: [(start, end), ...]}
        output_dir: 출력 폴더
        base_stem: 입력 파일 기본 이름
        mode: "pad" | "concat" | "both"
        fade_ms: concat 경계 페이드 길이 (ms)

    Returns:
        화자별 생성된 파일 맵 {"SPEAKER_00": {"pad": "...", "concat": "..."}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_samples, num_channels = audio_data.shape
    outputs_map: Dict[str, Dict[str, str]] = {}
    fade_samples = int(round((fade_ms / 1000.0) * sample_rate))

    do_pad = mode in ("pad", "both")
    do_concat = mode in ("concat", "both")

    for speaker, intervals in speaker_intervals.items():
        outputs_map[speaker] = {}

        # 1. PAD 트랙 생성
        if do_pad:
            pad_filename = f"{base_stem}_{speaker}.wav"
            pad_path = output_dir / pad_filename
            pad_buffer = np.zeros_like(audio_data)

            for s_start, s_end in intervals:
                idx_start = max(0, int(round(s_start * sample_rate)))
                idx_end = min(total_samples, int(round(s_end * sample_rate)))
                if idx_end > idx_start:
                    pad_buffer[idx_start:idx_end] = audio_data[idx_start:idx_end]

            sf.write(str(pad_path), pad_buffer, sample_rate, subtype="PCM_16")
            outputs_map[speaker]["pad"] = pad_filename

        # 2. CONCAT 트랙 생성
        if do_concat:
            concat_filename = f"{base_stem}_{speaker}_concat.wav"
            concat_path = output_dir / concat_filename
            chunks: List[np.ndarray] = []

            for s_start, s_end in intervals:
                idx_start = max(0, int(round(s_start * sample_rate)))
                idx_end = min(total_samples, int(round(s_end * sample_rate)))
                if idx_end > idx_start:
                    chunk = audio_data[idx_start:idx_end].copy()
                    if fade_samples > 0:
                        chunk = apply_fade(chunk, fade_samples)
                    chunks.append(chunk)

            if chunks:
                concat_data = np.concatenate(chunks, axis=0)
            else:
                # 발화 구간이 전혀 없는 경우 0.1초 무음 파일 생성
                concat_data = np.zeros((int(sample_rate * 0.1), num_channels), dtype=np.float32)

            sf.write(str(concat_path), concat_data, sample_rate, subtype="PCM_16")
            outputs_map[speaker]["concat"] = concat_filename

    return outputs_map


def export_rttm(
    segments: List[Segment],
    output_dir: Path,
    base_stem: str,
) -> Path:
    """표준 RTTM 형식으로 다이어리제이션 결과를 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = output_dir / f"{base_stem}_diarization.rttm"

    lines: List[str] = []
    for seg in segments:
        duration = max(0.0, seg.end - seg.start)
        # SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
        line = f"SPEAKER {base_stem} 1 {seg.start:.3f} {duration:.3f} <NA> <NA> {seg.speaker} <NA> <NA>"
        lines.append(line)

    with open(rttm_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    return rttm_path


def export_json(
    input_path: Path,
    sample_rate: int,
    duration_sec: float,
    mode: str,
    keep_overlap: bool,
    speakers: List[str],
    segments: List[Segment],
    outputs_map: Dict[str, Dict[str, str]],
    output_dir: Path,
    base_stem: str,
) -> Path:
    """다이어리제이션 메타데이터 및 출력 파일 정보를 JSON 파일로 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_stem}_diarization.json"

    data = {
        "input": input_path.name,
        "sample_rate": sample_rate,
        "duration_sec": round(duration_sec, 3),
        "mode": mode,
        "keep_overlap": keep_overlap,
        "speakers": speakers,
        "segments": [seg.to_dict() for seg in segments],
        "outputs": outputs_map,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path
