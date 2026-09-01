"""화자 분리(Speaker Diarization) 및 발화 구간 처리 모듈.

pyannote.audio 파이프라인을 로드하고 오디오를 분석하여 
각 화자별 발화 구간(세그먼트)과 겹침(overlap) 구간을 계산합니다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from speaker_split.utils import (
    format_gated_repo_error,
    format_oom_error,
    get_hf_token,
)

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"


@dataclass
class Segment:
    """화자 발화 세그먼트 정보"""
    speaker: str
    start: float
    end: float
    overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "overlap": self.overlap,
        }


def get_device(device_pref: str = "auto") -> torch.device:
    """지정된 선호도에 따라 torch.device 객체를 반환합니다."""
    device_pref = device_pref.lower().strip()
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[오류] CUDA 디바이스를 사용할 수 없습니다. "
                "NVIDIA 그래픽 드라이버 및 PyTorch CUDA 지원 버전을 확인하세요."
            )
        return torch.device("cuda")
    elif device_pref == "cpu":
        return torch.device("cpu")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_diarization_pipeline(
    model_id: str = DEFAULT_MODEL,
    hf_token: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Any:
    """
    pyannote.audio Diarization 파이프라인을 로드하고 지정된 디바이스로 이동합니다.
    """
    token = get_hf_token(hf_token)

    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    except Exception as e:
        err_str = str(e)
        if any(keyword in err_str.lower() for keyword in ["401", "403", "gated", "unauthorized", "restricted", "access"]):
            raise PermissionError(format_gated_repo_error(err_str)) from e
        raise RuntimeError(f"[오류] 모델 로드 실패 ({model_id}): {err_str}") from e

    if pipeline is None:
        raise RuntimeError(
            format_gated_repo_error("파이프라인을 불러오지 못했습니다. 토큰 권한 및 모델 동의 여부를 확인하세요.")
        )

    if device is not None:
        try:
            pipeline.to(device)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(format_oom_error(str(e))) from e
        except Exception as e:
            print(f"[경고] {device} 디바이스 이동 실패 ({e}), CPU로 계속 진행합니다.", flush=True)
            pipeline.to(torch.device("cpu"))

    return pipeline


def run_diarization(
    pipeline: Any,
    audio_path: Path,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Segment]:
    """
    파이프라인을 실행하여 원시 발화 세그먼트 목록을 반환합니다.
    """
    params: Dict[str, Any] = {}
    if num_speakers is not None and num_speakers > 0:
        params["num_speakers"] = num_speakers
    else:
        if min_speakers is not None and min_speakers > 0:
            params["min_speakers"] = min_speakers
        if max_speakers is not None and max_speakers > 0:
            params["max_speakers"] = max_speakers

    try:
        diarization = pipeline(str(audio_path), **params)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(format_oom_error(str(e))) from e
    except Exception as e:
        err_str = str(e)
        if any(keyword in err_str.lower() for keyword in ["401", "403", "gated", "unauthorized"]):
            raise PermissionError(format_gated_repo_error(err_str)) from e
        raise RuntimeError(f"[오류] 다이어리제이션 실행 실패: {err_str}") from e

    raw_segments: List[Segment] = []
    # itertracks(yield_label=True) -> (Segment(start, end), track, speaker_label)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append(
            Segment(
                speaker=str(speaker),
                start=float(turn.start),
                end=float(turn.end),
                overlap=False,
            )
        )

    # 시간순 정렬
    raw_segments.sort(key=lambda s: (s.start, s.end))
    return raw_segments


def process_segments_and_overlaps(
    raw_segments: List[Segment],
    keep_overlap: bool = False,
    min_duration: float = 0.25,
) -> Tuple[List[Segment], Dict[str, List[Tuple[float, float]]]]:
    """
    발화 구간을 분석하여 겹침(overlap) 구간을 감지하고,
    화자별 최종 유효 발화 시간 구간 목록 [ (start, end), ... ]을 생성합니다.

    Returns:
        annotated_segments: 원본 세그먼트에 overlap 플래그가 기록된 목록
        speaker_active_intervals: 화자별 오디오 추출용 유효 구간 딕셔너리 {speaker: [(s, e), ...]}
    """
    if not raw_segments:
        return [], {}

    # 1. 모든 세그먼트의 경계 지점을 수집하여 타임라인 분할
    time_points = set()
    for seg in raw_segments:
        time_points.add(seg.start)
        time_points.add(seg.end)
    sorted_times = sorted(time_points)

    # 2. 각 시간 간격 [t_i, t_{i+1}] 별 활성 화자 확인
    intervals_overlap: List[Tuple[float, float]] = []
    for i in range(len(sorted_times) - 1):
        t_start = sorted_times[i]
        t_end = sorted_times[i + 1]
        if t_end - t_start <= 1e-5:
            continue
        t_mid = (t_start + t_end) / 2.0
        active_speakers = {
            seg.speaker for seg in raw_segments
            if seg.start <= t_mid <= seg.end
        }
        if len(active_speakers) >= 2:
            intervals_overlap.append((t_start, t_end))

    # 3. 원본 세그먼트 중 overlap 구간과 겹치는 세그먼트에 overlap=True 마킹
    annotated_segments: List[Segment] = []
    for seg in raw_segments:
        is_ov = False
        for ov_start, ov_end in intervals_overlap:
            if max(seg.start, ov_start) < min(seg.end, ov_end) - 1e-4:
                is_ov = True
                break
        annotated_segments.append(
            Segment(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                overlap=is_ov,
            )
        )

    # 4. 화자별 오디오 추출 구간 계산
    speakers = sorted(list({seg.speaker for seg in raw_segments}))
    speaker_active_intervals: Dict[str, List[Tuple[float, float]]] = {spk: [] for spk in speakers}

    for spk in speakers:
        spk_segments = [s for s in raw_segments if s.speaker == spk]
        if keep_overlap:
            # 겹침을 유지하는 경우: 원본 구간을 그대로 사용하고 연속 구간 병합
            intervals = [(s.start, s.end) for s in spk_segments]
        else:
            # 겹침을 제외하는 경우: 세그먼트에서 overlap 구간을 제외한 순수 독점 구간만 추출
            intervals = []
            for i in range(len(sorted_times) - 1):
                t_start = sorted_times[i]
                t_end = sorted_times[i + 1]
                if t_end - t_start <= 1e-5:
                    continue
                t_mid = (t_start + t_end) / 2.0
                active_speakers = [
                    seg.speaker for seg in raw_segments
                    if seg.start <= t_mid <= seg.end
                ]
                # 해당 화자 혼자만 말한 구간인 경우
                if active_speakers == [spk]:
                    intervals.append((t_start, t_end))

        # 인접/연속된 구간 병합 (Merge adjacent intervals)
        merged: List[Tuple[float, float]] = []
        for s_start, s_end in sorted(intervals):
            if not merged:
                merged.append((s_start, s_end))
            else:
                prev_start, prev_end = merged[-1]
                if s_start <= prev_end + 1e-4:  # 연속 또는 중첩
                    merged[-1] = (prev_start, max(prev_end, s_end))
                else:
                    merged.append((s_start, s_end))

        # min_duration 필터링 (최소 길이 미만 구간 제외)
        filtered = [
            (s_start, s_end) for s_start, s_end in merged
            if (s_end - s_start) >= min_duration
        ]
        speaker_active_intervals[spk] = filtered

    return annotated_segments, speaker_active_intervals
