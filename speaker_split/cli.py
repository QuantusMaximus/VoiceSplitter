"""화자 분리 CLI 진입점 및 실행 컨트롤러.

argparse를 통해 사용자의 명령줄 인자를 파싱하고 전체 화자 분리 및 파일 생성 파이프라인을 실행합니다.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

# Windows 콘솔 한글/인코딩 호환성 보장
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from speaker_split import __version__
from speaker_split.diarize import (
    DEFAULT_MODEL,
    get_device,
    load_diarization_pipeline,
    process_segments_and_overlaps,
    run_diarization,
)
from speaker_split.export import (
    export_json,
    export_rttm,
    export_tracks,
    load_audio,
)
from speaker_split.utils import (
    check_ffmpeg,
    get_hf_token,
    init_env,
    print_ffmpeg_missing_error,
)


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자 파서를 구성합니다."""
    parser = argparse.ArgumentParser(
        prog="python -m speaker_split",
        description="화자 분리 Python CLI 도구 - 음성 파일에서 화자를 분리하여 개별 WAV 트랙으로 저장합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  python -m speaker_split input.wav
  python -m speaker_split input.wav --num-speakers 2 --output-dir output
  python -m speaker_split input.wav --mode concat
  python -m speaker_split input.wav --mode both -o output --min-duration 0.3
  python -m speaker_split "C:\\녹음\\대화.m4a" --mode concat --keep-overlap
""",
    )

    parser.add_argument(
        "input",
        type=str,
        help="입력 오디오 파일 경로 (WAV, MP3, M4A, FLAC 등)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="출력 결과 저장 디렉토리 (기본값: output)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="고정 화자 수 (기본값: 2)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="최소 화자 수 (자동 감지 시 사용)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="최대 화자 수 (자동 감지 시 사용)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pad", "concat", "both"],
        default="pad",
        help="트랙 저장 모드: pad (원래 길이 유지+무음), concat (말한 부분만 순차 연결), both (둘 다 생성) (기본값: pad)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.25,
        help="이보다 짧은 발화 구간(초)은 버림 (기본값: 0.25)",
    )
    parser.add_argument(
        "--keep-overlap",
        action="store_true",
        default=False,
        help="화자 간 겹치는 발화(크로스토크) 구간을 버리지 않고 양쪽 화자 파일에 유지 (기본값: 제외/무음)",
    )
    parser.add_argument(
        "--fade-ms",
        type=float,
        default=10.0,
        help="concat 모드 연결 경계 페이드 길이 (ms, 기본값: 10.0, 0이면 비활성화)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face Access Token (지정 시 환경변수/.env보다 우선)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="연산 디바이스 선택 (auto: GPU 가용 시 CUDA 사용, cpu, cuda) (기본값: auto)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="상세 로그 출력",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="버전 정보 출력",
    )

    return parser


def run(
    input_path_str: str,
    output_dir_str: str = "output",
    num_speakers: Optional[int] = 2,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    mode: str = "pad",
    min_duration: float = 0.25,
    keep_overlap: bool = False,
    fade_ms: float = 10.0,
    hf_token: Optional[str] = None,
    device_pref: str = "auto",
    verbose: bool = False,
) -> int:
    """화자 분리 파이프라인의 핵심 실행 함수."""
    start_time = time.time()
    init_env()

    # 1. 입력 파일 유효성 확인
    input_path = Path(input_path_str).resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"[오류] 입력 오디오 파일을 찾을 수 없습니다: {input_path}", file=sys.stderr)
        return 1

    # 2. FFmpeg 가용성 검증
    if not check_ffmpeg():
        print_ffmpeg_missing_error()
        return 1

    # 3. Hugging Face 토큰 사전 확인
    try:
        get_hf_token(hf_token)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    # 4. 연산 디바이스 결정
    try:
        device = get_device(device_pref)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = input_path.stem

    print(f"==================================================")
    print(f" VoiceSplitter - 화자 분리 CLI (v{__version__})")
    print(f"==================================================")
    print(f">> 입력 파일: {input_path.name}")
    print(f">> 저장 위치: {output_dir}")
    print(f">> 저장 모드: {mode} (keep_overlap={keep_overlap}, fade={fade_ms}ms)")
    print(f">> 실행 디바이스: {device}")
    print(f"--------------------------------------------------")

    # 5. 오디오 로드 및 기본 정보 확인
    print(f"[1/5] 오디오 파일 로드 중...")
    try:
        audio_data, sample_rate = load_audio(input_path)
    except Exception as e:
        print(f"[오류] 오디오 파일 로드 실패: {e}", file=sys.stderr)
        return 1

    total_samples, num_channels = audio_data.shape
    duration_sec = total_samples / float(sample_rate)
    print(f"      - 샘플레이트: {sample_rate} Hz, 채널: {num_channels}ch, 길이: {duration_sec:.2f}초")

    # 6. 다이어리제이션 모델 로드
    print(f"[2/5] pyannote 모델 로드 중 ({DEFAULT_MODEL})...")
    try:
        pipeline = load_diarization_pipeline(
            model_id=DEFAULT_MODEL,
            hf_token=hf_token,
            device=device,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    # 7. 다이어리제이션 실행
    print(f"[3/5] 화자 다이어리제이션 분석 실행 중 (오디오 길이에 따라 수십 초 소요될 수 있습니다)...")
    try:
        raw_segments = run_diarization(
            pipeline=pipeline,
            audio_path=input_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1

    # 8. 세그먼트 및 겹침(overlap) 처리
    print(f"[4/5] 발화 구간 정리 및 겹침(overlap) 처리 중...")
    annotated_segments, speaker_active_intervals = process_segments_and_overlaps(
        raw_segments=raw_segments,
        keep_overlap=keep_overlap,
        min_duration=min_duration,
    )

    detected_speakers = sorted(list(speaker_active_intervals.keys()))
    print(f"      - 감지된 화자 ({len(detected_speakers)}명): {', '.join(detected_speakers) if detected_speakers else '없음'}")
    print(f"      - 총 유효 발화 세그먼트: {len(annotated_segments)}개")

    # 9. 오디오 트랙 및 메타데이터 내보내기
    print(f"[5/5] 오디오 트랙 및 결과 메타데이터 파일 내보내기...")
    try:
        outputs_map = export_tracks(
            audio_data=audio_data,
            sample_rate=sample_rate,
            speaker_intervals=speaker_active_intervals,
            output_dir=output_dir,
            base_stem=base_stem,
            mode=mode,
            fade_ms=fade_ms,
        )

        rttm_path = export_rttm(
            segments=annotated_segments,
            output_dir=output_dir,
            base_stem=base_stem,
        )

        json_path = export_json(
            input_path=input_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            mode=mode,
            keep_overlap=keep_overlap,
            speakers=detected_speakers,
            segments=annotated_segments,
            outputs_map=outputs_map,
            output_dir=output_dir,
            base_stem=base_stem,
        )
    except Exception as e:
        print(f"[오류] 파일 내보내기 실패: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - start_time
    print(f"--------------------------------------------------")
    print(f"[완료] 처리가 완료되었습니다! (소요 시간: {elapsed:.2f}초)")
    print(f"\n[생성된 파일 목록]")
    for spk, files in outputs_map.items():
        if "pad" in files:
            print(f"  - [{spk} 원본길이 유지/무음] {output_dir / files['pad']}")
        if "concat" in files:
            print(f"  - [{spk} 연속 발화 이어붙임] {output_dir / files['concat']}")
    print(f"  - [RTTM 타임라인] {rttm_path}")
    print(f"  - [JSON 메타데이터] {json_path}")
    print(f"==================================================")

    return 0


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI 진입 함수."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    ret = run(
        input_path_str=parsed_args.input,
        output_dir_str=parsed_args.output_dir,
        num_speakers=parsed_args.num_speakers,
        min_speakers=parsed_args.min_speakers,
        max_speakers=parsed_args.max_speakers,
        mode=parsed_args.mode,
        min_duration=parsed_args.min_duration,
        keep_overlap=parsed_args.keep_overlap,
        fade_ms=parsed_args.fade_ms,
        hf_token=parsed_args.hf_token,
        device_pref=parsed_args.device,
        verbose=parsed_args.verbose,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
