"""유틸리티 함수 및 환경 검증 모듈.

- .env 로드 및 Hugging Face 토큰 관리
- FFmpeg 및 CUDA 가용성 검증
- 친절한 한국어 에러 메시지 및 가이드 출력
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 필수 모델 및 토큰 발급 URL
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
MODEL_DIARIZATION_URL = "https://huggingface.co/pyannote/speaker-diarization-3.1"
MODEL_SEGMENTATION_URL = "https://huggingface.co/pyannote/segmentation-3.0"


def init_env(env_path: Optional[Path] = None) -> None:
    """프로젝트 루트의 .env 파일 또는 지정된 경로의 환경변수를 로드합니다."""
    if env_path and env_path.is_file():
        load_dotenv(dotenv_path=env_path)
    else:
        # 현재 작업 디렉토리 또는 모듈 상위 디렉토리의 .env 탐색
        root_env = Path.cwd() / ".env"
        if root_env.is_file():
            load_dotenv(dotenv_path=root_env)
        else:
            load_dotenv()


def check_ffmpeg() -> bool:
    """FFmpeg가 시스템 PATH에 존재하는지 확인합니다."""
    return shutil.which("ffmpeg") is not None


def print_ffmpeg_missing_error() -> None:
    """FFmpeg 미설치 시 해결 방법을 출력합니다."""
    msg = f"""
[오류] FFmpeg를 찾을 수 없습니다!
오디오 파일 변환 및 처리를 위해 FFmpeg가 필요합니다.

■ Windows 10 해결 방법:
  1) 패키지 관리자를 사용하는 경우 (PowerShell 관리자 권한):
     - winget 사용 시: winget install Gyan.FFmpeg
     - choco 사용 시:  choco install ffmpeg
     - scoop 사용 시:  scoop install ffmpeg
  2) 수동 다운로드 및 설치:
     - https://www.gyan.dev/ffmpeg/builds/ 에서 'ffmpeg-release-essentials.zip' 다운로드
     - 압축 해제 후 bin 폴더(예: C:\\ffmpeg\\bin)를 시스템 환경 변수 'Path'에 추가
  3) 터미널을 다시 시작한 후 실행하세요.
"""
    print(msg, file=sys.stderr)


def get_hf_token(cli_token: Optional[str] = None) -> str:
    """
    Hugging Face Access Token을 획득합니다.
    우선순위: CLI 인자 (--hf-token) > 환경변수 (HF_TOKEN) > .env 파일
    """
    token = cli_token or os.environ.get("HF_TOKEN", "").strip()

    # 플레이스홀더 값 체크
    if not token or token == "hf_your_token_here":
        raise ValueError(get_hf_token_missing_message())
    
    return token


def get_hf_token_missing_message() -> str:
    """HF_TOKEN 누락 시 안내 메시지를 반환합니다."""
    return f"""
[오류] Hugging Face Access Token (HF_TOKEN)을 찾을 수 없습니다!

pyannote.audio 모델을 다운로드하려면 Hugging Face 토큰이 필요합니다.

■ 설정 방법 (다음 중 하나 선택):
  1) 프로젝트 루트의 .env 파일에 토큰 저장:
     .env 파일 생성 후 아래 내용 작성
     HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  2) 환경변수 설정:
     - PowerShell: $env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     - CMD:        set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  3) CLI 실행 시 직접 옵션으로 전달:
     python -m speaker_split input.wav --hf-token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

■ 토큰 발급 및 모델 사용 동의 안내:
  1. 토큰 발급 페이지: {HF_TOKEN_URL} (Read 권한)
  2. 모델 사용 조건 동의 (웹페이지 접속 후 'Agree and access repository' 클릭):
     - {MODEL_DIARIZATION_URL}
     - {MODEL_SEGMENTATION_URL}
"""


def format_gated_repo_error(error_msg: str) -> str:
    """Gated 모델 접근 불가 또는 401/403 권한 에러 시 상세 안내 메시지를 반환합니다."""
    return f"""
[오류] Hugging Face 모델 접근 권한 오류 (Gated Model Access Denied / 401 / 403)
상세: {error_msg}

pyannote/speaker-diarization-3.1 모델을 사용하기 위해서는 Hugging Face 웹사이트에서 
해당 모델들의 사용자 동의(User Agreement)를 먼저 완료해야 합니다.

■ 필수 조치 사항:
  1) Hugging Face에 로그인한 후 아래 2개 모델 페이지에 접속하여 각각 동의 버튼을 누르세요.
     ① 다이어리제이션 모델 동의:
        {MODEL_DIARIZATION_URL}
     ② 세그멘테이션 모델 동의:
        {MODEL_SEGMENTATION_URL}
  2) 사용 중인 토큰이 올바른지 확인하세요:
     {HF_TOKEN_URL}
  3) 승인 완료 후 즉시 다시 실행하시면 정상 작동합니다.
"""


def format_oom_error(error_msg: str) -> str:
    """GPU Out of Memory (OOM) 발생 시 안내 메시지를 반환합니다."""
    return f"""
[오류] GPU 메모리 부족 (CUDA Out of Memory)
상세: {error_msg}

■ 해결 방법:
  1) CPU 모드로 실행:
     python -m speaker_split input.wav --device cpu
  2) VRAM을 많이 사용하는 다른 프로그램(게임, Stable Diffusion, ComfyUI 등)을 종료 후 다시 시도
"""
