# VoiceSplitter (화자 분리 Python CLI)

**VoiceSplitter**는 대화 녹음 파일(남성과 여성의 대화, 인터뷰 등)에서 화자별 발화 구간을 정밀하게 분리하여 개별 WAV 오디오 파일로 저장하는 Python CLI 도구입니다.

`pyannote.audio`의 최신 다이어리제이션 파이프라인(`pyannote/speaker-diarization-3.1`)을 기반으로 동작합니다.

---

## 1. 이 앱이 하는 일 / 하지 않는 일

### ✔ 지원하는 기능
- **화자 다이어리제이션(Speaker Diarization)**: 음성 파일에서 "누가 언제 말했는지"를 초 단위로 정확히 감지합니다.
- **화자별 독립 트랙 생성**: 각 화자(`SPEAKER_00`, `SPEAKER_01` 등)의 음성을 별도 WAV 파일로 분할 저장합니다.
- **2가지 트랙 저장 모드 (`--mode`)**:
  - **`pad` (기본값)**: 원본 오디오의 전체 길이를 그대로 유지하며, 해당 화자가 말하지 않은 구간(침묵, 타 화자, 크로스토크)은 0(무음)으로 처리합니다. DAW나 영상 편집기에 올려 타임라인을 그대로 맞추기에 최적입니다.
  - **`concat`**: 침묵 및 타 화자 구간을 잘라내고, 해당 화자가 발화한 부분만 시간 순서대로 이어붙입니다. 이어붙이는 경계면에 부드러운 페이드(기본 10ms)를 적용하여 팝/클릭 노이즈를 방지합니다. 특정 화자의 목소리만 연속으로 듣고 싶을 때 유용합니다.
  - **`both`**: `pad` 트랙과 `concat` 트랙을 모두 한 번에 생성합니다.
- **크로스토크(겹침 발화) 제어 (`--keep-overlap`)**: 두 사람이 동시에 말한 구간을 기본값에서는 무음/제외 처리하거나, 옵션을 켜서 유지할 수 있습니다.
- **상세 메타데이터 출력**: 타임스탬프가 담긴 표준 RTTM 파일(`_diarization.rttm`)과 JSON 파일(`_diarization.json`)을 함께 저장합니다.

### ✖ 하지 않는 일
- **성별/화자 자동 명명**: 음성만으로 남성/여성을 자동 판별하지 않습니다. `SPEAKER_00`, `SPEAKER_01`로 분리된 결과물을 사용자가 직접 들어보고 원하는 화자를 선택합니다.
- **혼합 음원 완전 분리(Source Separation)**: 두 사람이 완전히 동시에 겹쳐서 말한 하나의 마이크 오디오를 분리해내는 보컬 분리 모델이 아닙니다. 다이어리제이션(시간 구간 분할) 기반이므로, 기본 모드에서는 겹치는 구간을 깨끗하게 제외하여 화자별 순수 발화만 보존합니다.

---

## 2. Windows 10 설치 및 환경 구성

### 요구 사항
- **OS**: Windows 10 / 11 (64-bit)
- **Python**: Python 3.10 이상 (Python 3.11 권장)
- **FFmpeg**: 시스템 PATH에 등록되어 있어야 함

### 단계별 설치 순서

#### 1단계: FFmpeg 설치 및 PATH 등록
FFmpeg가 설치되어 있지 않다면 아래 방법 중 하나로 설치합니다.

- **PowerShell (관리자 권한)**:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  *(또는 `choco install ffmpeg` / `scoop install ffmpeg`)*

- **수동 설치**:
  1. [gyan.dev FFmpeg 다운로드](https://www.gyan.dev/ffmpeg/builds/)에서 `ffmpeg-release-essentials.zip`을 다운로드합니다.
  2. 압축을 풀고 `bin` 폴더 경로(예: `C:\ffmpeg\bin`)를 시스템 환경 변수 `Path`에 추가합니다.

설치 확인:
```cmd
ffmpeg -version
```

#### 2단계: 가상환경 생성 및 활성화
프로젝트 루트 디렉토리에서 Python 가상환경을 생성합니다.

```powershell
# Python 3.11 가상환경 생성
py -3.11 -m venv .venv

# 가상환경 활성화 (PowerShell)
.venv\Scripts\Activate.ps1

# (CMD를 사용하는 경우)
# .venv\Scripts\activate.bat
```

> **참고 (PowerShell 실행 권한 에러 시)**:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 를 실행하세요.

#### 3단계: 필수 패키지 설치

- **일반 CPU 환경**:
  ```bash
  pip install -r requirements.txt
  ```

- **NVIDIA GPU (CUDA) 가속 사용 시 (권장)**:
  NVIDIA 그래픽 카드(예: RTX 2060 이상)가 있는 경우, CUDA 버전의 PyTorch를 먼저 설치하면 훨씬 빠른 속도로 화자 분리를 수행할 수 있습니다.
  ```bash
  # CUDA 12.1 지원 PyTorch 설치
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

  # 나머지 의존성 설치
  pip install -r requirements.txt
  ```

---

## 3. Hugging Face 토큰 발급 및 모델 사용 동의

`pyannote.audio` 3.1 모델을 다운로드하여 사용하려면 Hugging Face 계정과 사용자 동의가 필요합니다.

1. **Hugging Face 계정 로그인 / 가입**: [https://huggingface.co](https://huggingface.co)
2. **Access Token 발급**:
   - [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 에 접속합니다.
   - `New token`을 클릭하고 `Read` 권한으로 토큰을 생성하여 복사합니다.
3. **Gated 모델 사용자 동의 (2개 필수)**:
   - 아래 2개 페이지에 접속하여 각각 **"Agree and access repository"** 버튼을 클릭합니다.
     - ① 다이어리제이션 파이프라인: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
     - ② 세그멘테이션 모델: [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. **`.env` 파일에 토큰 설정**:
   프로젝트 루트에 `.env` 파일을 생성하고 발급받은 토큰을 입력합니다.
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## 4. CLI 사용법 및 실행 예시

### 기본 실행 (`pad` 모드)
원본 파일의 길이를 유지하면서 화자별로 침묵 처리된 트랙을 생성합니다.
```bash
python -m speaker_split input.wav
```

### `concat` 모드 실행
침묵을 제거하고 각 화자가 말한 구간만 연속으로 이어붙입니다.
```bash
python -m speaker_split input.wav --mode concat
```

### `both` 모드 실행 (pad + concat 동시 생성)
```bash
python -m speaker_split input.wav --mode both -o output
```

### 주요 옵션 활용 예시
```bash
# 3인 대화 분리 및 최소 0.3초 미만 발화 무시
python -m speaker_split dialogue.mp3 --num-speakers 3 --min-duration 0.3

# 크로스토크(겹침 발화)를 버리지 않고 화자 파일에 유지
python -m speaker_split dialogue.m4a --mode concat --keep-overlap

# GPU 대신 CPU 강제 지정
python -m speaker_split input.wav --device cpu

# CLI 옵션으로 직접 토큰 전달
python -m speaker_split input.wav --hf-token "hf_xxxxxxxxxxxxxxxx"
```

### CLI 인자 목록

| 인자 | 설명 | 기본값 |
|---|---|---|
| `input` | 입력 오디오 경로 (WAV, MP3, M4A, FLAC 등) | **필수** |
| `-o`, `--output-dir` | 출력 파일 저장 폴더 | `output` |
| `--mode` | 저장 모드 (`pad` \| `concat` \| `both`) | `pad` |
| `--num-speakers` | 고정 화자 수 | `2` |
| `--min-speakers` | 최소 화자 수 (자동 감지 시) | `None` |
| `--max-speakers` | 최대 화자 수 (자동 감지 시) | `None` |
| `--min-duration` | 이보다 짧은 발화 구간(초)은 무시 | `0.25` |
| `--keep-overlap` | 겹치는 발화(크로스토크) 구간을 양쪽 화자 트랙에 유지 | `off` |
| `--fade-ms` | concat 연결 경계 페이드 길이 (ms, 0이면 비활성화) | `10.0` |
| `--hf-token` | Hugging Face Access Token | `None` |
| `--device` | 연산 디바이스 (`auto` \| `cpu` \| `cuda`) | `auto` |
| `--verbose` | 상세 로그 출력 | `off` |

---

## 5. 출력 파일 규칙

모든 결과물은 지정된 출력 디렉토리(기본: `output/`)에 저장됩니다.

```
output/
├── <파일명>_SPEAKER_00.wav          # pad 트랙 (mode=pad 또는 both)
├── <파일명>_SPEAKER_00_concat.wav   # concat 트랙 (mode=concat 또는 both)
├── <파일명>_SPEAKER_01.wav          # pad 트랙
├── <파일명>_SPEAKER_01_concat.wav   # concat 트랙
├── <파일명>_diarization.rttm        # 표준 RTTM 타임라인
└── <파일명>_diarization.json        # JSON 메타데이터
```

### JSON 메타데이터 예시 (`<파일명>_diarization.json`)
```json
{
  "input": "대화.wav",
  "sample_rate": 48000,
  "duration_sec": 123.45,
  "mode": "both",
  "keep_overlap": false,
  "speakers": [
    "SPEAKER_00",
    "SPEAKER_01"
  ],
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 1.20,
      "end": 3.45,
      "duration": 2.25,
      "overlap": false
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.40,
      "end": 5.10,
      "duration": 1.70,
      "overlap": true
    }
  ],
  "outputs": {
    "SPEAKER_00": {
      "pad": "대화_SPEAKER_00.wav",
      "concat": "대화_SPEAKER_00_concat.wav"
    },
    "SPEAKER_01": {
      "pad": "대화_SPEAKER_01.wav",
      "concat": "대화_SPEAKER_01_concat.wav"
    }
  }
}
```

---

## 6. 어떤 파일을 여성/남성 목소리로 쓰면 되나요?

- 다이어리제이션 완료 후 `output/` 폴더에 생성된 `_SPEAKER_00.wav` (또는 `_concat.wav`)와 `_SPEAKER_01.wav` 파일을 미디어 플레이어로 직접 재생해 봅니다.
- 첫 번째로 말하기 시작한 화자가 통상 `SPEAKER_00`으로 지정됩니다.
- 들어보신 후 원하는 화자(예: 여성 목소리)의 파일만 골라 사용하시면 됩니다.

---

## 7. 자주 발생하는 오류 및 해결 방법

### 1) Hugging Face 토큰 오류 (`[오류] Hugging Face Access Token...`)
- 원인: 토큰이 설정되지 않았거나 잘못 입력되었습니다.
- 해결: `.env` 파일에 `HF_TOKEN=hf_...` 형태로 올바른 토큰을 입력하거나 `--hf-token` 옵션으로 전달합니다.

### 2) Gated Repo 권한 오류 (`401 / 403 / Gated Model Access Denied`)
- 원인: 모델 사용 조건에 동의하지 않았습니다.
- 해결: Hugging Face에 로그인한 상태에서 아래 2개 페이지를 방문하여 **"Agree and access repository"**를 눌러 동의를 완료해야 합니다.
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

### 3) FFmpeg 미설치 오류 (`[오류] FFmpeg를 찾을 수 없습니다!`)
- 원인: 시스템 환경 변수 `Path`에 `ffmpeg`가 등록되지 않았습니다.
- 해결: `winget install Gyan.FFmpeg`로 설치 후 터미널을 다시 열고 실행합니다.

### 4) GPU 메모리 부족 (`CUDA Out of Memory`)
- 원인: GPU VRAM이 부족하여 연산이 실패했습니다.
- 해결:
  - `--device cpu` 옵션을 추가하여 CPU 모드로 실행합니다.
  - VRAM을 많이 사용하는 백그라운드 프로그램(Stable Diffusion, ComfyUI, 게임 등)을 종료합니다.
