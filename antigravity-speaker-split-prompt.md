# Antigravity 작업지시서 — 화자 분리 Python CLI

사용자는 빈 프로젝트 폴더만 만들었다. 패키지 구조, 의존성, 코드, README, 실행 방법까지 에이전트가 전부 구성한다.

---

# 작업 목표

Windows 10에서 동작하는 **화자 분리 Python CLI 앱**을 현재 프로젝트 폴더에 처음부터 만들어라.  
나는 폴더만 만들었다. 네가 프로젝트를 완성해라.

녹음 파일(남성과 여성이 대화 등)을 넣으면 화자별로 오디오를 나눠 저장한다.  
성별을 자동 판별할 필요는 없다. `SPEAKER_00`, `SPEAKER_01`처럼 나눈 뒤 사용자가 직접 고른다.

# 핵심 기능

1. CLI로 입력 오디오 경로를 받는다.
2. `pyannote.audio` speaker diarization으로 “누가 언제 말했는지” 찾는다.
3. 화자별로 개별 WAV 파일을 저장한다.
4. **트랙 모드를 두 가지 제공한다.**
   - `pad` (기본값): 원본 길이를 유지한다. 해당 화자가 말하지 않는 구간은 **무음**으로 둔다.
   - `concat`: 무음/다른 화자 구간을 **잘라내고**, 그 화자가 말한 부분만 **순서대로 이어붙인다.**
5. 겹치는 말(크로스토크)은 기본값에서 제외하거나 무음 처리한다. `--keep-overlap`가 있을 때만 겹침 구간을 해당 화자 파일에 남긴다.
6. 화자 수는 기본 2명. `--num-speakers`로 지정 가능하게 한다.
7. Hugging Face 토큰은 환경변수 `HF_TOKEN` 또는 프로젝트 루트 `.env`의 `HF_TOKEN`에서 읽는다.
8. 토큰/모델 동의 문제가 있으면, 어떤 페이지에서 무엇을 동의해야 하는지 에러 메시지에 명확히 출력한다.

# 트랙 모드 상세 (`--mode`)

## `pad` (기본)

- 출력 WAV 길이는 원본과 같다.
- 해당 화자 구간만 원본 오디오를 복사하고, 나머지(침묵 + 다른 화자 + 기본값에서 겹침)는 0으로 채운다.
- 타임라인을 맞춰 편집하거나 DAW에 올릴 때 유용하다.

## `concat`

- 해당 화자의 발화 구간만 시간 순서대로 잘라 이어붙인다.
- 무음, 다른 화자 구간, (기본값에서) 겹침 구간은 버린다.
- 출력 파일 길이는 그 화자가 말한 시간의 합이다.
- 여성/남성 목소리만 연속으로 듣고 저장하고 싶을 때 사용한다.
- 이어붙일 때 클릭 노이즈를 줄이려면 구간 경계에 아주 짧은 fade in/out(예: 5~15ms)을 선택적으로 적용하라. 과도한 크로스페이드는 하지 마라.
- `concat` 결과에도 어떤 원본 구간을 붙였는지 JSON에 기록한다.

두 모드를 한 번에 뽑는 옵션도 제공한다.

- `--mode both` : `pad`와 `concat` 파일을 모두 저장한다.

파일명 규칙:

```
output/<원본파일명>_SPEAKER_00.wav              # mode=pad 이거나 both의 pad
output/<원본파일명>_SPEAKER_00_concat.wav       # mode=concat 이거나 both의 concat
output/<원본파일명>_SPEAKER_01.wav
output/<원본파일명>_SPEAKER_01_concat.wav
```

`mode=concat`만 지정하면 `_concat` 접미사 없이 저장해도 되지만, README에 파일명 규칙을 하나로 고정해 적어라.  
권장: **모드와 무관하게 pad는 접미사 없음, concat는 `_concat`.**  
`--mode concat`만 써도 `_concat`를 붙여 혼동을 줄여라.

# 기술 요구

- Python 3.10+ (3.11 권장)
- Windows 10 기준
- 메인 모델: `pyannote/speaker-diarization-3.1` (없으면 현재 안정적인 동등 pyannote 파이프라인. 사용하는 모델 ID를 README에 명시)
- 오디오 I/O: `torchaudio` 또는 `soundfile` + ffmpeg 중 안정적인 조합
- GPU(CUDA)가 있으면 자동 사용, 없으면 CPU
- `.env` 로딩: `python-dotenv`
- FFmpeg가 PATH에 없으면 설치 안내를 출력하고 종료
- 불필요한 GUI, 웹앱, Docker는 만들지 말 것

# CLI 인터페이스

실행 예:

```bash
python -m speaker_split input.wav
python -m speaker_split input.wav --num-speakers 2 --output-dir output
python -m speaker_split input.wav --mode concat
python -m speaker_split input.wav --mode both -o output --min-duration 0.3
python -m speaker_split "C:\녹음\대화.m4a" --mode concat --keep-overlap
```

인자:

| 인자 | 설명 | 기본값 |
|---|---|---|
| `input` | 입력 오디오 경로 (wav, mp3, m4a, flac 등) | 필수 |
| `--output-dir` / `-o` | 출력 폴더 | `output` |
| `--num-speakers` | 화자 수 | `2` |
| `--min-speakers` | 최소 화자 수 | 없음 |
| `--max-speakers` | 최대 화자 수 | 없음 |
| `--mode` | `pad` \| `concat` \| `both` | `pad` |
| `--min-duration` | 이보다 짧은 구간(초)은 버림 | `0.25` |
| `--keep-overlap` | 겹침 구간을 해당 화자 파일에 남김 | off |
| `--fade-ms` | concat 경계 fade 길이(ms). `0`이면 없음 | `10` |
| `--hf-token` | HF 토큰. 있으면 env/.env보다 우선 | 없음 |
| `--device` | `auto` \| `cpu` \| `cuda` | `auto` |
| `--verbose` | 상세 로그 | off |

`--num-speakers`가 있으면 그것을 우선한다.  
없으면 `--min-speakers` / `--max-speakers`를 파이프라인에 넘긴다.  
둘 다 없으면 pyannote 자동 감지. 다만 기본 UX는 2명 대화이므로 `--num-speakers 2`가 기본이다.

# 출력 파일

```
output/<원본파일명>_SPEAKER_00.wav
output/<원본파일명>_SPEAKER_01.wav
output/<원본파일명>_SPEAKER_00_concat.wav   # concat 또는 both
output/<원본파일명>_SPEAKER_01_concat.wav
output/<원본파일명>_diarization.rttm
output/<원본파일명>_diarization.json
```

JSON에 최소한 아래를 넣어라.

```json
{
  "input": "대화.wav",
  "sample_rate": 48000,
  "duration_sec": 123.45,
  "mode": "both",
  "keep_overlap": false,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {"speaker": "SPEAKER_00", "start": 1.20, "end": 3.45, "overlap": false},
    {"speaker": "SPEAKER_01", "start": 3.40, "end": 5.10, "overlap": true}
  ],
  "outputs": {
    "SPEAKER_00": {
      "pad": "대화_SPEAKER_00.wav",
      "concat": "대화_SPEAKER_00_concat.wav"
    }
  }
}
```

# 프로젝트 구성

대략 이렇게 만들어라. 더 단순해도 되지만 모듈은 나눠라.

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── pyproject.toml
└── speaker_split/
    ├── __init__.py
    ├── __main__.py
    ├── cli.py
    ├── diarize.py
    ├── export.py
    └── utils.py
```

`.gitignore`에 넣을 것:

- `.env`
- `output/`
- `__pycache__/`
- `.venv/`
- `*.wav`, `*.mp3`, `*.m4a`, `*.flac` (실수 커밋 방지. 샘플은 포함하지 않음)

`.env.example`:

```
HF_TOKEN=hf_your_token_here
```

실제 토큰을 코드나 문서에 넣지 마라.

# 구현 세부

- 다이어리제이션 입력은 내부적으로 16kHz mono로 변환한다.
- **저장되는 화자별 WAV는 가능하면 원본 샘플레이트/채널을 유지**한다.  
  유지가 어려우면 16kHz mono WAV로 저장하고 README에 명시한다.
- 화자별 `pad` 트랙: 원본과 같은 길이의 빈 버퍼를 만든 뒤 해당 화자 구간만 복사.
- 화자별 `concat` 트랙: 필터된 구간을 시간순으로 이어붙임.
- 겹침 판정: 두 화자 구간이 시간상 겹치면 overlap으로 표시.  
  `--keep-overlap`가 꺼져 있으면 그 겹친 시간은 양쪽 `pad`에서 무음, `concat`에서는 제외.
- `--min-duration`보다 짧은 구간은 버린다.
- 진행 상황 print: 모델 로드 / 다이어리제이션 / overlap 처리 /보내기 / 저장.
- 메인 로직은 함수로 분리해서 CLI가 아닌 코드에서도 호출 가능하게 한다.
- Windows 경로와 한글 파일명을 고려한다. (`pathlib` 사용)
- 예외 메시지는 한국어로 이해하기 쉽게 쓴다. 기술 키워드(HF_TOKEN, gated, ffmpeg)는 그대로 둔다.

# 에러 처리

다음을 감지하고 해결 방법을 출력한 뒤 non-zero로 종료한다.

1. 입력 파일이 없음
2. ffmpeg 없음
3. `HF_TOKEN` 없음
4. gated 모델 미동의 / 401 / 403
5. CUDA 요청했는데 GPU 없음 → CPU로 자동 폴백하거나 명확히 실패. `auto`면 CPU 폴백
6. 메모리 부족 시 짧은 안내

gated 관련 메시지에 반드시 포함할 URL:

- https://huggingface.co/settings/tokens
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

# README에 반드시 적을 내용

한국어로 작성.

1. 이 앱이 하는 일 / 안 하는 일
   - 화자 분리 후 파일 저장
   - `pad`(길이 유지+무음) / `concat`(말한 부분만 이어붙임)
   - 남성/여성 자동 판별 없음
   - 동시에 말한 목소리를 음원 분리 모델처럼 깨끗이 풀어내지는 않음
2. Windows 10 설치 순서
   - Python 3.10+
   - FFmpeg를 PATH에 추가
   - venv 생성
   - `pip install -r requirements.txt`
   - GPU 사용 시 PyTorch CUDA 휠 안내
3. Hugging Face
   - 계정이 이미 있으면 재가입 불필요
   - Read 토큰 발급
   - 모델 이용 동의 2개 필수
   - `.env`에 `HF_TOKEN` 저장
4. 실행 예시 (`pad`, `concat`, `both`)
5. 출력 파일 설명
6. 어떤 파일을 여성 목소리로 쓰면 되는지 (직접 들어보고 고른다)
7. 자주 나는 오류 (토큰, gated, ffmpeg, CUDA, 메모리)

# 품질 기준

- 빈 폴더에서 에이전트가 실행 가능한 프로젝트로 완성할 것
- 의존성은 최소로
- 코드에 HF 토큰 하드코딩 금지
- 샘플 오디오 포함 금지
- `python -m speaker_split --help`가 동작해야 함
- argparse 또는 typer 중 하나만 쓰고, help 문구는 한국어+영문 옵션명

# 완료 조건

- 모듈, CLI, requirements, .env.example, .gitignore, README가 모두 있음
- README만 보고 설치/토큰/실행이 가능
- `--mode pad|concat|both`가 문서와 코드에 존재
- 2명 대화 파일을 넣으면 화자별 wav가 생성되는 구조
- 로컬에서 모델 추론이 당장 안 되더라도 코드와 문서는 완성되어 있어야 함

# 작업 순서

1. 프로젝트 파일 골격 생성
2. CLI와 타입/인자 정의
3. diarize / overlap / export(pad, concat) 구현
4. README와 .env.example 작성
5. `--help` 출력 기준으로 빠진 옵션이 없는지 점검
6. 간단한 더미 구간 데이터로 export 함수만이라도 검증 가능하면 검증

지금 바로 프로젝트 파일을 생성하고, 빠진 파일이 있으면 이어서 채워라.
