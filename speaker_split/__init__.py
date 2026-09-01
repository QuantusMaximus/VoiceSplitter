"""화자 분리 Python CLI 도구 (VoiceSplitter / speaker_split)

pyannote.audio를 이용해 오디오 파일에서 화자를 분리하고 개별 트랙(pad, concat)으로 저장합니다.
"""

import os

# Windows 심볼릭 링크 경고 억제
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# huggingface_hub use_auth_token 호환성 패치
try:
    import huggingface_hub
    _orig_download = huggingface_hub.hf_hub_download

    def _compat_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            tok = kwargs.pop("use_auth_token")
            if "token" not in kwargs:
                kwargs["token"] = tok
        return _orig_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _compat_download
except Exception:
    pass

__version__ = "0.1.0"
__all__ = ["__version__"]

