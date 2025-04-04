from typing import Literal

TTSEncoding = Literal[
    "linear16",
]

TTSModels = Literal["tts_models/en/ljspeech/glow-tts"]
TTSLanguages = Literal["en", "hi"]
TTSContainer = Literal["wav"]