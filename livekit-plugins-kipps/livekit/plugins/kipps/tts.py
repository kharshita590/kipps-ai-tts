from __future__ import annotations
import re
from dataclasses import dataclass
from typing import  List

import aiohttp
from livekit.agents import tts, utils
from .log import logger
from .models import TTSEncoding, TTSLanguages, TTSModels, TTSContainer

NUM_CHANNELS = 1
SENTENCE_END_REGEX = re.compile(r'.*[-.—!?,;:…।|]$')
API_BASE_URL = "https://a712-34-143-151-241.ngrok-free.app"

@dataclass
class _TTSOptions:
    model: TTSModels
    encoding: TTSEncoding
    container: TTSContainer
    sample_rate: int
    language: TTSLanguages


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels = "tts_models/en/ljspeech/glow-tts",
        sample_rate: int = 24000,
        language: TTSLanguages = "en",
        container: TTSContainer = "wav",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
       kipps tts
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding="linear16",
            sample_rate=sample_rate,
            container=container,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self,
        text: str,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            text=text,
            opts=self._opts,
            session=self._ensure_session(),
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using your TTS API endpoint"""

    def __init__(
        self,
        tts: TTS,
        text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=text)
        self._text = text
        self._opts = opts
        self._session = session

    @utils.log_exceptions(logger=logger)
    async def _run(self):
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )
        request_id, segment_id = utils.shortuuid(), utils.shortuuid()

        url = f"{API_BASE_URL}/speak"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {"text": self._text}

        async with self._session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"API error: {resp.status} - {error_text}")
            async for data in resp.content.iter_chunked(1024):
                for frame in bstream.write(data):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=segment_id,
                            frame=frame,
                        )
                    )

        for frame in bstream.flush():
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    segment_id=segment_id,
                    frame=frame,
                )
            )


def _split_into_chunks(text: str, chunk_size: int = 250) -> List[str]:
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text.strip())
            break

        chunk_text = text[:chunk_size]
        last_break_index = -1
        for i in range(len(chunk_text) - 1, -1, -1):
            if SENTENCE_END_REGEX.match(chunk_text[:i + 1]):
                last_break_index = i
                break

        if last_break_index == -1:
            last_space = chunk_text.rfind(' ')
            if last_space != -1:
                last_break_index = last_space
            else:
                last_break_index = chunk_size - 1

        chunks.append(text[:last_break_index + 1].strip())
        text = text[last_break_index + 1:].strip()

    return chunks