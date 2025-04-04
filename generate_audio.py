import asyncio
import aiohttp
import wave
from livekit.plugins import kipps

from dotenv import load_dotenv
load_dotenv()

async def test_tts():
    async with aiohttp.ClientSession() as session:
        tts = kipps.TTS(
            http_session=session  
        )
        frames = []
        async for audio in tts.synthesize(text="Hello, this is a test of the TTS system."):
            frames.append(audio.frame)
        with wave.open("output.wav", "wb") as wav_file:
            wav_file.setnchannels(1)  
            wav_file.setsampwidth(2)  
            wav_file.setframerate(tts._opts.sample_rate)

            for frame in frames:
                frame_bytes = frame.data.tobytes()
                wav_file.writeframes(frame_bytes)

        print("Audio saved to output.wav")

if __name__ == "__main__":
    asyncio.run(test_tts())