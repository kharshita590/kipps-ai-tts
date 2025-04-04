import logging
import os 
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    tts,
    tokenize
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero, kipps


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
print(LIVEKIT_URL)
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by Kipps. Your interface with users will be voice and your task it to be a helpful assistant."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY )

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Since lightning model does not natively support streaming TTS, we can use it with a StreamAdapter (Optional)
    kipps_tts = tts.StreamAdapter(
        tts=kipps.TTS(),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=kipps.TTS(), # replace with smallest_tts to use smallest with streaming
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)

    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            ws_url="wss://whisper-5ckhpu87.livekit.cloud",
        ),
    )
