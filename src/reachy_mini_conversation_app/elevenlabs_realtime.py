"""ElevenLabs Conversational AI realtime handler for the Reachy Mini conversation app.

Replaces the OpenAI Realtime API integration with ElevenLabs' Conversational AI SDK,
while preserving the full tool-calling infrastructure for robot motor control.
"""

from __future__ import annotations

import json
import uuid
import base64
import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Final, Literal, Optional, Tuple

import cv2
import numpy as np
import gradio as gr
from elevenlabs import ElevenLabs
from elevenlabs.conversational_ai.conversation import (
    AsyncConversation,
    AsyncAudioInterface,
    ClientTools,
    ConversationInitiationData,
)
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    ALL_TOOLS,
    get_tool_specs,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager
from reachy_mini_conversation_app.tools.tool_constants import SystemTool


logger = logging.getLogger(__name__)

# ElevenLabs Conversational AI uses 16 kHz PCM mono for both input and output
ELEVENLABS_SAMPLE_RATE: Final[int] = 16000

# Best ElevenLabs voice model for low-latency conversational AI (Flash v2.5)
ELEVENLABS_TTS_MODEL: Final[str] = "eleven_flash_v2_5"

# Default LLM for the agent (good tool-calling support, cost-effective)
ELEVENLABS_LLM_MODEL: Final[str] = "gpt-4o-mini"

# Map OpenAI voice names → ElevenLabs voice IDs (used when voice.txt contains an OpenAI name)
_OPENAI_TO_ELEVENLABS_VOICE: Dict[str, str] = {
    "cedar": "IKne3meq5aSn9XLyUdCD",   # Charlie (warm, conversational)
    "alloy": "21m00Tcm4TlvDq8ikWAM",   # Rachel (neutral)
    "aria":  "9BWtsMINqrJLrRacOk9x",   # Aria
    "ballad": "TX3LPaxmHKxFdv7VOQHJ",  # Liam
    "verse": "pFZP5JQG7iQjIQuC4Bku",   # Lily
    "sage":  "XrExE9yKIg1WjnnlVkGX",   # Matilda
    "coral": "CwhRBWXzGAHq8TQ4Fs17",   # Roger
}
_DEFAULT_VOICE_ID: Final[str] = "IKne3meq5aSn9XLyUdCD"  # Charlie


def _resolve_voice_id(voice_name: str | None) -> str:
    """Return an ElevenLabs voice_id for the given voice name.

    - If ``voice_name`` looks like an ElevenLabs voice ID (long alphanumeric string), use it directly.
    - If it matches a known OpenAI voice name, return the mapped ElevenLabs ID.
    - Otherwise fall back to the default voice.
    """
    if not voice_name:
        return _DEFAULT_VOICE_ID
    # ElevenLabs voice IDs are long alphanumeric strings (≥20 chars)
    if len(voice_name) >= 20 and voice_name.replace("_", "").isalnum():
        return voice_name
    return _OPENAI_TO_ELEVENLABS_VOICE.get(voice_name, _DEFAULT_VOICE_ID)


def _build_elevenlabs_tool(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI-format tool spec to an ElevenLabs client-tool definition."""
    return {
        "type": "client",
        "name": spec["name"],
        "description": spec.get("description", ""),
        "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
        "expects_response": True,
        "execution_mode": "immediate",
    }


# ---------------------------------------------------------------------------
# Custom audio interface bridging fastrtc ↔ ElevenLabs
# ---------------------------------------------------------------------------

class FastRTCAudioInterface(AsyncAudioInterface):
    """Bridges fastrtc audio with ElevenLabs ``AsyncConversation``.

    - Microphone audio arrives via :meth:`feed_audio` (called from :meth:`receive`).
    - Agent audio is put into ``output_queue`` inside :meth:`output`.
    - Interruptions clear audio-only items from the queue.
    """

    def __init__(
        self,
        output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]",
        head_wobbler: Any | None = None,
    ) -> None:
        self._input_callback: Callable[[bytes], None] | None = None
        self._output_queue = output_queue
        self._head_wobbler = head_wobbler

    # --- AsyncAudioInterface contract ---

    async def start(self, input_callback: Callable[[bytes], None]) -> None:
        """Store the ElevenLabs input callback; called before the session starts."""
        self._input_callback = input_callback

    async def stop(self) -> None:
        """Release the callback; called after the session ends."""
        self._input_callback = None

    async def output(self, audio: bytes) -> None:
        """Receive PCM audio from ElevenLabs and forward to the speaker queue."""
        if not audio:
            return
        # Feed head wobbler (expects base64-encoded PCM, same as OpenAI delta)
        if self._head_wobbler is not None:
            try:
                self._head_wobbler.feed(base64.b64encode(audio).decode("utf-8"))
            except Exception:
                pass
        arr = np.frombuffer(audio, dtype=np.int16)
        await self._output_queue.put((ELEVENLABS_SAMPLE_RATE, arr.reshape(1, -1)))

    async def interrupt(self) -> None:
        """Drain audio-only items from the queue when the user interrupts the agent."""
        saved: list[AdditionalOutputs] = []
        while not self._output_queue.empty():
            try:
                item = self._output_queue.get_nowait()
                if isinstance(item, AdditionalOutputs):
                    saved.append(item)
            except asyncio.QueueEmpty:
                break
        for item in saved:
            await self._output_queue.put(item)

    # --- fastrtc integration ---

    def feed_audio(self, pcm_bytes: bytes) -> None:
        """Push raw 16-bit PCM bytes into ElevenLabs (called from ``receive``)."""
        if self._input_callback is not None:
            self._input_callback(pcm_bytes)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

class ElevenLabsRealtimeHandler(AsyncStreamHandler):
    """ElevenLabs Conversational AI handler for fastrtc ``Stream``.

    Drop-in replacement for ``OpenaiRealtimeHandler``:
    * Same ``AsyncStreamHandler`` interface (``start_up``, ``receive``, ``emit``, ``shutdown``).
    * Preserves the full tool-calling infrastructure (``BackgroundToolManager``, ``ToolDependencies``).
    * Dynamically creates / updates the ElevenLabs agent with the current profile's instructions
      and tool specs so no manual dashboard configuration is required.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=ELEVENLABS_SAMPLE_RATE,
            input_sample_rate=ELEVENLABS_SAMPLE_RATE,
        )
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Audio/conversation state
        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()
        self._audio_interface: FastRTCAudioInterface | None = None
        self.conversation: AsyncConversation | None = None
        self.tool_manager = BackgroundToolManager()

        # Activity tracking for idle signals
        self.last_activity_time: float = 0.0
        self.start_time: float = 0.0
        self.is_idle_tool_call: bool = False

        # Partial transcript debouncing
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_transcript_sequence: int = 0
        self.partial_debounce_delay: float = 0.5

        # API key tracking
        self._shutdown_requested: bool = False
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

    # ------------------------------------------------------------------
    # fastrtc interface
    # ------------------------------------------------------------------

    def copy(self) -> "ElevenLabsRealtimeHandler":
        """Create a new handler copy (called by fastrtc for each new connection)."""
        return ElevenLabsRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Resolve the API key, ensure the agent exists, and run the conversation session."""
        api_key = config.ELEVENLABS_API_KEY

        if self.gradio_mode and not api_key:
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_key = args[3] if len(args) > 3 and isinstance(args[3], str) and args[3] else None
            if textbox_key:
                api_key = textbox_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_key
            else:
                api_key = config.ELEVENLABS_API_KEY

        if not api_key or not api_key.strip():
            logger.warning("ELEVENLABS_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            api_key = "DUMMY"

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()

        try:
            agent_id = await asyncio.get_event_loop().run_in_executor(
                None, self._ensure_agent_sync, api_key
            )
        except Exception as exc:
            logger.error("Failed to ensure ElevenLabs agent: %s", exc)
            return

        await self._run_conversation_session(api_key, agent_id)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive microphone audio and forward to ElevenLabs (16 kHz PCM)."""
        if self._audio_interface is None:
            return

        input_sample_rate, audio_frame = frame

        # Flatten / pick mono channel
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            audio_frame = audio_frame[:, 0] if audio_frame.shape[1] > 1 else audio_frame[:, 0]
        else:
            audio_frame = audio_frame.flatten()

        # Resample to 16 kHz if the source rate differs
        if input_sample_rate != ELEVENLABS_SAMPLE_RATE:
            n_out = int(len(audio_frame) * ELEVENLABS_SAMPLE_RATE / input_sample_rate)
            audio_frame = resample(audio_frame, n_out)

        audio_frame = audio_to_int16(audio_frame)
        self._audio_interface.feed_audio(audio_frame.tobytes())

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return next audio chunk or UI message for the speaker / chatbot."""
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as exc:
                logger.warning("Idle signal failed: %s", exc)
            self.last_activity_time = asyncio.get_event_loop().time()

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Gracefully close the ElevenLabs session and release resources."""
        self._shutdown_requested = True

        if self.conversation is not None:
            try:
                await self.conversation.end_session()
            except Exception:
                pass
            self.conversation = None

        await self.tool_manager.shutdown()

        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------
    # Personality (profile) switching – called from Gradio / headless UI
    # ------------------------------------------------------------------

    async def apply_personality(self, profile: str | None) -> str:
        """Switch personality profile at runtime.

        The new instructions take effect on the next session restart because
        ElevenLabs does not support live system-prompt updates.
        """
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            logger.info("Set custom profile to %r", profile)

            if self.conversation is not None:
                try:
                    await self.conversation.end_session()
                except Exception:
                    pass
            return "Applied personality. Will take effect on reconnection."
        except Exception as exc:
            logger.error("Error applying personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

    async def get_available_voices(self) -> list[str]:
        """Return the list of available ElevenLabs-mapped voice names."""
        return list(_OPENAI_TO_ELEVENLABS_VOICE.keys())

    # ------------------------------------------------------------------
    # Idle signal
    # ------------------------------------------------------------------

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Inject an idle nudge into the conversation via a contextual update."""
        logger.debug("Sending idle signal, idle=%.1fs", idle_duration)
        self.is_idle_tool_call = True

        if self.conversation is None:
            return

        msg = (
            f"[No activity for {idle_duration:.0f}s] "
            "You've been idle for a while. Feel free to get creative – "
            "dance, show an emotion, look around, do nothing, or just be yourself!"
        )
        try:
            # ElevenLabs SDK exposes send_contextual_update for non-interrupting text injection
            if hasattr(self.conversation, "send_contextual_update"):
                await self.conversation.send_contextual_update(msg)
            elif hasattr(self.conversation, "send_user_message"):
                await self.conversation.send_user_message(msg)
            else:
                logger.debug("Idle signal: no suitable injection method found in SDK version")
        except Exception as exc:
            logger.warning("Failed to send idle signal: %s", exc)

    # ------------------------------------------------------------------
    # Internal: callbacks from ElevenLabs
    # ------------------------------------------------------------------

    async def _on_agent_response(self, text: str) -> None:
        """Called when the agent produces a complete utterance."""
        logger.info("Agent response: %s", text[:200])
        self.last_activity_time = asyncio.get_event_loop().time()
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))

    async def _on_user_transcript(self, transcript: str) -> None:
        """Called when user speech is fully transcribed."""
        logger.info("User transcript: %s", transcript[:200])

        # Cancel pending partial-transcript debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

    async def _emit_debounced_partial(self, transcript: str, sequence: int) -> None:
        """Emit a partial transcript to the UI after a short debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)
            if self.partial_transcript_sequence == sequence:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user_partial", "content": transcript})
                )
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # Internal: core session
    # ------------------------------------------------------------------

    async def _run_conversation_session(self, api_key: str, agent_id: str) -> None:
        """Establish and run the ElevenLabs conversation WebSocket session."""
        self._audio_interface = FastRTCAudioInterface(
            output_queue=self.output_queue,
            head_wobbler=self.deps.head_wobbler,
        )

        client_tools = self._build_client_tools()

        # Start the background tool manager (cleanup + listener tasks)
        self.tool_manager.start_up(tool_callbacks=[])

        # Override system prompt (and optionally voice) per session
        conversation_config_override: Dict[str, Any] = {
            "agent": {
                "prompt": {"prompt": get_session_instructions()},
                "first_message": "Hello! I'm Reachy Mini. How can I help you?",
            },
        }

        # Use a synchronous ElevenLabs client here because AsyncConversation's
        # __init__ typically wraps a sync client to build the signed URL.
        sync_client = ElevenLabs(api_key=api_key)

        # Wrap callbacks so they create asyncio tasks (ElevenLabs may call them synchronously)
        loop = asyncio.get_event_loop()

        def _agent_cb(text: str) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._on_agent_response(text))
            )

        def _user_cb(transcript: str) -> None:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._on_user_transcript(transcript))
            )

        self.conversation = AsyncConversation(
            client=sync_client,
            agent_id=agent_id,
            requires_auth=True,
            audio_interface=self._audio_interface,
            config=ConversationInitiationData(
                conversation_config_override=conversation_config_override,
            ),
            client_tools=client_tools,
            callback_agent_response=_agent_cb,
            callback_user_transcript=_user_cb,
        )

        try:
            logger.info(
                "Starting ElevenLabs conversation session (agent=%s, profile=%r)",
                agent_id,
                getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
            )
            await self.conversation.start_session()
            await self.conversation.wait_for_session_end()
            logger.info("ElevenLabs conversation session ended")
        except Exception as exc:
            logger.error("ElevenLabs session error: %s", exc)
        finally:
            self.conversation = None
            self._audio_interface = None
            await self.tool_manager.shutdown()

    # ------------------------------------------------------------------
    # Internal: tool registration
    # ------------------------------------------------------------------

    def _build_client_tools(self) -> ClientTools:
        """Register all profile tools as ElevenLabs ``ClientTools`` async handlers."""
        client_tools = ClientTools()
        _system_tool_names = {t.value for t in SystemTool}

        for tool_name in list(ALL_TOOLS.keys()):
            name = tool_name  # capture for closure

            async def _handler(params: dict, _name: str = name) -> str:
                is_idle = self.is_idle_tool_call
                if is_idle:
                    self.is_idle_tool_call = False

                logger.info("Tool call: %s params=%s is_idle=%s", _name, params, is_idle)

                # Show pending notification in the chatbot UI
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": f"🛠️ Calling tool {_name} with {json.dumps(params)}",
                        }
                    )
                )

                # Execute the tool
                result = await dispatch_tool_call(_name, json.dumps(params), self.deps)

                # Special-case camera: ElevenLabs cannot receive images, but we can
                # display the captured frame in the Gradio UI and swap the result.
                if _name == "camera" and isinstance(result, dict) and "b64_im" in result:
                    if self.deps.camera_worker is not None:
                        np_img = self.deps.camera_worker.get_latest_frame()
                        if np_img is not None:
                            rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": gr.Image(value=rgb_frame)})
                            )
                    result = {
                        "description": (
                            "Image captured. "
                            "Run with --local-vision for AI-based visual analysis."
                        )
                    }

                # Emit the tool result to the chatbot UI
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(result),
                            "metadata": {"title": f"🛠️ Used tool {_name}", "status": "done"},
                        }
                    )
                )

                # Re-sync head wobble after any tool that may have taken time
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.reset()

                return json.dumps(result)

            client_tools.register(name=tool_name, fn=_handler, is_async=True)

        return client_tools

    # ------------------------------------------------------------------
    # Internal: agent management
    # ------------------------------------------------------------------

    def _ensure_agent_sync(self, api_key: str) -> str:
        """Create or update the ElevenLabs agent (synchronous, run in thread executor).

        Returns the agent_id to use for the conversation.
        """
        client = ElevenLabs(api_key=api_key)

        el_tools = [_build_elevenlabs_tool(spec) for spec in get_tool_specs()]
        instructions = get_session_instructions()
        voice_id = _resolve_voice_id(get_session_voice())

        agent_config: Dict[str, Any] = {
            "name": "Reachy Loco",
            "conversation_config": {
                "agent": {
                    "prompt": {
                        "prompt": instructions,
                        "llm": ELEVENLABS_LLM_MODEL,
                    },
                    "first_message": "Hello! I'm Reachy Mini. How can I help you?",
                    "language": "en",
                    "tools": el_tools,
                },
                "tts": {
                    "model_id": ELEVENLABS_TTS_MODEL,
                    "voice_id": voice_id,
                },
                "turn": {
                    "turn_timeout": 7,
                },
            },
        }

        existing_id = config.ELEVENLABS_AGENT_ID
        if existing_id and existing_id.strip():
            logger.info("Updating existing ElevenLabs agent: %s", existing_id)
            try:
                client.conversational_ai.agents.update(
                    agent_id=existing_id,
                    **agent_config,
                )
                logger.info("Agent updated: %s", existing_id)
                return existing_id
            except Exception as exc:
                logger.warning(
                    "Failed to update agent %s (%s). Creating a new one.", existing_id, exc
                )

        logger.info("Creating new ElevenLabs agent")
        agent = client.conversational_ai.agents.create(**agent_config)
        new_id: str = agent.agent_id
        logger.info("Created ElevenLabs agent: %s", new_id)

        # Persist the new agent ID so future runs reuse it
        config.ELEVENLABS_AGENT_ID = new_id  # type: ignore[attr-defined]
        self._persist_agent_id_to_env(new_id)
        return new_id

    def _persist_agent_id_to_env(self, agent_id: str) -> None:
        """Write ELEVENLABS_AGENT_ID to the `.env` file when possible."""
        try:
            # Try instance path first, fall back to cwd
            candidate_dirs = []
            if self.instance_path:
                candidate_dirs.append(Path(self.instance_path))
            candidate_dirs.append(Path.cwd())

            for base_dir in candidate_dirs:
                env_path = base_dir / ".env"
                if not env_path.exists():
                    continue
                lines = env_path.read_text(encoding="utf-8").splitlines()
                replaced = False
                for i, line in enumerate(lines):
                    if line.strip().startswith("ELEVENLABS_AGENT_ID="):
                        lines[i] = f"ELEVENLABS_AGENT_ID={agent_id}"
                        replaced = True
                        break
                if not replaced:
                    lines.append(f"ELEVENLABS_AGENT_ID={agent_id}")
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                logger.info("Persisted ELEVENLABS_AGENT_ID=%s to %s", agent_id, env_path)
                return

            # No .env found; just log
            logger.info(
                "No .env file found; ELEVENLABS_AGENT_ID=%s not persisted (add it manually).",
                agent_id,
            )
        except Exception as exc:
            logger.warning("Could not persist ELEVENLABS_AGENT_ID: %s", exc)
