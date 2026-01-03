import asyncio
import base64
import json
import logging
import random
from typing import Any, Final, Literal, Optional, Tuple, List, Dict
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from scipy.signal import resample
from numpy.typing import NDArray
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from websockets.exceptions import ConnectionClosed

from google import genai
from google.genai.types import (
    LiveConnectConfig,
    FunctionDeclaration,
    Schema,
    Type,
    Tool,
    VoiceConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    Content,
    Part,
    Blob,
    FunctionResponse
)

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)

logger = logging.getLogger(__name__)

# FastRTC / Audio constants
INPUT_SAMPLE_RATE: Final[Literal[16000]] = 16000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


def _map_json_schema_type_to_gemini_type(json_type: str) -> Type:
    """Map JSON schema types to Gemini Schema types."""
    mapping = {
        "string": Type.STRING,
        "number": Type.NUMBER,
        "integer": Type.INTEGER,
        "boolean": Type.BOOLEAN,
        "array": Type.ARRAY,
        "object": Type.OBJECT,
    }
    return mapping.get(json_type, Type.STRING)


def _convert_schema(json_schema: Dict[str, Any]) -> Schema:
    """Recursively convert a JSON schema dict to a Gemini Schema object."""
    schema_type = _map_json_schema_type_to_gemini_type(json_schema.get("type", "string"))
    
    properties = {}
    if "properties" in json_schema:
        for key, prop in json_schema["properties"].items():
            properties[key] = _convert_schema(prop)
            
    required = json_schema.get("required", [])
    
    return Schema(
        type=schema_type,
        properties=properties if properties else None,
        required=required if required else None,
        description=json_schema.get("description"),
        enum=json_schema.get("enum")
    )


def _convert_tool_specs_to_gemini(tool_specs: List[Dict[str, Any]]) -> List[Tool]:
    """Convert OpenAI-style tool specs to Gemini Tool objects."""
    function_declarations = []
    
    for spec in tool_specs:
        if spec["type"] != "function":
            continue
            
        func_spec = spec
        # OpenAI spec structure: {"type": "function", "name": ..., "description": ..., "parameters": ...}
        # Sometimes nested under "function" key depending on version, but core_tools.py returns flat dicts 
        # based on the `spec()` method:
        # { "type": "function", "name": self.name, "description": self.description, "parameters": self.parameters_schema }
        
        name = func_spec["name"]
        description = func_spec["description"]
        parameters = func_spec["parameters"]
        
        # Convert JSON schema parameters to Gemini Schema
        # Note: Gemini expects the root of parameters to be an OBJECT schema
        input_schema = _convert_schema(parameters)
        
        function_declarations.append(
            FunctionDeclaration(
                name=name,
                description=description,
                parameters=input_schema
            )
        )
        
    if not function_declarations:
        return []
        
    return [Tool(function_declarations=function_declarations)]


class GeminiRealtimeHandler(AsyncStreamHandler):
    """A Gemini Live API handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Output queue for fastrtc
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.client: Optional[genai.Client] = None
        self.session: Any = None  # Gemini Live session
        
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        
        self._connected_event: asyncio.Event = asyncio.Event()
        self._shutdown_requested: bool = False
        
        # Audio buffer for receiving from microphone
        self._input_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def copy(self) -> "GeminiRealtimeHandler":
        """Create a copy of the handler."""
        return GeminiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)
    
    async def apply_personality(self, profile: str | None) -> str:
        """Apply personality changes.
        
        For Gemini, we might need to restart the session to change system instructions reliably.
        """
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            
            # Restart session to apply new instructions
            await self._restart_session()
            return f"Applied personality '{profile}' and restarted session."
        except Exception as e:
            logger.error(f"Failed to apply personality: {e}")
            return f"Error: {e}"

    async def start_up(self) -> None:
        """Start the handler."""
        api_key = config.GOOGLE_API_KEY
        if not api_key:
             logger.error("GOOGLE_API_KEY is missing!")
             # If strictly needed, handle UI prompt for key like in OpenAI handler
             # For now, assuming env var is set.
        
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        
        # Start the session loop in background
        asyncio.create_task(self._run_realtime_session())

    async def _restart_session(self) -> None:
        """Restart the Gemini session."""
        self._shutdown_requested = True
        if self.session:
            # How to close? The session context manager handles it.
            # We signal loop to break.
            pass
            
        # Give it a moment to close
        await asyncio.sleep(0.5)
        self._shutdown_requested = False
        asyncio.create_task(self._run_realtime_session())

    async def _run_realtime_session(self) -> None:
        """Manage the Gemini Live session."""
        model_name = config.MODEL_NAME
        
        while not self._shutdown_requested:
            # Get tools
            tool_specs = get_tool_specs()
            gemini_tools = _convert_tool_specs_to_gemini(tool_specs)
            
            # Get instructions
            instructions = get_session_instructions()
            
            # Voice config
            voice_name = "Puck" # Default Gemini voice, mapping needed if we want to support 'alloy' etc names
            # Or read from get_session_voice() and map it
            requested_voice = get_session_voice()
            # Simple mapping or fallback
            voice_map = {
                "alloy": "Puck", "echo": "Kore", "fable": "Fenrir", 
                "onyx": "Kore", "nova": "Aoede", "shimmer": "Puck"
            }
            voice_name = voice_map.get(requested_voice, "Puck")

            config_live = LiveConnectConfig(
                response_modalities=["AUDIO"], # We want audio back
                system_instruction=Content(parts=[Part(text=instructions)]),
                tools=gemini_tools,
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name="Leda"
                        )
                    )
                )
            )

            try:
                logger.info(f"Connecting to Gemini Live: {model_name}")
                async with self.client.aio.live.connect(model=model_name, config=config_live) as session:
                    self.session = session
                    self._connected_event.set()
                    logger.info("Gemini Live connected.")
                    
                    # Start sender task
                    sender_task = asyncio.create_task(self._send_audio_loop())
                    
                    # Receiver loop
                    async for response in session.receive():
                        if self._shutdown_requested:
                            break
                            
                        server_content = response.server_content
                        if server_content is None:
                            continue
                            
                        model_turn = server_content.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                # Handle Audio
                                if part.inline_data:
                                    # Gemini audio is raw PCM usually, or we need to check mime_type
                                    # Default is PCM 24kHz usually for Live API?
                                    # Docs say: "Audio is returned as raw PCM 16-bit, 24kHz, Little Endian"
                                    audio_bytes = part.inline_data.data
                                    
                                    # Feed wobbler
                                    if self.deps.head_wobbler:
                                        # Wobbler expects base64 encoded string of raw bytes?
                                        # Let's check openai_realtime.py: 
                                        # self.deps.head_wobbler.feed(event.delta) -> event.delta is b64 string
                                        # So we encode it.
                                        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                                        self.deps.head_wobbler.feed(b64_audio)
                                    
                                    # Queue for playback
                                    # Audio is int16, 24kHz.
                                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
                                    await self.output_queue.put((24000, audio_np))
                                    self.last_activity_time = asyncio.get_event_loop().time()
                                    
                                # Handle Text (Transcript)
                                if part.text:
                                    logger.debug(f"Gemini text: {part.text}")
                                    await self.output_queue.put(
                                        AdditionalOutputs({"role": "assistant", "content": part.text})
                                    )
                                    
                                # Handle Function Calls
                                if part.executable_code:
                                    # Code execution not supported, we use function calls
                                    pass
                                    
                                if part.function_call:
                                    fc = part.function_call
                                    tool_name = fc.name
                                    args = fc.args
                                    # args is a dict
                                    
                                    logger.info(f"Gemini Tool Call: {tool_name}")
                                    
                                    # Convert args to JSON string for dispatch
                                    args_json = json.dumps(args)
                                    
                                    # Dispatch
                                    try:
                                        result = await dispatch_tool_call(tool_name, args_json, self.deps)
                                    except Exception as e:
                                        result = {"error": str(e)}
                                        
                                    # Send result back
                                    # Gemini expects tool_response
                                    await session.send_tool_response(
                                        function_responses=[FunctionResponse(
                                            name=tool_name,
                                            response=result
                                        )]
                                    )
                                    
                                    # Show in UI
                                    await self.output_queue.put(
                                        AdditionalOutputs(
                                            {
                                                "role": "assistant",
                                                "content": json.dumps(result),
                                                "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                                            },
                                        ),
                                    )
                                    
                                    # Handle Camera Image (Similar to OpenAI handler)
                                    if tool_name == "camera" and "b64_im" in result:
                                        b64_im = result["b64_im"]
                                        # Gemini supports sending images.
                                        # We can send it as user input?
                                        # Yes, send message with image.
                                        img_bytes = base64.b64decode(b64_im)
                                        await session.send_realtime_input(
                                            media={"mime_type": "image/jpeg", "data": img_bytes}
                                        )
                                        await session.send_realtime_input(text="Here is what I see.")
                                        
                                        # Display in UI
                                        if self.deps.camera_worker:
                                            np_img = self.deps.camera_worker.get_latest_frame()
                                            if np_img is not None:
                                                rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                                                img = gr.Image(value=rgb_frame)
                                                await self.output_queue.put(
                                                    AdditionalOutputs({"role": "assistant", "content": img})
                                                )
                                    
                        # Handle turn complete or other signals?
                        # Gemini handles VAD automatically.
                        
                    sender_task.cancel()
                    
            except Exception as e:
                logger.error(f"Gemini session error: {e}")
                if not self._shutdown_requested:
                    logger.info("Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)
            finally:
                if 'sender_task' in locals() and not sender_task.done():
                    sender_task.cancel()
                self.session = None
                self._connected_event.clear()

    async def _send_audio_loop(self) -> None:
        """Loop to send audio from input queue to Gemini."""
        while True:
            try:
                chunk = await self._input_audio_queue.get()
                if self.session and not self._shutdown_requested:
                    try:
                        # Send audio chunk using media kwarg
                        await self.session.send_realtime_input(
                            media={"mime_type": "audio/pcm;rate=16000", "data": chunk}
                        )
                    except ConnectionClosed:
                        logger.debug("Gemini WebSocket closed in sender loop.")
                        break
                    except Exception as e:
                        # If session is closing, we might get other errors
                        if self._shutdown_requested:
                            break
                        logger.error(f"Error sending audio to Gemini: {e}")
                else:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audio loop: {e}")
                
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio from microphone."""
        sample_rate, audio_data = frame
        
        # Reshape/Resample similar to OpenAI handler
        if audio_data.ndim == 2:
             if audio_data.shape[1] > audio_data.shape[0]:
                 audio_data = audio_data.T
             if audio_data.shape[1] > 1:
                 audio_data = audio_data[:, 0]
                 
        if sample_rate != INPUT_SAMPLE_RATE:
             audio_data = resample(audio_data, int(len(audio_data) * INPUT_SAMPLE_RATE / sample_rate))
             
        audio_data = audio_to_int16(audio_data)
        
        # Put bytes into queue for sender loop
        self._input_audio_queue.put_nowait(audio_data.tobytes())

    async def _safe_send_message(self, text: str) -> None:
        """Safely send a text message to the session."""
        if self.session and not self._shutdown_requested:
            try:
                await self.session.send_realtime_input(text=text)
            except Exception as e:
                logger.warning(f"Failed to send message: {e}")

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio to speaker."""
        # Handle idle check similar to OpenAI handler
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
             # Send idle prompt to Gemini
             if self.session:
                 # We can just send text
                 timestamp_msg = f"You've been idle for {idle_duration:.1f}s. Do something creative!"
                 # Fire and forget
                 asyncio.create_task(self._safe_send_message(timestamp_msg))
                 self.last_activity_time = asyncio.get_event_loop().time()
        
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        # Logic to close session handled in main loop or here
