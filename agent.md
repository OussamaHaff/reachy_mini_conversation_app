# Reachy Mini Conversation Agent

## Identity
This agent is the software brain for the **Reachy Mini** robot, enabling it to engage in real-time, multimodal interactions with humans. It combines conversational AI with physical embodiment.

## Core Capabilities
- **Speech**: Real-time bidirectional audio conversation using OpenAI's Realtime API and `fastrtc`.
- **Vision**:
    - **Remote**: Sends camera frames to OpenAI's multimodal models for analysis via the `camera` tool.
    - **Local**: Uses SmolVLM2 (via `--local-vision`) for on-device scene understanding in a background thread.
    - **Tracking**: Face tracking via YOLO or MediaPipe to maintain eye contact (updates `MovementManager` offsets).
- **Motion & Expression**:
    - **Primary Moves**: Sequential actions like `dance`, `play_emotion`, `goto_target`.
    - **Secondary Moves**: Additive offsets for `head_wobbler` (speech-reactive) and `face_tracking`.
    - **Idle Behavior**: Automatically triggers `BreathingMove` or prompts the LLM for creative actions when inactive.

## Architecture
The app follows a layered architecture connecting the user, AI services, and robot hardware:

1.  **Connection Layer**: `OpenaiRealtimeHandler` manages WebSocket connections to OpenAI and audio streams via `fastrtc`.
2.  **Perception Layer**: Handles microphone input (resampling/reshaping) and camera frames.
3.  **Decision Layer (LLM)**: The OpenAI Realtime model receives audio/images, decides on text/audio responses, and calls **tools**.
4.  **Action Layer**: `MovementManager` runs a 100Hz control loop to fuse primary moves and secondary offsets into a single `set_target` command for the robot.

## Configuration
Configuration is loaded from `.env` files via `src/reachy_mini_conversation_app/config.py`.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Required**. OpenAI API key. | - |
| `MODEL_NAME` | OpenAI Realtime model name. | `gpt-realtime` |
| `REACHY_MINI_CUSTOM_PROFILE` | Name of the active personality profile folder. | `None` (uses `default`) |
| `LOCAL_VISION_MODEL` | Hugging Face model ID for local vision. | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| `HF_HOME` | Cache directory for local models. | `./cache` |

## Tooling System
Tools are the mechanism by which the LLM interacts with the robot and the world.

### Definition
Tools must inherit from the `Tool` base class in `src/reachy_mini_conversation_app/tools/core_tools.py` and implement:
- `name`: Unique identifier.
- `description`: Natural language description for the LLM.
- `parameters_schema`: JSON Schema defining arguments.
- `async def __call__(self, deps: ToolDependencies, **kwargs)`: Execution logic.

**`ToolDependencies`** provides access to:
- `reachy_mini`: Direct robot control.
- `movement_manager`: High-level motion control.
- `camera_worker`: Frame capture.
- `head_wobbler`: Speech animation control.

### Registration & Loading
Tools are loaded dynamically based on the active profile's `tools.txt`.
1.  **Profile Tools**: Searched first in `src/reachy_mini_conversation_app/profiles/<profile_name>/`.
2.  **Shared Tools**: Searched second in `src/reachy_mini_conversation_app/tools/`.

### Example Tool
```python
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

class MyTool(Tool):
    name = "my_tool"
    description = "Does something cool."
    parameters_schema = {
        "type": "object",
        "properties": {
            "intensity": {"type": "number"}
        }
    }

    async def __call__(self, deps: ToolDependencies, intensity: float = 1.0) -> dict:
        # Use deps.reachy_mini or deps.movement_manager here
        return {"status": "success", "result": "done"}
```

## Profiles
The agent's personality is modular.
-   **Location**: `src/reachy_mini_conversation_app/profiles/`
-   **Structure**:
    -   `instructions.txt`: System prompt. Supports `[include/path]` syntax for composition.
    -   `tools.txt`: List of allowed tools (one per line).
    -   `*.py`: Custom tool implementations.

## Movement Logic (`moves.py`)
-   **Control Loop**: Runs at ~100Hz.
-   **Composition**: Final Pose = Primary Move (interpolated) + Speech Offsets + Face Tracking Offsets.
-   **Listening Mode**: Freezes antenna movements to reduce noise/distraction while the user speaks.
-   **Breathing**: Automatically engaged after `idle_inactivity_delay` (0.3s) if no other move is active.

## Development
-   **Package Manager**: `uv` is recommended.
-   **Linting**: `ruff check .`
-   **Testing**: `pytest`
-   **Running**: `reachy-mini-conversation-app` (or `python -m reachy_mini_conversation_app.main`)

## Key Files
-   `src/reachy_mini_conversation_app/main.py`: Entry point, initializes managers and Gradio/Console UI.
-   `src/reachy_mini_conversation_app/openai_realtime.py`: Core logic for API interaction and event handling.
-   `src/reachy_mini_conversation_app/tools/core_tools.py`: Tool base class and registry.
-   `src/reachy_mini_conversation_app/moves.py`: `MovementManager` implementation.
-   `src/reachy_mini_conversation_app/vision/processors.py`: Local vision logic (`SmolVLM2`).