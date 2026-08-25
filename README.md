#Advanced AI Driving Assistant

An Advanced Driver Assistance System (ADAS) combined with a real-time AI Voice Assistant, designed to run natively on a dashcam or smartphone feed. This project processes video streams in real-time, detects objects and hazards on the road, calculates distances using perspective transformations, and provides intelligent, conversational audio feedback to the driver.

<img width="1072" height="603" alt="image" src="https://github.com/user-attachments/assets/25872c54-51f9-4382-a542-4bc3a2998e5f" />


## Features

- **Real-Time Object Detection**: Uses a fine-tuned YOLO model to detect vehicles, pedestrians, animals, potholes, and traffic signs.
- **Dynamic Lane Detection**: Maps the drivable area using Hough lines and polynomial curve fitting.
- **Collision Warning System**: Calculates Time-To-Collision (TTC) and flags immediate hazards directly in the driver's path.
- **AI Co-Pilot (Push-to-Talk)**: Speak directly to the system using your phone's microphone. The AI sees what the camera sees and can answer questions about the driving environment.
- **Mobile-Friendly Web Dashboard**: A glassmorphic web UI served via Flask, allowing you to mount your phone on the dashboard to view the HUD and toggle settings.

---

## Core Concepts

### 1. Bird's Eye View (BEV) & Perspective Mapping
To accurately calculate the distance to a vehicle on a 2D screen, the system uses **Inverse Perspective Mapping (IPM)**. 
Because a camera captures the world in perspective (where distant objects appear smaller and higher up), we compute a transformation matrix that "warps" the camera's view into a top-down, Bird's Eye View. By mapping the bottom edge of a detected vehicle's bounding box onto this top-down plane, we can estimate its real-world distance in meters, allowing for accurate collision warnings.

### 2. Lane Detection & Safe Corridors
The engine applies a region-of-interest mask over the road surface, converts the image to grayscale, and applies edge detection (Canny). It then uses the **Hough Transform** to find continuous lines representing the lane boundaries. 
These boundaries are merged into a green "safe corridor" polygon. Any object (like a pothole or pedestrian) that enters this specific polygon is immediately prioritized by the system as a higher-risk hazard compared to objects on the sidewalk.

### 3. Decoupled Architecture
To ensure the camera feed never stutters, the system is highly multithreaded:
- **Engine Thread**: Runs the OpenCV YOLO model and lane detection as fast as possible.
- **Encoder Thread**: Asynchronously compresses the raw frames into JPEG chunks for the web stream without slowing down the AI detection.
- **Co-Pilot Thread**: Runs completely independently, handling Groq Whisper transcriptions, LLM vision queries, and Text-To-Speech (TTS) playback in the background.

---

## Project Structure

```text
ADAS Version 2/
├── src/
│   ├── adas_engine.py       # Core OpenCV pipeline, YOLO inference, Lane/BEV math
│   ├── adas_copilot.py      # Groq AI Voice Assistant and audio processing logic
│   ├── server.py            # Flask web server and MJPEG video streaming
│   ├── evaluate_thresholds.py # Utility script for testing model confidence
│   └── templates/           # HTML/CSS UI for the mobile web app
├── scripts/
│   ├── run.bat              # Command-line launcher for the server
│   └── StartApp.vbs         # Invisible desktop shortcut launcher
├── config/
│   └── config.json          # Remembers your selected camera/video source
├── models/                  # (Not tracked) Stores the YOLO .pt / .onnx weights
└── videos/                  # (Not tracked) Sample dashcam footage for testing
```

## How to Run

1. Ensure your virtual environment (`.venv`) is set up with all dependencies (`ultralytics`, `opencv-python`, `flask`, `groq`, `edge-tts`).
2. Ensure you have **ffmpeg** installed on your system PATH (required for audio format conversions).
3. Double-click **`scripts/StartApp.vbs`** to start the backend silently.
4. Open the provided `http://127.0.0.1:5000` link in your browser (or connect via your phone on the same Wi-Fi network).
5. Select your camera source or test video, and hit **Launch Co-Pilot**.

### Voice Controls
On the web dashboard, press and hold the **Ask Co-Pilot** button, ask a question like *"Is it safe to change lanes?"*, and release. The system will transcribe your voice using Whisper, analyze the current video frame using a Vision LLM, and speak the answer back to you.
