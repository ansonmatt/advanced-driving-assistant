import os
import queue
import threading
import time
import asyncio
import uuid
import base64
import cv2
import edge_tts
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from groq import Groq
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sounddevice as sd
import soundfile as sf

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")
load_dotenv(dotenv_path=env_path, override=True)
# Import the engine logic
from adas_engine import get_args_parser, run_engine

# A thread-safe queue to pass events to the AI
event_queue = queue.Queue()

# Global subtitle state
current_subtitle = ""
subtitle_time = 0.0

# Audio recording state
is_recording = False
audio_buffer = []
audio_stream = None

# Latest frame snapshot for vision context (set by engine_callback)
_latest_frame = None
_latest_frame_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_buffer.append(indata.copy())

def trigger_voice_query(audio_bytes: bytes):
    """
    Called by the web server when the user releases the push-to-speak button.
    audio_bytes: raw WebM/Opus audio from the browser (sent via POST).
    """
    global current_subtitle, subtitle_time

    with _latest_frame_lock:
        frame_snapshot = _latest_frame

    # Encode the current frame for vision context
    b64_img = None
    if frame_snapshot is not None:
        _, buf = cv2.imencode('.jpg', frame_snapshot)
        b64_img = base64.b64encode(buf).decode('utf-8')

    # Save browser audio to a temp file that Groq's Whisper can read
    temp_audio_path = f"temp_browser_{uuid.uuid4().hex}.webm"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_bytes)

    event_queue.put({
        'type': 'browser_voice',
        'audio_path': temp_audio_path,
        'image': b64_img
    })

    current_subtitle = "Processing your question..."
    subtitle_time = time.time()


def ai_copilot_worker():
    global current_subtitle, subtitle_time
    
    # Initialize Pygame mixer for audio playback
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[Co-Pilot] Audio mixer init failed: {e}")
        return

    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n[Co-Pilot] WARNING: GROQ_API_KEY environment variable not set!")
        return

    client = Groq(api_key=api_key)

    print("[Co-Pilot] AI voice assistant is online and ready.")
    
    last_speak_time = 0
    cooldown = 5.0 # wait 5 seconds before warning again to avoid spam
    
    system_prompt = (
        "You are an advanced, helpful AI driving co-pilot. You do not drive the car; you only observe and gently advise the driver based on telemetry or dashcam images. "
        "Do not act overconfident or pretend to have more knowledge than provided. Never say phrases like 'stopping now', 'engaging brakes' or 'correcting', as you have no physical control over the vehicle. "
        "Keep your answers concise, natural, and helpful."
    )

    while True:
        event = event_queue.get()
        if event is None:
            break # Exit signal
            
        try:
            warning_text = ""
            
            if event['type'] == 'warning':
                current_time = time.time()
                if current_time - last_speak_time < cooldown:
                    continue
                # Generate warning response
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt + " State the warning. Keep it under 6 words."},
                        {"role": "user", "content": event['data']}
                    ],
                    temperature=0.7,
                    max_tokens=30,
                )
                warning_text = response.choices[0].message.content.strip()
                
            elif event['type'] == 'vision_request':
                current_subtitle = "Thinking..."
                subtitle_time = time.time()
                
                # Generate vision response
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt + " Describe the scene you see in 1 or 2 short sentences. Focus on traffic, weather, and critical objects."},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": "Describe this scene."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{event['data']}"}}
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )
                warning_text = response.choices[0].message.content.strip()

            elif event['type'] == 'voice_chat':
                # 1. Save audio to temporary wav file
                wav_filename = f"temp_query_{uuid.uuid4().hex}.wav"
                sf.write(wav_filename, event['audio'], event['samplerate'])
                
                # 2. Send to Groq Whisper
                with open(wav_filename, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                      file=(wav_filename, file.read()),
                      model="whisper-large-v3-turbo",
                      prompt="The user is asking a question about their driving dashcam video.",
                      language="en",
                    )
                query_text = transcription.text
                try: os.remove(wav_filename)
                except: pass
                
                print(f"[Co-Pilot] Heard: {query_text}")
                
                # 3. Send query + image to Vision model
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": f"The driver asks: '{query_text}'. Answer them based on this dashcam frame."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{event['image']}"}}
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=150,
                )
                warning_text = response.choices[0].message.content.strip()
            elif event['type'] == 'browser_voice':
                # Audio uploaded from browser mic (WebM/Opus file on disk)
                audio_path = event.get('audio_path', '')
                wav_path = audio_path + ".wav"
                current_subtitle = "Listening..."
                subtitle_time = time.time()

                try:
                    # Convert raw WebM from browser to standard WAV using ffmpeg
                    import subprocess
                    subprocess.run([
                        "ffmpeg", "-y", "-i", audio_path, 
                        "-ac", "1", "-ar", "16000", "-f", "wav", wav_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    with open(wav_path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=(os.path.basename(wav_path), f.read()),
                            model="whisper-large-v3-turbo",
                            prompt="The user is asking a question about their driving dashcam video.",
                            language="en",
                        )
                    query_text = transcription.text.strip()
                except Exception as te:
                    query_text = ""
                    print(f"[Co-Pilot] Transcription error: {te}")
                finally:
                    try: os.remove(audio_path)
                    except: pass
                    try: os.remove(wav_path)
                    except: pass

                if not query_text:
                    current_subtitle = "Couldn't hear that. Please try again."
                    subtitle_time = time.time()
                    continue

                print(f"[Co-Pilot] Heard (browser): {query_text}")
                current_subtitle = f"You said: {query_text[:50]}..."
                subtitle_time = time.time()

                # Build message — include vision if we have a frame
                user_content = [{"type": "text", "text": f"The driver asks: '{query_text}'. Answer them concisely."}]
                if event.get('image'):
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{event['image']}"}})
                    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
                else:
                    model_name = "llama-3.1-8b-instant"

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=150,
                )
                warning_text = response.choices[0].message.content.strip()

            if warning_text:
                # Clean up quotes
                warning_text = warning_text.replace('"', '').replace("'", "")
                print(f"\n[Co-Pilot] Says: {warning_text}\n")
                
                # Display subtitle on video frame
                current_subtitle = warning_text
                subtitle_time = time.time()
                
                # Generate high-quality realistic AI speech via Edge TTS
                filename = f"temp_{uuid.uuid4().hex}.mp3"
                communicate = edge_tts.Communicate(warning_text, "en-US-ChristopherNeural", rate="+15%")
                asyncio.run(communicate.save(filename))
                
                # Speak it
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                pygame.mixer.music.unload()
                try: os.remove(filename)
                except: pass
                
                last_speak_time = time.time()

        except Exception as e:
            print(f"[Co-Pilot] AI Error: {e}")
            current_subtitle = "Error connecting to AI..."
            subtitle_time = time.time()

# Track state
last_collision_state = False
last_departure_state = "System Ready"

total_frames = 0
total_collision_alerts = 0
total_lane_departures = 0
vision_cooldown = 0

def engine_callback(frame_idx, stats, detections, frame, key):
    global last_collision_state, last_departure_state
    global total_frames, total_collision_alerts, total_lane_departures
    global vision_cooldown, current_subtitle, subtitle_time
    global is_recording, audio_buffer, audio_stream
    global _latest_frame

    total_frames = frame_idx

    # Always snapshot the latest frame for browser voice queries
    with _latest_frame_lock:
        _latest_frame = frame.copy()

    # 1. Voice Chat Integration (Press 't' to toggle recording)
    if key == ord('t'):
        if not is_recording:
            # Start recording
            is_recording = True
            audio_buffer = []
            audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
            audio_stream.start()
        else:
            # Stop recording and send
            is_recording = False
            if audio_stream:
                audio_stream.stop()
                audio_stream.close()
                audio_stream = None
            
            if audio_buffer:
                audio_data = np.concatenate(audio_buffer, axis=0)
                # Capture frame context
                _, buffer = cv2.imencode('.jpg', frame)
                b64_img = base64.b64encode(buffer).decode('utf-8')
                
                event_queue.put({
                    'type': 'voice_chat', 
                    'audio': audio_data, 
                    'samplerate': 16000,
                    'image': b64_img
                })
                
                current_subtitle = "Processing voice query..."
                subtitle_time = time.time()

    # 1.5. Vision Integration Trigger (Press 'v')
    if key == ord('v'):
        if time.time() - vision_cooldown > 5.0:
            vision_cooldown = time.time()
            _, buffer = cv2.imencode('.jpg', frame)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            event_queue.put({'type': 'vision_request', 'data': b64_img})

    # Ensure UI shows recording pulse
    if is_recording:
        pulse_text = "Recording" + "." * (int(time.time() * 2) % 4)
        current_subtitle = pulse_text
        subtitle_time = time.time()

    # 2. Draw Live Subtitles with modern Pillow UI
    if current_subtitle and time.time() - subtitle_time < 8.0:
        text = f"Co-Pilot: {current_subtitle}"
        
        # Simple text wrapping to fit screen and handle newlines
        max_chars = 75
        lines = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                lines.extend([paragraph[i:i+max_chars] for i in range(0, len(paragraph), max_chars)])
        if not lines:
            lines = [text]
        
        h, w = frame.shape[:2]
        
        # Convert OpenCV BGR to Pillow RGBA
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert('RGBA')
        
        # Create transparent overlay for UI
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Use modern Windows system font, make text smaller
        try:
            font = ImageFont.truetype("segoeui.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        # Calculate max text width to draw a centered pill shape
        max_text_width = max(draw.textlength(line, font=font) for line in lines)
        box_width = max_text_width + 60
        box_x1 = (w - box_width) / 2
        box_x2 = box_x1 + box_width
        
        bar_height = max(50, 20 + 30 * len(lines))
        box_y1 = h - bar_height - 25
        box_y2 = h - 25
        
        # Draw sleek rounded translucent pill background
        draw.rounded_rectangle(
            [(box_x1, box_y1), (box_x2, box_y2)], 
            radius=16, 
            fill=(10, 10, 10, 150),  # Dark with high transparency
            outline=(150, 150, 150, 200), 
            width=1
        )
        
        # Draw anti-aliased centered text
        for i, line in enumerate(lines):
            line_w = draw.textlength(line, font=font)
            x_pos = (w - line_w) / 2
            y_pos = box_y1 + 12 + (i * 30)
            draw.text((x_pos, y_pos), line, font=font, fill=(250, 250, 250, 255))
            
        # Composite back to OpenCV BGR and copy in-place
        pil_img = Image.alpha_composite(pil_img, overlay)
        frame_new = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
        np.copyto(frame, frame_new)

    # 3. Check for collision alert trigger
    if stats['collision_alert'] and not last_collision_state:
        total_collision_alerts += 1
        critical_obj = "an obstacle"
        for d in detections:
            if d[3]: # is_critical
                critical_obj = d[0]
                break
        nearest_m = stats['nearest_m']
        if nearest_m is not None:
            prompt = f"Collision alert triggered! {critical_obj} detected at {nearest_m:.1f} meters."
            event_queue.put({'type': 'warning', 'data': prompt})

    # 4. Check for lane departure
    if "DEPARTING" in stats['departure_status'] and stats['departure_status'] != last_departure_state:
        total_lane_departures += 1
        prompt = f"Lane departure warning: {stats['departure_status']}."
        event_queue.put({'type': 'warning', 'data': prompt})

    # 5. Check for Green Light Chime
    if stats.get('green_light_chime'):
        event_queue.put({'type': 'warning', 'data': "The light is green. You may proceed."})

    last_collision_state = stats['collision_alert']
    last_departure_state = stats['departure_status']


if __name__ == "__main__":
    ai_thread = threading.Thread(target=ai_copilot_worker, daemon=True)
    ai_thread.start()

    parser = get_args_parser()
    args = parser.parse_args()

    try:
        run_engine(args, callback=engine_callback)
    except KeyboardInterrupt:
        pass
    finally:
        event_queue.put(None)
        ai_thread.join(timeout=1.0)
