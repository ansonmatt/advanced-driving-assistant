import os
import json
import threading
import time
import cv2
from flask import Flask, render_template, Response, request, jsonify

from adas_engine import get_args_parser, run_engine, toggle_lanes, lanes_are_enabled
from adas_copilot import engine_callback as copilot_callback, ai_copilot_worker, event_queue, trigger_voice_query

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state — engine writes raw frames; encoder thread encodes to JPEG
# ---------------------------------------------------------------------------
_raw_frame      = None          # latest raw BGR frame (numpy array)
_jpeg_frame     = None          # latest JPEG bytes ready to stream
_frame_lock     = threading.Lock()
_frame_ready    = threading.Condition(_frame_lock)

engine_thread       = None
encoder_thread      = None
engine_stop_event   = threading.Event()
encoder_stop_event  = threading.Event()


# ---------------------------------------------------------------------------
# Encoder thread — separate from engine so encoding never stalls it
# ---------------------------------------------------------------------------
def encoder_worker():
    """Picks up raw frames and encodes them to JPEG as fast as possible."""
    global _jpeg_frame
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]

    while not encoder_stop_event.is_set():
        with _frame_ready:
            _frame_ready.wait(timeout=0.1)
            frame = _raw_frame

        if frame is None:
            continue

        ret, buf = cv2.imencode('.jpg', frame, encode_params)
        if ret:
            with _frame_lock:
                _jpeg_frame = buf.tobytes()


# ---------------------------------------------------------------------------
# Engine callback — stores raw frame and signals the encoder
# ---------------------------------------------------------------------------
def engine_callback(frame_idx, stats, detections, frame, key):
    global _raw_frame

    # 1. Run the Copilot AI (draws subtitles, triggers AI voice warnings)
    copilot_callback(frame_idx, stats, detections, frame, key)

    # 2. Hand the raw frame to the encoder thread — no blocking encode here
    with _frame_ready:
        _raw_frame = frame.copy()
        _frame_ready.notify_all()


# ---------------------------------------------------------------------------
# MJPEG streaming generator — yields frames as fast as they are encoded
# ---------------------------------------------------------------------------
def gen_frames():
    last_sent = None
    while True:
        with _frame_lock:
            jpeg = _jpeg_frame

        if jpeg is not None and jpeg is not last_sent:
            last_sent = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# Engine thread bootstrap
# ---------------------------------------------------------------------------
def start_engine_thread():
    parser = get_args_parser()
    args = parser.parse_args(["--headless"])
    print(f"Starting Engine Thread with source: {args.source}")
    run_engine(args, callback=engine_callback, stop_event=engine_stop_event)


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    return render_template('index.html', config=config)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    if 'source' in data:
        config['source'] = data['source']
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return jsonify({"status": "success", "message": "Source updated!"})


@app.route('/api/start', methods=['POST'])
def start_engine():
    global engine_thread, encoder_thread, engine_stop_event, encoder_stop_event
    global _raw_frame, _jpeg_frame

    # Stop any existing engine
    if engine_thread is not None and engine_thread.is_alive():
        engine_stop_event.set()
        engine_thread.join(timeout=3.0)

    # Stop any existing encoder
    if encoder_thread is not None and encoder_thread.is_alive():
        encoder_stop_event.set()
        encoder_thread.join(timeout=2.0)

    # Reset shared frame buffers
    with _frame_lock:
        _raw_frame  = None
        _jpeg_frame = None

    # Start fresh encoder thread
    encoder_stop_event.clear()
    encoder_thread = threading.Thread(target=encoder_worker, daemon=True, name="jpeg-encoder")
    encoder_thread.start()

    # Start fresh engine thread
    engine_stop_event.clear()
    engine_thread = threading.Thread(target=start_engine_thread, daemon=True, name="adas-engine")
    engine_thread.start()

    return jsonify({"status": "success", "message": "ADAS Engine Started!"})


@app.route('/api/stop', methods=['POST'])
def stop_engine():
    engine_stop_event.set()
    encoder_stop_event.set()
    return jsonify({"status": "success", "message": "ADAS Engine Stopped!"})


@app.route('/api/toggle_lanes', methods=['POST'])
def api_toggle_lanes():
    """Flip lane detection on/off at runtime."""
    new_state = toggle_lanes()
    state_str = "ON" if new_state else "OFF"
    print(f"[Server] Lane detection toggled: {state_str}")
    return jsonify({"status": "success", "lanes_enabled": new_state})


@app.route('/api/lanes_state', methods=['GET'])
def api_lanes_state():
    return jsonify({"lanes_enabled": lanes_are_enabled()})


@app.route('/api/voice', methods=['POST'])
def api_voice():
    """Receive browser mic audio (WebM/Opus) and forward to the Copilot AI."""
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"status": "error", "message": "No audio data received"}), 400
    try:
        trigger_voice_query(audio_bytes)
        return jsonify({"status": "success", "message": "Voice query received"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Start the Copilot AI background thread
    ai_thread = threading.Thread(target=ai_copilot_worker, daemon=True)
    ai_thread.start()

    print("\n" + "="*50)
    print("Mobile Web App is running!")
    print("Connect your phone to the same Wi-Fi network and open:")
    print("http://<YOUR_COMPUTER_IP>:5000")
    print("PRODUCTION SERVER ACTIVE: Waitress WSGI")
    print("="*50 + "\n")

    # Automatically open the browser
    import webbrowser
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:5000')
    threading.Thread(target=open_browser, daemon=True).start()

    # Run using Waitress for production-scale deployment
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000, threads=8)
