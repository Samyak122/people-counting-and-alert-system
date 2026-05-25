import os
import cv2
import numpy as np
import supervision as sv
from flask import Flask, Response, jsonify, render_template, request
from twilio.rest import Client
from threading import Thread, Lock
from queue import Queue
from collections import defaultdict
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ✅ Lazy load YOLO to avoid torch import issues
model = None
tracker = None

def load_model():
    """Load YOLOv8m for better crowd detection"""
    global model
    if model is None:
        from ultralytics import YOLO
        # Use YOLOv8m (medium) instead of YOLOv8n for better accuracy on crowds
        model = YOLO("yolov8m.pt")
        # Try to use GPU if available
        try:
            model.to("cuda")
            print("✅ GPU (CUDA) detected - using GPU for faster inference")
        except:
            print("⚠️ GPU not available - using CPU")
    return model

def load_tracker():
    """Load ByteTrack for person tracking"""
    global tracker
    if tracker is None:
        from supervision.tracker import ByteTrack
        tracker = ByteTrack()
    return tracker

app = Flask(__name__)

# ==============================
# Twilio Setup
# ==============================
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_NUMBER")
recipients_str = os.getenv("RECIPIENT_NUMBERS", "")
recipients = [num.strip() for num in recipients_str.split(",") if num.strip()] if recipients_str else []

# Initialize Twilio client only if credentials are provided
client = Client(account_sid, auth_token) if account_sid and auth_token else None

def send_sms_alert(message):
    """Send SMS alert with better error handling"""
    if not client or not recipients:
        print("⚠️ Twilio not configured - SMS alerts disabled")
        return
    
    for number in recipients:
        try:
            client.messages.create(
                body=message,
                from_=twilio_number,
                to=number
            )
            print(f"✅ SMS Sent to: {number}")
        except Exception as e:
            print(f"❌ SMS Error: {e}")

# ==============================
# Upload Folder Setup
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# Default Video Source
# ==============================
DEFAULT_VIDEO = "mall_counting.mp4"
cap = cv2.VideoCapture(DEFAULT_VIDEO)

# ==============================
# Full Screen Area (meters) - Adjustable for perspective
# ==============================
frame_width_m = 10  # Increased for realistic crowd scenarios
frame_height_m = 8
frame_area_m2 = frame_width_m * frame_height_m

# ==============================
# Global Stats with Threading Safety
# ==============================
stats_lock = Lock()
people_count = 0
density_m2 = 0
crowd_level = "LOW"
alert_sent = False
tracked_ids = set()  # Track unique people
fps = 0
frame_time = 0

# ==============================
# Zone-Based Density Analysis
# ==============================
ZONES = {
    "Zone_1": {"x": 0, "y": 0, "w": 0.5, "h": 0.5, "density": 0, "count": 0},
    "Zone_2": {"x": 0.5, "y": 0, "w": 0.5, "h": 0.5, "density": 0, "count": 0},
    "Zone_3": {"x": 0, "y": 0.5, "w": 0.5, "h": 0.5, "density": 0, "count": 0},
    "Zone_4": {"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5, "density": 0, "count": 0},
}

# ==============================
# Performance Settings
# ==============================
FRAME_SKIP = 4  # Process every 4th frame (heavy detection every 4 frames)
JPEG_QUALITY = 40  # Very low quality for faster transmission
DETECTION_SCALE = 0.5  # 50% resolution for detection (much faster)
CONF_THRESHOLD = 0.10  # Very low for crowd detection
IOU_THRESHOLD = 0.2   # Allow overlapping boxes
FRAME_ROTATION = 0  # 0, 90, 180, 270 degrees
IS_STREAM_SOURCE = False  # Flag to detect if using IP camera/stream
DISPLAY_SCALE = 0.75  # Resize display frames to 75% for faster encoding

# Annotator
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

# Global variables
frame_counter = 0
last_detections = None
last_frame = None
heatmap = None
processing_queue = Queue(maxsize=5)

print("✅ Enhanced Crowd Detection System Initialized")
print(f"   - Using YOLOv8m model (better for crowds)")
print(f"   - ByteTrack enabled for person tracking")
print(f"   - Zone-based density analysis: 4 zones")
print(f"   - Heatmap generation enabled")

# ==============================
# HOME PAGE
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# UPLOAD VIDEO FILE
# ==============================
@app.route("/upload", methods=["POST"])
def upload_video():
    global cap, IS_STREAM_SOURCE
    file = request.files["video"]
    if file.filename == "":
        return "❌ No file selected!"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    cap.release()
    cap = cv2.VideoCapture(filepath)
    IS_STREAM_SOURCE = False  # Local video file
    return "✅ Video Uploaded Successfully! Go Back."

# ==============================
# LIVE STREAM LINK (IP CAMERA)
# ==============================
@app.route("/live", methods=["POST"])
def live_stream():
    global cap, IS_STREAM_SOURCE
    stream_url = request.form.get("stream")
    if stream_url == "":
        return "❌ No Live Link Provided!"
    cap.release()
    cap = cv2.VideoCapture(stream_url)
    IS_STREAM_SOURCE = True  # Enable optimization for IP streams
    return "✅ Live Stream Started Successfully! Go Back."

# ==============================
# WEBCAM START
# ==============================
@app.route("/webcam")
def webcam():
    global cap, IS_STREAM_SOURCE
    cap.release()
    IS_STREAM_SOURCE = False  # Webcam is not a stream
    for device_id in range(5):
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            if ret:
                print(f"✅ Webcam found at device {device_id}")
                return f"✅ Webcam Started Successfully (Device {device_id})! Go Back."
            else:
                cap.release()
    cap = cv2.VideoCapture(0)
    return "❌ ERROR: No webcam detected!"

# ==============================
# ZONE-BASED DENSITY CALCULATION
# ==============================
def calculate_zone_density(detections, frame_h, frame_w):
    """Calculate crowd density for each zone"""
    global ZONES
    
    for zone_name, zone in ZONES.items():
        zone_x1 = int(zone["x"] * frame_w)
        zone_y1 = int(zone["y"] * frame_h)
        zone_x2 = int((zone["x"] + zone["w"]) * frame_w)
        zone_y2 = int((zone["y"] + zone["h"]) * frame_h)
        zone_area = (zone_x2 - zone_x1) * (zone_y2 - zone_y1) / 1_000_000  # Convert to m²
        
        # Count people in this zone
        count = 0
        if detections.xyxy.shape[0] > 0:
            for box in detections.xyxy:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2:
                    count += 1
        
        zone["count"] = count
        zone["density"] = count / zone_area if zone_area > 0 else 0

# ==============================
# HEATMAP GENERATION
# ==============================
def generate_heatmap(detections, frame_h, frame_w):
    """Generate crowd density heatmap (optimized)"""
    heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)
    
    if detections.xyxy.shape[0] > 0:
        for box in detections.xyxy:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, x2 = max(0, x1), min(frame_w, x2)
            y1, y2 = max(0, y1), min(frame_h, y2)
            heatmap[y1:y2, x1:x2] += 1
    
    # Smooth heatmap with smaller kernel for better performance
    blur_kernel = 11  # Smaller kernel = faster processing
    heatmap = cv2.GaussianBlur(heatmap, (blur_kernel, blur_kernel), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ==============================
# VIDEO STREAM WITH ENHANCED DETECTION
# ==============================
def generate_frames():
    global people_count, density_m2, crowd_level, alert_sent, frame_counter
    global last_detections, last_frame, heatmap, tracked_ids, fps, frame_time
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_counter += 1
        
        # ✅ APPLY FRAME ROTATION (fix horizontal/vertical issues)
        if FRAME_ROTATION == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif FRAME_ROTATION == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif FRAME_ROTATION == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        frame_h, frame_w = frame.shape[:2]
        
        # ✅ DETECTION: Process with YOLOv8m (skip frames for better performance)
        if frame_counter % FRAME_SKIP == 0:
            detect_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            yolo_model = load_model()
            
            results = yolo_model.predict(
                detect_frame,
                classes=[0],
                imgsz=640,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
            last_detections = sv.Detections.from_ultralytics(results[0])
            
            # Scale detections back
            if last_detections.xyxy.shape[0] > 0:
                last_detections.xyxy = last_detections.xyxy / DETECTION_SCALE
            
            # ✅ TRACKING: Apply ByteTrack to avoid duplicate counting
            try:
                byte_tracker = load_tracker()
                last_detections = byte_tracker.update_with_detections(last_detections)
                if last_detections.tracker_id is not None:
                    tracked_ids.update(last_detections.tracker_id)
            except:
                pass  # Fallback if tracking fails
            
            # Count unique tracked people
            people_count = len(tracked_ids) if len(tracked_ids) > 0 else len(last_detections.xyxy)
            
            # ✅ DENSITY CALCULATION
            density_m2 = people_count / frame_area_m2
            
            # ✅ ZONE-BASED DENSITY
            calculate_zone_density(last_detections, frame_h, frame_w)
            
            # ✅ GENERATE HEATMAP (skip frames for performance)
            if frame_counter % (FRAME_SKIP * 2) == 0:
                heatmap = generate_heatmap(last_detections, frame_h, frame_w)
            
            # ✅ CROWD LEVEL & ALERTS
            if density_m2 < 0.3:
                crowd_level = "LOW"
                alert_sent = False
            elif density_m2 < 0.8:
                crowd_level = "MEDIUM"
                alert_sent = False
            elif density_m2 < 2.0:
                crowd_level = "HIGH"
                if not alert_sent:
                    send_sms_alert(f"⚠️ HIGH CROWD ALERT! People: {people_count}, Density: {density_m2:.2f}")
                    alert_sent = True
            else:  # density_m2 >= 2.0
                crowd_level = "DANGER"
                send_sms_alert(f"🚨 DANGER! EXTREME OVERCROWDING! People: {people_count}, Density: {density_m2:.2f}")
        
        # ✅ DRAW DETECTIONS on frame
        if last_detections is not None:
            frame = box_annotator.annotate(scene=frame, detections=last_detections)
            
            # Draw zone boundaries
            for zone_name, zone in ZONES.items():
                z_x1 = int(zone["x"] * frame_w)
                z_y1 = int(zone["y"] * frame_h)
                z_x2 = int((zone["x"] + zone["w"]) * frame_w)
                z_y2 = int((zone["y"] + zone["h"]) * frame_h)
                
                # Color based on density
                if zone["density"] > 1.5:
                    color = (0, 0, 255)  # Red - danger
                elif zone["density"] > 0.8:
                    color = (0, 165, 255)  # Orange - high
                else:
                    color = (0, 255, 0)  # Green - safe
                
                cv2.rectangle(frame, (z_x1, z_y1), (z_x2, z_y2), color, 2)
                cv2.putText(frame, f"{zone_name}: {zone['count']} p", (z_x1+5, z_y1+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # ✅ ADD STATS OVERLAY
        cv2.putText(frame, f"People: {people_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Density: {density_m2:.2f} p/m²", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Level: {crowd_level}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize for faster JPEG encoding
        display_frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
        
        # Encode frame with low quality for faster transmission
        _, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

# ==============================
# ROUTES
# ==============================
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    with stats_lock:
        zone_stats = {name: {"count": z["count"], "density": round(z["density"], 2)} 
                     for name, z in ZONES.items()}
        return jsonify({
            "people": people_count,
            "density": round(density_m2, 2),
            "level": crowd_level,
            "fps": round(fps, 1),
            "zones": zone_stats,
            "tracked_ids": len(tracked_ids)
        })

@app.route("/heatmap")
def get_heatmap():
    """Endpoint to fetch heatmap"""
    global heatmap
    if heatmap is None:
        return "No heatmap data yet", 404
    _, buffer = cv2.imencode(".jpg", heatmap)
    return Response(buffer.tobytes(), mimetype="image/jpeg")

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """API to adjust detection settings"""
    global CONF_THRESHOLD, IOU_THRESHOLD, DETECTION_SCALE, frame_width_m, frame_height_m, frame_area_m2, FRAME_ROTATION
    
    if request.method == "POST":
        data = request.json
        if "conf" in data:
            CONF_THRESHOLD = float(data["conf"])
        if "iou" in data:
            IOU_THRESHOLD = float(data["iou"])
        if "scale" in data:
            DETECTION_SCALE = float(data["scale"])
        if "frame_width" in data:
            frame_width_m = float(data["frame_width"])
            frame_area_m2 = frame_width_m * frame_height_m
        if "frame_height" in data:
            frame_height_m = float(data["frame_height"])
            frame_area_m2 = frame_width_m * frame_height_m
        if "rotation" in data:
            FRAME_ROTATION = int(data["rotation"]) % 360  # 0, 90, 180, 270
        
        return jsonify({"status": "✅ Settings updated"})
    
    return jsonify({
        "conf": CONF_THRESHOLD,
        "iou": IOU_THRESHOLD,
        "scale": DETECTION_SCALE,
        "frame_width": frame_width_m,
        "frame_height": frame_height_m,
        "rotation": FRAME_ROTATION
    })

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=False, threaded=True)