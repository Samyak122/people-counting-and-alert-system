import os
import cv2
import supervision as sv
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# ==============================
# Upload Folder Setup
# ==============================
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# Load YOLO Model
# ==============================
model = YOLO("yolov8n.pt")

# ==============================
# Default Video Source
# ==============================
DEFAULT_VIDEO = "mall_counting.mp4"
cap = cv2.VideoCapture(DEFAULT_VIDEO)

# ==============================
# Full Screen Area (meters)
# ==============================
frame_width_m = 60
frame_height_m = 50
frame_area_m2 = frame_width_m * frame_height_m

# ==============================
# Global Stats
# ==============================
people_count = 0
density_m2 = 0
crowd_level = "LOW"

# Annotator
box_annotator = sv.BoxAnnotator(thickness=2)

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
    global cap

    file = request.files["video"]
    if file.filename == "":
        return "❌ No file selected!"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    cap.release()
    cap = cv2.VideoCapture(filepath)

    return "✅ Video Uploaded Successfully! Go Back."


# ==============================
# LIVE STREAM LINK (IP CAMERA)
# ==============================
@app.route("/live", methods=["POST"])
def live_stream():
    global cap

    stream_url = request.form.get("stream")

    if stream_url == "":
        return "❌ No Live Link Provided!"

    cap.release()
    cap = cv2.VideoCapture(stream_url)

    return "✅ Live Stream Started Successfully! Go Back."


# ==============================
# WEBCAM START
# ==============================
@app.route("/webcam")
def webcam():
    global cap
    cap.release()
    cap = cv2.VideoCapture(0)
    return "✅ Webcam Started Successfully! Go Back."


# ==============================
# VIDEO STREAM FUNCTION
# ==============================
def generate_frames():
    global people_count, density_m2, crowd_level

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # YOLO Detection (Only Person Class)
        results = model.predict(
            frame,
            classes=[0],
            imgsz=640,
            conf=0.3,
            verbose=False
        )

        detections = sv.Detections.from_ultralytics(results[0])

        # People Count
        people_count = len(detections.xyxy)

        # Density Calculation
        density_m2 = people_count / frame_area_m2

        # Crowd Level
        if density_m2 < 0.3:
            crowd_level = "LOW"
        elif density_m2 < 0.8:
            crowd_level = "MEDIUM"
        else:
            crowd_level = "HIGH"

        # Draw Bounding Boxes
        frame = box_annotator.annotate(scene=frame, detections=detections)

        # Convert Frame to JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")


# ==============================
# VIDEO ROUTE
# ==============================
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ==============================
# STATS ROUTE
# ==============================
@app.route("/stats")
def stats():
    return jsonify({
        "people": people_count,
        "density": round(density_m2, 2),
        "level": crowd_level
    })


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)