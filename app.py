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

# Default Video
DEFAULT_VIDEO = "mall_counting.mp4"
cap = cv2.VideoCapture(DEFAULT_VIDEO)

# ==============================
# Full Screen Real Area (meters)
# Change based on camera footage
# Example Classroom: 6m x 5m
# ==============================
frame_width_m = 6
frame_height_m = 5
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
# Home Route
# ==============================
@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Upload Route
# ==============================
@app.route("/upload", methods=["POST"])
def upload_video():
    global cap

    file = request.files["video"]
    if file.filename == "":
        return "No file selected!"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    cap.release()
    cap = cv2.VideoCapture(filepath)

    return "Video Uploaded Successfully! Go Back."


# ==============================
# Video Stream Function
# ==============================
def generate_frames():
    global people_count, density_m2, crowd_level

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Detection
        results = model.predict(
            frame,
            classes=[0],
            imgsz=640,
            conf=0.3,
            verbose=False
        )

        detections = sv.Detections.from_ultralytics(results[0])

        # ==============================
        # Full Screen People Count
        # ==============================
        people_count = len(detections.xyxy)

        # ==============================
        # Density in People per m²
        # ==============================
        density_m2 = people_count / frame_area_m2

        # ==============================
        # Crowd Level Thresholds
        # ==============================
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
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# ==============================
# Video Route
# ==============================
@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ==============================
# Stats Route
# ==============================
@app.route("/stats")
def stats():
    return jsonify({
        "people": people_count,
        "density": round(density_m2, 2),
        "level": crowd_level
    })


# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
