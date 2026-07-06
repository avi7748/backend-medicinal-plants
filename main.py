import os

import cv2, base64, threading, time
import numpy as np
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime, timedelta
from urllib.parse import unquote
from database import DatabaseManager
import os;

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'port': int(os.getenv('MYSQL_PORT', 4000)),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE'),
    'ssl_verify_cert': True,
    'ssl_verify_identity': True,
    'ssl_ca': os.getenv('MYSQL_SSL_CA', 'isrgrootx1.pem')
}
db = DatabaseManager(DB_CONFIG)

model = YOLO("best.pt")

camera_ip = None
camera_url = None
camera_connected = False
camera_running = False
camera_lock = threading.Lock()

latest_frame = None
frame_lock = threading.Lock()

location_lock = threading.Lock()

last_db_save = datetime.now()

latest_location = {
    "lat": 18.0,
    "lng": 73.0
}

species_id_map = db.fetch_species_map()

WRONG_CLASS_ID = 3
CORRECT_NAME = "Centella asiatica"


@socketio.on('save_plant_location')
def handle_location(data):
    global latest_location

    lat = data.get("lat")
    lng = data.get("lng")

    if lat is None or lng is None:
        print("❌ INVALID GPS RECEIVED:", data)
        return

    with location_lock:
        latest_location["lat"] = float(lat)
        latest_location["lng"] = float(lng)



@app.route('/api/species')
def fetch_library():
    return jsonify(db.fetch_all_species())


@app.route('/api/plantinfo/<path:name>')
def get_plantinfo(name):
    try:
        name = unquote(name).strip()

        conn = db.get_connection()
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM species_info WHERE name = %s OR scientific_name = %s"
        cursor.execute(query, (name, name))

        row = cursor.fetchone()
        conn.close()

        return jsonify(row if row else {})

    except Exception as e:
        print("API Error:", e)
        return jsonify({})


@app.route('/api/history')
def get_history():
    try:
        page = request.args.get("page", default=1, type=int)
        limit = request.args.get("limit", default=50, type=int)

        data = db.get_detections(page=page, limit=limit)

        return jsonify(data)

    except Exception as e:
        print("History API Error:", e)

        return jsonify({
            "records": [],
            "page": 1,
            "pages": 1,
            "total": 0
        })


@app.route("/api/connect-camera", methods=["POST"])
def connect_camera():
    global camera_ip
    global camera_url
    global camera_connected
    global camera_running

    data = request.get_json()
    ip = data.get("ip", "").strip()

    if ip == "":
        return jsonify({
            "success": False,
            "message": "IP Address Required"
        })

    url = f"http://{ip}:8080/video"
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        cap.release()
        return jsonify({
            "success": False,
            "message": "Unable to connect"
        })

    cap.release()

    with camera_lock:
        camera_ip = ip
        camera_url = url
        camera_connected = True
        camera_running = True

    return jsonify({
        "success": True,
        "message": "Camera Connected"
    })


@app.route("/api/disconnect-camera", methods=["POST"])
def disconnect_camera():
    global camera_url
    global camera_ip
    global latest_frame
    global camera_running

    with camera_lock:
        camera_url = None
        camera_ip = None
        camera_connected = False
        camera_running = False

    with frame_lock:
        latest_frame = None

    print("Camera Disconnected")

    return jsonify({
        "success": True
    })


@app.route("/api/process-frame", methods=["POST"])
def process_frame():
    if "frame" not in request.files:
        return jsonify({
            "predictions": []
        })
    file = request.files["frame"]
    image_bytes = file.read()
    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )
    frame = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )
    if frame is None:
        return jsonify({
            "predictions": []
        })
    detections = run_yolo_detection(
        frame,
        save_to_db=False
    )
    return jsonify({
        "predictions": detections
    })


def camera_thread():
    global latest_frame

    while True:
        if camera_url is None:
            time.sleep(1)
            continue

        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            if not camera_running:
                print("Camera Disconnected")
                break

            ret, frame = cap.read()

            if not ret:
                print("Camera Lost")
                break

            with frame_lock:
                latest_frame = frame

        cap.release()

        with frame_lock:
            latest_frame = None

        time.sleep(2)


def run_yolo_detection(img, save_to_db=False):
    global last_db_save
    raw_h, raw_w = img.shape[:2]
    scale_x = 960 / raw_w
    scale_y = 720 / raw_h
    results = model.predict(
        img,
        imgsz=320,
        conf=0.6,
        verbose=False
    )
    detections = []
    if len(results[0].boxes) > 0:
        top_box = results[0].boxes[0]
        cls_id = int(top_box.cls[0])
        if cls_id == WRONG_CLASS_ID:
            top_name = CORRECT_NAME
        else:
            top_name = model.names[cls_id]
        top_conf = float(top_box.conf[0])
        if save_to_db:
            if datetime.now() - last_db_save > timedelta(seconds=10):
                plant_name = top_name.strip().lower()
                s_id = None
                for key in species_id_map:
                    if key and key.strip().lower() == plant_name:
                        s_id = species_id_map[key]
                        break
                if s_id:
                    with location_lock:
                        lat = latest_location["lat"]
                        lng = latest_location["lng"]
                    success = db.save_detection(
                        s_id,
                        int(top_conf * 100),
                        lat,
                        lng
                    )
                    if success:
                        last_db_save = datetime.now()
            socketio.emit("request_location", {
                "detected": True,
                "name": top_name,
                "conf": int(top_conf * 100),
                "time": datetime.now().strftime("%H:%M:%S")
            })
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            label = (
                CORRECT_NAME
                if int(b.cls[0]) == WRONG_CLASS_ID
                else model.names[int(b.cls[0])]
            )
            detections.append({
                "bbox": [
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y
                ],
                "label": label,
                "conf": float(b.conf[0])
            })
    else:
        if save_to_db:
            socketio.emit("request_location", {
                "detected": False,
                "name": "No Medicinal Plant Detected",
                "conf": 0,
                "time": datetime.now().strftime("%H:%M:%S")
            })
    return detections


def inference_loop():
    global last_db_save

    while True:
        if latest_frame is None:
            socketio.sleep(0.1)
            continue

        with frame_lock:
            img = latest_frame.copy()

        detections = run_yolo_detection(
            img,
            save_to_db=True)

        stream_img = cv2.resize(img, (960, 720))
        _, buffer = cv2.imencode('.jpg', stream_img, [cv2.IMWRITE_JPEG_QUALITY, 50])

        socketio.emit('detection_data', {
            "image": base64.b64encode(buffer).decode('utf-8'),
            "predictions": detections
        })

        socketio.sleep(0.01)


if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    socketio.start_background_task(inference_loop)
    socketio.run(app, host="0.0.0.0", port=5000)