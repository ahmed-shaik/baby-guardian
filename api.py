from flask import Flask, request, jsonify, send_from_directory
from app.pipeline import PosePipeline
import os

app = Flask(__name__, static_folder="dashboard/dist", static_url_path="")

import threading

pipeline = None
pipeline_lock = threading.Lock()

def get_pipeline():
    global pipeline
    if pipeline is None:
        with pipeline_lock:
            if pipeline is None:
                print("🔥 Loading PosePipeline...")
                pipeline = PosePipeline()
    return pipeline

# 🔥 GLOBAL STATE (for frontend sync)
current_source = None
is_running = False


# =========================
# FRONTEND ROUTES
# =========================

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    file_path = os.path.join(app.static_folder, path)

    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


# =========================
# API ROUTES (IMPORTANT)
# =========================

# 🔥 REQUIRED: frontend calls this
@app.route("/api/source", methods=["POST"])
def set_source():
    global current_source, is_running

    data = request.get_json()

    if not data or "source" not in data:
        return jsonify({"ok": False, "error": "No source provided"}), 400

    current_source = data["source"]

    try:
        # 👉 You can later replace this with real streaming
        is_running = True

        return jsonify({
            "ok": True,
            "source": current_source
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# 🔥 REQUIRED: frontend calls this
@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "running": is_running,
        "source": current_source
    })


# 🔥 Existing API
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    import uuid
    filepath = f"temp_{uuid.uuid4().hex}.jpg"
    file.save(filepath)

    try:
        result = get_pipeline().analyze_image(filepath)

        response = {
            "frame_index": result.frame_index,
            "timestamp_ms": result.timestamp_ms,
            "persons": result.persons,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "track_id": d.track_id,
                }
                for d in result.detections
            ]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


# =========================
# RUN (LOCAL ONLY)
# =========================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)