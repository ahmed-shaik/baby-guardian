from flask import Flask, request, jsonify, send_from_directory
from app.pipeline import PosePipeline
import os

app = Flask(__name__, static_folder="dashboard/dist", static_url_path="")

pipeline = PosePipeline()

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
# API ROUTES
# =========================

# 🔥 FIX 1: Start Stream (your UI needs this)
@app.route("/api/start-stream", methods=["POST", "GET"])
def start_stream():
    try:
        data = request.get_json(silent=True) or {}
        stream_url = data.get("url") or request.args.get("url")

        if not stream_url:
            return jsonify({"error": "No stream URL provided"}), 400

        # For now: just acknowledge (you can integrate real streaming later)
        return jsonify({
            "status": "started",
            "stream_url": stream_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🔥 FIX 2: Analyze Image
@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = "temp.jpg"
    file.save(filepath)

    try:
        result = pipeline.analyze_image(filepath)

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


# 🔥 Optional: Health check
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =========================
# RUN (LOCAL ONLY)
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)