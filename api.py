from flask import Flask, request, jsonify, send_from_directory
from app.pipeline import PosePipeline
import os

# 👇 IMPORTANT: point Flask to your built frontend
app = Flask(__name__, static_folder="dashboard/dist", static_url_path="")

pipeline = PosePipeline()


# =========================
# FRONTEND ROUTES (UI)
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
        # React fallback (important for routing)
        return send_from_directory(app.static_folder, "index.html")


# =========================
# API ROUTES
# =========================

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


# =========================
# RUN (local only)
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)