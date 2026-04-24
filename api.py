from flask import Flask, request, jsonify
from app.pipeline import PosePipeline
import os

app = Flask(__name__)
pipeline = PosePipeline()


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Baby Guardian API is live"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    filepath = "temp.jpg"
    file.save(filepath)

    try:
        result = pipeline.analyze_image(filepath)

        # Convert safely to JSON
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
