from flask import Blueprint, request, jsonify
from ..models.manual_wv import train_model
import os

FILE_PATH = os.path.dirname(__file__)
train_bp = Blueprint("train", __name__)

@train_bp.route("", methods=["POST"])
def train_advanced_model():
    os.makedirs(f"{FILE_PATH}/../cache/manual_wv/samples", exist_ok=True)
    file_found = False
    for file in request.files:
        image = request.files[file]
        if image.filename == "":
            continue
        image.save(f"{FILE_PATH}/../cache/manual_wv/samples/{image.filename}")
        file_found = True
    if not file_found:
        return jsonify({"error": "No files found"}), 400
    train_model()
    return jsonify({"status": f"Model Training Complete"}), 200

@train_bp.route("/status", methods=["GET"])
def get_training_status():
    if os.path.exists(f"{FILE_PATH}/../cache/manual_wv/writer.weights.h5"):
        return '', 200
    else:
        return '', 404
