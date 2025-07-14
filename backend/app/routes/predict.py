from flask import Blueprint, request, jsonify
from ..models.manual_wv import perform_manual_writer_verification
from ..models.ofsfd import predict_signature_type
import os

FILE_PATH = os.path.dirname(__file__)
predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/advanced-writer-verification", methods=["POST"])
def manual_writer_verification():
    if not os.path.exists(f"{FILE_PATH}/../cache/manual_wv/writer.weights.h5"):
        return jsonify({"error": "Model not trained"}), 400
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files["file"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    os.makedirs(f"{FILE_PATH}/../cache/manual_wv/samples", exist_ok=True)
    image.save(f"{FILE_PATH}/../cache/manual_wv/samples/test.png")
    
    result = perform_manual_writer_verification()
    return jsonify({"result": f"{result}"}), 200

@predict_bp.route("/signature-verification", methods=["POST"])
def signature_verification():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files["file"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    os.makedirs(f"{FILE_PATH}/../cache/ofsfd/samples", exist_ok=True)
    image.save(f"{FILE_PATH}/../cache/ofsfd/samples/test.png")
    
    label, confidence = predict_signature_type(f"{FILE_PATH}/../cache/ofsfd/samples/test.png")
    # label, confidence = predict_signature_type("backend\tests\ofsfd\1_0.png")
    
    return jsonify({"label": label, "confidence": confidence}), 200
