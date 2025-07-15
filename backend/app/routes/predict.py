from flask import Blueprint, request, jsonify
from ..models.manual_wv import perform_manual_writer_verification
from ..models.automatic_wv import perform_automatic_writer_verification
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

@predict_bp.route("/automatic-writer-verification", methods=["POST"])
def automatic_writer_verification():
    if "file1" not in request.files or "file2" not in request.files:
        return jsonify({"error":"Two samples required: 'file1' and 'file2'"}),400

    sample1 = request.files["file1"]
    sample2 = request.files["file2"]

    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/samples")
    os.makedirs(output_dir, exist_ok=True)

    sample1_path = os.path.join(output_dir, "known.png")
    sample2_path = os.path.join(output_dir,"unknown.png")

    sample1.save(sample1_path)
    sample2.save(sample2_path)

    result = perform_automatic_writer_verification(sample1_path, sample2_path)

    return jsonify({
        "message": "Files received successfully",
        "known_file": sample1.filename,
        "questioned_file": sample2.filename
    }), 200

@predict_bp.route("/signature-verification", methods=["POST"])
def signature_verification():
    if "sign_file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files["sign_file"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    os.makedirs(f"{FILE_PATH}/../cache/ofsfd/samples", exist_ok=True)
    image.save(f"{FILE_PATH}/../cache/ofsfd/samples/test.png")
    
    label, confidence = predict_signature_type(f"{FILE_PATH}/../cache/ofsfd/samples/test.png")
    # label, confidence = predict_signature_type("backend\tests\ofsfd\1_0.png")
    
    return jsonify({"label": label, "confidence": confidence}), 200
