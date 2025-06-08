from flask import Blueprint, request, jsonify
from ..models.manual_wv import perform_manual_writer_verification
import os

FILE_PATH = os.path.dirname(__file__)
os.makedirs(f"{FILE_PATH}/../temp/manual_wv/samples", exist_ok=True)
predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/advanced-writer-verification", methods=["POST"])
def manual_writer_verification():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    image = request.files["file"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400
    image.save(f"{FILE_PATH}/../temp/manual_wv/samples/test.png")
    
    result = perform_manual_writer_verification()

    return jsonify({"result": f"{result}"}), 200
