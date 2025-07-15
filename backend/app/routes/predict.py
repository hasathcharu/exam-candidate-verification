from flask import Blueprint, request, jsonify
from ..models.manual_wv import perform_manual_writer_verification
from ..models.automatic_wv import perform_automatic_writer_verification
from ..models.automatic_wv import perform_pairwise_automatic_writer_verification
import os
from werkzeug.exceptions import InternalServerError

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
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/normal/samples")
    os.makedirs(output_dir, exist_ok=True)
    sample1_path = os.path.join(output_dir, "known.png")
    sample2_path = os.path.join(output_dir,"unknown.png")
    sample1.save(sample1_path)
    sample2.save(sample2_path)

    try:
        result = perform_automatic_writer_verification(sample1_path, sample2_path)
    except Exception as e:
        print("Error in writer verification:", str(e))
        raise InternalServerError("An error occurred during verification. Please try again.")

    return jsonify({
        "distance": result["distance"],
        "threshold": result["threshold"],
        "same_writer": result["same_writer"] 
    }), 200


@predict_bp.route("/pairwise-writer-verification", methods=["POST"])
def pairwise_writer_verification():
    required_files = ["file1_normal","file1_fast","file2_normal","file2_fast"]
    for fname in required_files:
        if fname not in request.files:
            return jsonify({"error":f"Missing file: {fname}"}),400
        
    output_dir = os.path.join(FILE_PATH,"../cache/automatic_wv/pairwise/samples")
    os.makedirs(output_dir, exist_ok=True)

    file1_normal = request.files["file1_normal"]
    file1_fast = request.files["file1_fast"]
    file2_normal = request.files["file2_normal"]
    file2_fast = request.files["file2_fast"]
    
    file1_normal.save(f"{FILE_PATH}/../cache/automatic_wv/pairwise/samples/file1_normal.png")
    file1_fast.save(f"{FILE_PATH}/../cache/automatic_wv/pairwise/samples/file1_fast.png")
    file2_normal.save(f"{FILE_PATH}/../cache/automatic_wv/pairwise/samples/file2_normal.png")
    file2_fast.save(f"{FILE_PATH}/../cache/automatic_wv/pairwise/samples/file2_fast.png")

    try:
        result = perform_pairwise_automatic_writer_verification(
            file1_normal_path=os.path.join(output_dir, "file1_normal.png"),
            file1_fast_path=os.path.join(output_dir, "file1_fast.png"),
            file2_normal_path=os.path.join(output_dir, "file2_normal.png"),
            file2_fast_path=os.path.join(output_dir, "file2_fast.png")
        )

    except Exception as e:
        print("Error in writer verification:", str(e))
        raise InternalServerError("An error occurred during verification. Please try again.")

    return jsonify({
        "probability": result["probability"],
        "same_writer": result["same_writer"],
    }), 200
