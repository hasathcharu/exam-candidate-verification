from flask import Blueprint, jsonify, send_from_directory
import os

RESULT_FOLDER = os.path.join(os.path.dirname(__file__), "../rcache/manual_wv/result")
result_bp = Blueprint("results", __name__)

@result_bp.route("<task_id>/<filename>")
def serve_file(task_id, filename):
    result_dir = os.path.join(RESULT_FOLDER, task_id)
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    if not task_id:
        return jsonify({"error": "Task ID is required"}), 400
    if filename not in os.listdir(result_dir):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(result_dir, filename)

