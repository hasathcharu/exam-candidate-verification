from flask import Blueprint, request, jsonify, send_from_directory
import os

RESULT_FOLDER = os.path.join(os.path.dirname(__file__), "../cache/manual_wv/result")
result_bp = Blueprint("results", __name__)

@result_bp.route("<filename>")
def serve_image(filename):
    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    if filename not in os.listdir(RESULT_FOLDER):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(RESULT_FOLDER, filename)

