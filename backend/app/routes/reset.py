from flask import Blueprint, request, jsonify
from ..models.manual_wv import train_model
import shutil
import os

FILE_PATH = os.path.dirname(__file__)
reset_bp = Blueprint("reset", __name__)

@reset_bp.route("", methods=["GET"])
def train_advanced_model():
    shutil.rmtree(f"{FILE_PATH}/../cache/manual_wv", ignore_errors=True)
    return '', 204
