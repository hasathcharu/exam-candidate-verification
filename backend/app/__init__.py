from flask import Flask
from flask_cors import CORS
import os

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp", "manual_wv")
os.makedirs(TEMP_DIR, exist_ok=True)

def create_app():
    app = Flask(__name__)
    CORS(app)

    from .routes import register_routes
    register_routes(app)

    return app
