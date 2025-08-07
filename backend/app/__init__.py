from flask import Flask
from flask_cors import CORS
import os
import gdown
import zipfile

def create_app():
    if not os.path.exists("app/weights/"):
        os.makedirs("app/weights/")
        print("Downloading model weights...")
        gdown.download("https://drive.google.com/uc?id=1b5TtjUppA8McS5S_VC3watfixafxHd6M", "app/weights.zip", quiet=False)
        with zipfile.ZipFile('app/weights.zip', 'r') as zip_ref:
            zip_ref.extractall('app/')
    app = Flask(__name__)
    CORS(app)

    from .routes import register_routes
    register_routes(app)

    return app
