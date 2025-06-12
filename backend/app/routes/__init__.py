from .predict import predict_bp
from .train import train_bp
from .reset import reset_bp
from .result import result_bp

def register_routes(app):
    app.register_blueprint(predict_bp, url_prefix="/api/v1/predict")
    app.register_blueprint(train_bp, url_prefix="/api/v1/train")
    app.register_blueprint(reset_bp, url_prefix="/api/v1/reset")
    app.register_blueprint(result_bp, url_prefix="/api/v1/results")
