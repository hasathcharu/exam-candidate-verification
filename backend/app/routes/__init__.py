from .predict import predict_bp

def register_routes(app):
    app.register_blueprint(predict_bp, url_prefix="/api/v1/predict")
