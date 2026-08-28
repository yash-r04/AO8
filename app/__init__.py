# app/__init__.py
from flask import Flask
from flask_session import Session
from app.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Server-side sessions stored in Redis
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_REDIS"] = __import__("redis").from_url(Config.REDIS_URL)
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_USE_SIGNER"] = True
    Session(app)

    from app.routes.auth import auth_bp
    from app.routes.models import models_bp
    from app.routes.evaluate import evaluate_bp
    from app.routes.results import results_bp

    app.register_blueprint(auth_bp,     url_prefix="/auth")
    app.register_blueprint(models_bp,   url_prefix="/models")
    app.register_blueprint(evaluate_bp, url_prefix="/evaluate")
    app.register_blueprint(results_bp,  url_prefix="/results")

    return app