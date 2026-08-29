# app/__init__.py
from flask import Flask
from flask_session import Session
from app.config import Config
import json
from datetime import datetime, date

class CustomJSONProvider(Flask.json_provider_class):
    def dumps(self, obj, **kwargs):
        def default(o):
            if isinstance(o, (datetime, date)):
                return o.isoformat()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        return json.dumps(obj, default=default, **kwargs)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)

def create_app():
    app = Flask(__name__)
    app.json_provider_class = CustomJSONProvider
    app.json = CustomJSONProvider(app)
    app.config.from_object(Config)

    import redis
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_REDIS"] = redis.from_url(
        Config.REDIS_URL,
        decode_responses=False,
        protocol=2,
        ssl_cert_reqs=None
    )
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