# app/workers/celery_app.py
import ssl
from celery import Celery
from app.config import Config

celery = Celery("ao8")

celery.conf.update(
    broker_url=Config.REDIS_URL,
    result_backend=Config.REDIS_URL,
    include=["app.workers.tasks"],
    broker_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE    # actual ssl constant, not a string
    },
    redis_backend_use_ssl={
        "ssl_cert_reqs": ssl.CERT_NONE
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_max_tasks_per_child=10,
)