from kafka import KafkaProducer
from config import config
import json

from utils.Singleton import Singleton
from cv2.typing import MatLike
import cv2
import base64


class Producer(Singleton):
    def __init__(this) -> None:
        this.__producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_URL,
            # value_serializer=this.serializer
        )

    def serializer(this, message):
        return json.dumps(message).encode("utf-8")

    def send_json(
        this,
        topic: str,
        dict: dict,
    ):
        this.__producer.send(topic=topic, value=json.dumps(dict).encode("utf-8"))
        return

    def send_image_by_frame(
        this,
        topic: str,
        frame: MatLike,
    ):
        img_str = cv2.imencode(".jpg", frame)[1].tobytes()
        jpg_as_text = base64.b64encode(img_str)

        this.__producer.send(topic=topic, value=jpg_as_text)
        return

    def get(this) -> KafkaProducer:
        return this.__producer
