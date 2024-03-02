from kafka import KafkaProducer
from config import config
import json

from utils.Singleton import Singleton


class Producer(Singleton):
    def __init__(self) -> None:
        self.__producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_URL,
            value_serializer=self.serializer
        )
        
    def serializer(self, message):
        return json.dumps(message).encode('utf-8')

    def get(self) -> KafkaProducer:
        return self.__producer


