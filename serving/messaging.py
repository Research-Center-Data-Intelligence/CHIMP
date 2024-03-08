# This module should probably be moved into a library that can be reused for all services.
import pika


class MessagingService:
    def __init__(self):
        self._connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self._channel = self._connection.channel()
        self._channel.exchange_declare(exchange='chimp', exchange_type='topic')

    def send(self, message: str, topic: str):
        routing_key = "serving." + topic
        self._channel.basic_publish(exchange='chimp', routing_key=routing_key, body=message)


class MessagingManager:

    def __init__(self):
        self._service = MessagingService()

    def init_app(self, app):
        app.messaging_manager = self

    def send(self, message: str, topic: str):
        self._service.send(message, topic)


messaging_manager = MessagingManager()
