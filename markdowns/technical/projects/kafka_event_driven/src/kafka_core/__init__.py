"""
kafka_core — thread-safe in-memory Kafka simulation layer.

Exports:
    MockKafkaBroker   : singleton-style in-memory broker
    MockKafkaProducer : produces messages to the mock broker
    MockKafkaConsumer : consumes messages from the mock broker
    TopicManager      : creates and inspects topics on the mock broker
"""

from kafka_core.mock_kafka import MockKafkaBroker
from kafka_core.producer import MockKafkaProducer
from kafka_core.consumer import MockKafkaConsumer
from kafka_core.topic_manager import TopicManager

__all__ = [
    "MockKafkaBroker",
    "MockKafkaProducer",
    "MockKafkaConsumer",
    "TopicManager",
]
