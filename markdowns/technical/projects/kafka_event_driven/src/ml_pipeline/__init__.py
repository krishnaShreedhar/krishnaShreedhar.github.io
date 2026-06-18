"""
ml_pipeline — Kafka-backed ML serving components.

Exports:
    FeatureStore, FeaturePipeline    : real-time feature computation
    MockModel, InferencePipeline     : online model inference
    ABRouter, ModelRegistry          : A/B traffic routing
    DriftDetector, MonitoringPipeline: prediction monitoring and PSI drift detection
"""

from ml_pipeline.feature_pipeline import FeatureStore, FeaturePipeline
from ml_pipeline.model_inference import MockModel, InferencePipeline
from ml_pipeline.ab_routing import ABRouter, ModelRegistry
from ml_pipeline.monitoring import DriftDetector, MonitoringPipeline

__all__ = [
    "FeatureStore",
    "FeaturePipeline",
    "MockModel",
    "InferencePipeline",
    "ABRouter",
    "ModelRegistry",
    "DriftDetector",
    "MonitoringPipeline",
]
