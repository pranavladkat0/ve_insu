from dataclasses import dataclass
from typing import Optional


# -----------------------------
# Data Ingestion Artifact
# -----------------------------
@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    feature_store_file_path: str


# -----------------------------
# Data Validation Artifact
# -----------------------------
@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    validation_report_file_path: str


# -----------------------------
# Data Transformation Artifact
# -----------------------------
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


# -----------------------------
# Model Trainer Artifact
# -----------------------------
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metric_artifact: Optional[object] = None


# -----------------------------
# Model Evaluation Artifact
# -----------------------------
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    s3_model_path: Optional[str]   # ✅ allow None (since skipping AWS)
    trained_model_path: str
    changed_accuracy_score: float   # ✅ correct field name


# -----------------------------
# Model Pusher Artifact
# -----------------------------
@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str


# -----------------------------
# Metrics Artifact
# -----------------------------
@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float