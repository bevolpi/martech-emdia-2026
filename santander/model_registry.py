# -*- coding: utf-8 -*-
"""
Utilitarios de versionamento de modelos treinados.

Cada versao salva:
- artefato joblib em models/<model_name>/<model_name>_<timestamp>.joblib
- metadata JSON ao lado do artefato
- linha consolidada em model_versions.csv
"""

import csv
import json
import math
import numbers
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_REGISTRY_DIR = Path("models")
METRICS_LOG_PATH = Path("model_versions.csv")


def _safe_name(value):
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in value)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def calcular_metricas_classificacao(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def salvar_modelo_versionado(
    model,
    model_name,
    metrics,
    params=None,
    dataset_path=None,
    feature_columns=None,
    target_column=None,
    legacy_model_path=None,
):
    trained_at = datetime.now().replace(microsecond=0).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = _safe_name(model_name)
    version_id = f"{safe_model_name}_{timestamp}"

    model_dir = MODEL_REGISTRY_DIR / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = model_dir / f"{version_id}.joblib"
    metadata_path = model_dir / f"{version_id}.json"

    joblib.dump(model, artifact_path)

    if legacy_model_path:
        joblib.dump(model, legacy_model_path)

    metadata = {
        "version_id": version_id,
        "model_name": safe_model_name,
        "trained_at": trained_at,
        "artifact_path": str(artifact_path),
        "legacy_model_path": legacy_model_path,
        "dataset_path": dataset_path,
        "target_column": target_column,
        "feature_columns": feature_columns or [],
        "metrics": metrics,
        "params": params or {},
    }

    metadata = _json_safe(metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, allow_nan=False)

    _append_metrics_log(metadata)

    print("\nModelo versionado com sucesso:")
    print(f"  Versao       : {version_id}")
    print(f"  Artefato     : {artifact_path}")
    print(f"  Metadata     : {metadata_path}")
    print(f"  Log metricas : {METRICS_LOG_PATH}")

    return metadata


def _append_metrics_log(metadata):
    metric_columns = sorted(metadata["metrics"].keys())
    fieldnames = [
        "version_id",
        "model_name",
        "trained_at",
        "artifact_path",
        "legacy_model_path",
        "dataset_path",
        "target_column",
        *metric_columns,
    ]

    existing_rows = []
    existing_fieldnames = fieldnames
    if METRICS_LOG_PATH.exists():
        with open(METRICS_LOG_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or fieldnames)

    all_fieldnames = list(dict.fromkeys([*existing_fieldnames, *fieldnames]))
    row = {
        "version_id": metadata["version_id"],
        "model_name": metadata["model_name"],
        "trained_at": metadata["trained_at"],
        "artifact_path": metadata["artifact_path"],
        "legacy_model_path": metadata["legacy_model_path"],
        "dataset_path": metadata["dataset_path"],
        "target_column": metadata["target_column"],
    }
    row.update(metadata["metrics"])

    with open(METRICS_LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)
