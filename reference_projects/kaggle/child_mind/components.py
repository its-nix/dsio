"""Packed multimodal components for the project-owned CMI experiment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dsio.config.components import ComponentConfig
from reference_projects.kaggle.child_mind.data import (
    PACKED_FEATURES,
    SENSOR_FEATURES,
    SENSOR_MASK_SLICE,
    SENSOR_PRESENT_INDEX,
    SENSOR_VALUE_SLICE,
    TABULAR_FEATURES,
)

CLASSES = 4


DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {
        "x": {"from": "data", "dtype": "float32"},
        "y": {"from": "attribute", "attribute": "target", "dtype": "int64"},
    },
}

INPUTS: ComponentConfig = {
    "reference": "dsio.experimental.data.items:StoredItems",
    "parameters": {"x": {"from": "data", "dtype": "float32"}},
}

OBJECTIVE: ComponentConfig = {
    "reference": "dsio.experimental.model.objectives:SupervisedObjective",
    "parameters": {
        "loss": {"reference": "torch.nn:CrossEntropyLoss"},
        "target_dtype": "int64",
        "target_shape": [],
        "metrics": {
            "accuracy": {"reference": "dsio.experimental.model.objectives:ClassIndexAccuracy"}
        },
    },
}

OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.multiclass:MulticlassOutput",
    "parameters": {"classes": CLASSES, "scores": True, "score_field": "probability"},
}


def _stages(*stages: ComponentConfig) -> ComponentConfig:
    return {
        "reference": "dsio.experimental.model.compositions:Stages",
        "parameters": {"stages": list(stages)},
    }


def _linear(inputs: int, outputs: int) -> ComponentConfig:
    return {
        "reference": "torch.nn:Linear",
        "parameters": {"in_features": inputs, "out_features": outputs},
    }


def model_config(*, hidden: int = 32, use_sensor: bool = True) -> ComponentConfig:
    """Return the flat CMI model as a fully named component composition."""
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError("hidden must be a positive integer")
    if not isinstance(use_sensor, bool):
        raise ValueError("use_sensor must be true or false")
    modalities = {
        "tabular": {
            "slice": [0, TABULAR_FEATURES * 2],
            "encoder": _stages(
                {
                    "reference": (
                        "dsio.experimental.model.standardization:PackedObservedStandardize"
                    )
                },
                _linear(TABULAR_FEATURES * 2, hidden),
                {"reference": "torch.nn:ReLU"},
            ),
        },
        "sensor": {
            "slice": [SENSOR_VALUE_SLICE.start, SENSOR_MASK_SLICE.stop],
            "presence": SENSOR_PRESENT_INDEX,
            "enabled": use_sensor,
            "append_presence": True,
            "encoder": _stages(
                {
                    "reference": (
                        "dsio.experimental.model.standardization:PackedObservedStandardize"
                    )
                },
                _linear(SENSOR_FEATURES * 2, hidden),
                {"reference": "torch.nn:ReLU"},
            ),
        },
    }
    return {
        "reference": "dsio.experimental.model.fusion:MultimodalFusion",
        "parameters": {
            "input_shape": [PACKED_FEATURES],
            "allow_leading_singleton": True,
            "order": ["tabular", "sensor"],
            "modalities": modalities,
            "head": _stages(
                _linear(hidden * 2 + 1, hidden),
                {"reference": "torch.nn:ReLU"},
                _linear(hidden, CLASSES),
            ),
        },
    }


def model_runtime_parameters(
    *,
    tabular_center: Sequence[float],
    tabular_scale: Sequence[float],
    sensor_center: Sequence[float],
    sensor_scale: Sequence[float],
) -> dict[str, Any]:
    """Route fitted statistics into both modality encoders."""
    return {
        "encoder_parameters": {
            "tabular": {
                "stage_parameters": {"0": {"mean": tabular_center, "scale": tabular_scale}}
            },
            "sensor": {"stage_parameters": {"0": {"mean": sensor_center, "scale": sensor_scale}}},
        }
    }


__all__ = [
    "CLASSES",
    "DATASET",
    "INPUTS",
    "OBJECTIVE",
    "OUTPUT",
    "model_config",
    "model_runtime_parameters",
]
