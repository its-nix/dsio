"""Identity-preserving raw-sequence dataset, temporal model, and objective."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from dsio.config.components import ComponentConfig
from reference_projects.kaggle.child_mind.components import CLASSES
from reference_projects.kaggle.child_mind.data import TABULAR_FEATURES

DATASET: ComponentConfig = {
    "reference": "dsio.experimental.data.windows:WindowItems",
    "parameters": {
        "x": {
            "parts": [
                {
                    "from": "entity_attribute",
                    "attribute": "tabular",
                    "dtype": "float32",
                    "shape": [-1],
                },
                {"from": "window", "dtype": "float32", "shape": [-1]},
                {
                    "from": "entity_attribute",
                    "attribute": "sensor_present",
                    "dtype": "float32",
                    "shape": [1],
                },
            ]
        },
        "y": {
            "from": "entity_attribute",
            "attribute": "label",
            "dtype": "int64",
        },
        "sample_weight": {"from": "fitted"},
        "group": True,
    },
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
        "sample_weighted": True,
    },
}

OUTPUT: ComponentConfig = {
    "reference": "dsio.experimental.inference.outputs.multiclass:MulticlassOutput",
    "parameters": {"classes": CLASSES, "scores": True, "score_field": "probability"},
}


def model_config(*, window_length: int, sensor_features: int, hidden: int = 32) -> ComponentConfig:
    """Return the raw-window CMI model as a named fusion composition."""
    signal_start = TABULAR_FEATURES * 2
    presence = signal_start + window_length * sensor_features
    return {
        "reference": "dsio.experimental.model.fusion:MultimodalFusion",
        "parameters": {
            "input_shape": [presence + 1],
            "order": ["tabular", "sensor"],
            "modalities": {
                "tabular": {
                    "slice": [0, signal_start],
                    "encoder": {
                        "reference": "dsio.experimental.model.compositions:Stages",
                        "parameters": {
                            "stages": [
                                {
                                    "reference": (
                                        "dsio.experimental.model.standardization:"
                                        "PackedObservedStandardize"
                                    )
                                },
                                {
                                    "reference": "torch.nn:Linear",
                                    "parameters": {
                                        "in_features": signal_start,
                                        "out_features": hidden,
                                    },
                                },
                                {"reference": "torch.nn:GELU"},
                            ]
                        },
                    },
                },
                "sensor": {
                    "slice": [signal_start, presence],
                    "presence": presence,
                    "append_presence": True,
                    "encoder": {
                        "reference": "dsio.experimental.model.compositions:Stages",
                        "parameters": {
                            "stages": [
                                {
                                    "reference": "torch.nn:Unflatten",
                                    "parameters": {
                                        "dim": 1,
                                        "unflattened_size": [window_length, sensor_features],
                                    },
                                },
                                {
                                    "reference": (
                                        "dsio.experimental.model.layout:TimeMajorToChannelFirst"
                                    ),
                                    "parameters": {
                                        "channels": sensor_features,
                                        "time": window_length,
                                    },
                                },
                                {
                                    "reference": (
                                        "dsio.experimental.model.standardization:"
                                        "InstanceStandardize"
                                    ),
                                    "parameters": {
                                        "eps": 1e-6,
                                        "nan_policy": "omit",
                                        "emit_observed": True,
                                    },
                                },
                                {
                                    "reference": (
                                        "dsio.experimental.model.convolution:PooledConv1d"
                                    ),
                                    "parameters": {
                                        "channels": sensor_features * 2,
                                        "hidden": hidden,
                                        "depth": 2,
                                        "kernel_size": 9,
                                        "stride": 4,
                                        "padding": 4,
                                        "activation": "gelu",
                                    },
                                },
                            ]
                        },
                    },
                },
            },
            "head": {
                "reference": "dsio.experimental.model.compositions:Stages",
                "parameters": {
                    "stages": [
                        {
                            "reference": "torch.nn:Linear",
                            "parameters": {
                                "in_features": hidden * 2 + 1,
                                "out_features": hidden,
                            },
                        },
                        {"reference": "torch.nn:GELU"},
                        {
                            "reference": "torch.nn:Linear",
                            "parameters": {
                                "in_features": hidden,
                                "out_features": CLASSES,
                            },
                        },
                    ]
                },
            },
        },
    }


def model_runtime_parameters(
    *, tabular_center: Sequence[float], tabular_scale: Sequence[float]
) -> dict[str, Any]:
    """Route fitted tabular statistics into the nested tabular encoder."""
    return {
        "encoder_parameters": {
            "tabular": {"stage_parameters": {"0": {"mean": tabular_center, "scale": tabular_scale}}}
        }
    }


def ablate_sequence_sensor(x: Tensor, *, window_length: int, sensor_features: int) -> Tensor:
    expected = TABULAR_FEATURES * 2 + window_length * sensor_features + 1
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"CMI sequence ablation expects [batch, {expected}]")
    result = x.clone()
    result[:, TABULAR_FEATURES * 2 :] = 0
    return result


__all__ = [
    "DATASET",
    "OBJECTIVE",
    "OUTPUT",
    "ablate_sequence_sensor",
    "model_config",
    "model_runtime_parameters",
]
