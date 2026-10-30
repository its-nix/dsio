"""Evaluate immutable MLflow evidence inside a caller-owned task."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from mlflow import MlflowClient
from mlflow.entities import (
    DatasetInput,
    InputTag,
    LoggedModelInput,
    LoggedModelStatus,
    Metric,
    Run,
)

from dsio.eval.metrics import METRICS, compute
from dsio.inference.loading import InferenceError, predict

_LOGGED_MODEL_URI = re.compile(r"models:/(m-[0-9a-f]{32})")
_TARGET_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_MLFLOW_MAX_KEY_LENGTH = 250
_MLFLOW_MAX_PARAM_VALUE_LENGTH = 6_000
_MLFLOW_MAX_METRICS_PER_BATCH = 1_000
_ATTEMPT_TAGS = (
    "dsio.prefect.flow_run_id",
    "dsio.prefect.task_key",
    "dsio.prefect.task_run_id",
    "dsio.prefect.dynamic_key",
    "dsio.prefect.attempt",
)


class EvaluationError(ValueError):
    """Evaluation evidence, arrays, metrics, or persistence violated its contract."""


def evaluate(
    *,
    run_id: str,
    model_uri: str,
    dataset_run_id: str,
    inputs: Mapping[str, Any],
    targets: np.ndarray[Any, Any],
    metrics: Sequence[str],
    prediction_field: str = "prediction",
    score_field: str | None = None,
    mask: np.ndarray[Any, Any] | None = None,
    target_names: Sequence[str] | None = None,
    subgroups: Mapping[str, np.ndarray[Any, Any]] | None = None,
    subgroup_metrics: Mapping[str, str] | None = None,
    ablations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute and record evaluation evidence on one caller-owned attempt Run.

    ``mask`` selects scoreable positions before metrics run. With ``target_names``, the
    final target axis is scored independently and each metric returns one value per name
    plus its arithmetic mean. The mask then matches every target axis except that named
    final axis. Without names it matches the complete target shape. ``subgroups`` are
    independent row masks scored with alias-to-metric ``subgroup_metrics``. Each named
    ``ablation`` selects one subgroup, copies and fills a declared last-axis input slice,
    and reports the subgroup's full metric, ablated metric, and full-minus-ablated delta.
    Row analyses intentionally do not combine with element masks or named target axes.
    """
    client = MlflowClient()
    evaluation_run = _attempt_run(client, run_id)
    _require_clean_attempt(client, evaluation_run)
    model_id, model_source_run_id = _model_evidence(client, model_uri)
    dataset_input = _dataset_evidence(client, dataset_run_id)
    names = _metric_names(metrics)
    _field_name(prediction_field, "prediction")
    if score_field is not None:
        _field_name(score_field, "score")
    if not isinstance(targets, np.ndarray):
        raise EvaluationError(
            f"evaluation targets must be a NumPy array, got {type(targets).__name__}"
        )
    if targets.ndim == 0 or targets.shape[0] == 0:
        raise EvaluationError("evaluation targets must contain at least one row")
    resolved_target_names = _target_names(target_names, targets, names)
    resolved_mask = _mask(mask, targets, resolved_target_names)
    resolved_subgroups, resolved_subgroup_metrics, resolved_ablations = _analyses(
        inputs=inputs,
        targets=targets,
        subgroups=subgroups,
        subgroup_metrics=subgroup_metrics,
        ablations=ablations,
        full_metric_names=names,
        mask=resolved_mask,
        target_names=resolved_target_names,
    )

    try:
        outputs = predict(model_uri, inputs)
    except InferenceError as error:
        raise EvaluationError(f"prediction boundary is incompatible: {error}") from error
    predicted = _output(outputs, prediction_field, "prediction")
    if predicted.shape != targets.shape:
        raise EvaluationError(
            f"evaluation target shape {targets.shape} does not match prediction field "
            f"{prediction_field!r} shape {predicted.shape}"
        )
    score = None if score_field is None else _output(outputs, score_field, "score")
    metric_targets, metric_predicted, metric_score = _metric_arrays(
        targets,
        predicted,
        score,
        resolved_mask,
        resolved_target_names,
        score_field,
    )
    if metric_targets.size == 0:
        raise EvaluationError("evaluation targets must contain at least one scoreable value")
    if resolved_target_names is None:
        _validate_metric_arrays(metric_targets, metric_predicted, metric_score, score_field)
    else:
        for index in range(len(resolved_target_names)):
            _validate_metric_arrays(
                metric_targets[:, index],
                metric_predicted[:, index],
                None if metric_score is None else metric_score[:, index],
                score_field,
            )
    try:
        values = _compute_metrics(
            names,
            metric_targets,
            metric_predicted,
            metric_score,
            resolved_target_names,
        )
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(
            f"metric inputs or configuration are incompatible: {error}"
        ) from error
    if any(not math.isfinite(value) for value in values.values()):
        raise EvaluationError("evaluation metrics must all be finite")

    analysis_values, analysis_evidence = _analyze(
        model_uri=model_uri,
        inputs=inputs,
        targets=targets,
        outputs=outputs,
        prediction_field=prediction_field,
        score_field=score_field,
        subgroups=resolved_subgroups,
        subgroup_metrics=resolved_subgroup_metrics,
        ablations=resolved_ablations,
    )
    values.update(analysis_values)

    _require_clean_attempt(client, _attempt_run(client, run_id))
    _model_evidence(client, model_uri)
    _dataset_evidence(client, dataset_run_id)
    linked_dataset = _evaluation_dataset_input(dataset_input, dataset_run_id)
    try:
        with TemporaryDirectory(prefix="dsio-evaluation-") as directory:
            artifact = Path(directory) / "predictions.npz"
            evidence = {
                "target": targets,
                **{f"prediction__{name}": value for name, value in outputs.items()},
                **analysis_evidence,
            }
            if resolved_mask is not None:
                evidence["mask"] = resolved_mask
            if resolved_target_names is not None:
                evidence["target_names"] = np.asarray(resolved_target_names, dtype=np.str_)
            # NumPy accepts arbitrary named arrays; its stub mistakes one for the
            # reserved allow_pickle keyword.
            np.savez_compressed(artifact, **evidence)  # type: ignore[arg-type]
            client.log_artifact(run_id, str(artifact), artifact_path="evaluation")
        client.log_inputs(
            run_id,
            datasets=[linked_dataset],
            models=[LoggedModelInput(model_id)],
        )
        client.set_tag(run_id, "dsio.evaluation.model_source_run_id", model_source_run_id)
        client.set_tag(run_id, "dsio.evaluation.dataset_source_run_id", dataset_run_id)
        client.log_param(run_id, "evaluation.metrics", json.dumps(names))
        client.log_param(run_id, "evaluation.prediction_field", prediction_field)
        client.log_param(run_id, "evaluation.masked", json.dumps(resolved_mask is not None))
        if resolved_target_names is not None:
            client.log_param(run_id, "evaluation.target_names", json.dumps(resolved_target_names))
        if score_field is not None:
            client.log_param(run_id, "evaluation.score_field", score_field)
        if resolved_subgroups:
            identities = _input_identities(inputs, len(targets))
            client.log_param(
                run_id,
                "evaluation.subgroups",
                json.dumps(
                    {
                        name: identities[selected].tolist()
                        for name, selected in sorted(resolved_subgroups.items())
                    },
                    sort_keys=True,
                ),
            )
            client.log_param(
                run_id,
                "evaluation.subgroup_metrics",
                json.dumps(resolved_subgroup_metrics, sort_keys=True),
            )
        if resolved_ablations:
            client.log_param(
                run_id,
                "evaluation.ablations",
                json.dumps(resolved_ablations, sort_keys=True),
            )
        _attempt_run(client, run_id)
        _model_evidence(client, model_uri)
        _dataset_evidence(client, dataset_run_id)
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[
                Metric(
                    name,
                    value,
                    timestamp,
                    0,
                    dataset_name=dataset_input.dataset.name,
                    dataset_digest=dataset_input.dataset.digest,
                )
                for name, value in values.items()
            ],
        )
    except EvaluationError:
        raise
    except Exception as error:
        raise EvaluationError(
            f"could not persist native MLflow evaluation evidence: {error}"
        ) from error
    return values


def _analyses(
    *,
    inputs: Mapping[str, Any],
    targets: np.ndarray[Any, Any],
    subgroups: Mapping[str, np.ndarray[Any, Any]] | None,
    subgroup_metrics: Mapping[str, str] | None,
    ablations: Mapping[str, Mapping[str, Any]] | None,
    full_metric_names: tuple[str, ...],
    mask: np.ndarray[Any, Any] | None,
    target_names: tuple[str, ...] | None,
) -> tuple[
    dict[str, np.ndarray[Any, Any]],
    dict[str, str],
    dict[str, dict[str, Any]],
]:
    if subgroups is None and subgroup_metrics is None and ablations is None:
        return {}, {}, {}
    if mask is not None or target_names is not None:
        raise EvaluationError(
            "evaluation subgroups and ablations cannot be combined with mask or target_names"
        )
    if targets.ndim != 1:
        raise EvaluationError("evaluation subgroups require one-dimensional row targets")
    if not isinstance(subgroups, Mapping) or not subgroups:
        raise EvaluationError("evaluation analyses require at least one subgroup")
    if not isinstance(subgroup_metrics, Mapping) or not subgroup_metrics:
        raise EvaluationError("evaluation analyses require subgroup_metrics")
    resolved_subgroups: dict[str, np.ndarray[Any, Any]] = {}
    for name, selected in subgroups.items():
        _analysis_name(name, "subgroup")
        if not isinstance(selected, np.ndarray) or selected.dtype.kind != "b":
            raise EvaluationError(f"evaluation subgroup {name!r} must be a boolean NumPy array")
        if selected.ndim != 1 or selected.shape != (len(targets),):
            raise EvaluationError(
                f"evaluation subgroup {name!r} must have shape ({len(targets)},), "
                f"got {selected.shape}"
            )
        if not bool(selected.any()):
            raise EvaluationError(f"evaluation subgroup {name!r} must select at least one row")
        resolved_subgroups[name] = selected
    resolved_metrics = _analysis_metrics(subgroup_metrics, "subgroup_metrics")
    resolved_ablations: dict[str, dict[str, Any]] = {}
    generated = set(full_metric_names)
    for alias in resolved_metrics:
        for subgroup in resolved_subgroups:
            _generated_metric(generated, f"{alias}.{subgroup}")
    if ablations is not None and not isinstance(ablations, Mapping):
        raise EvaluationError("evaluation ablations must be a mapping")
    for name, raw in (ablations or {}).items():
        _analysis_name(name, "ablation")
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"evaluation ablation {name!r} must be a mapping")
        extra = sorted(set(raw) - {"input_field", "slice", "subgroup", "fill", "metrics"})
        if extra:
            raise EvaluationError(f"evaluation ablation {name!r} has unknown fields {extra}")
        field = raw.get("input_field")
        if not isinstance(field, str) or not field:
            raise EvaluationError(f"evaluation ablation {name!r} input_field must name an input")
        value = inputs.get(field)
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
            raise EvaluationError(
                f"evaluation ablation {name!r} input {field!r} must be a numeric NumPy array"
            )
        if value.ndim < 2 or value.shape[0] != len(targets):
            raise EvaluationError(
                f"evaluation ablation {name!r} input {field!r} must have {len(targets)} "
                "rows and a last feature axis"
            )
        bounds = raw.get("slice")
        if (
            not isinstance(bounds, Sequence)
            or isinstance(bounds, str | bytes)
            or len(bounds) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in bounds)
        ):
            raise EvaluationError(f"evaluation ablation {name!r} slice must be [start, stop]")
        start, stop = bounds
        if start < 0 or stop <= start or stop > value.shape[-1]:
            raise EvaluationError(
                f"evaluation ablation {name!r} slice [{start}, {stop}] must be non-empty "
                f"and within last-axis extent {value.shape[-1]}"
            )
        raw_subgroup = raw.get("subgroup")
        if not isinstance(raw_subgroup, str) or raw_subgroup not in resolved_subgroups:
            raise EvaluationError(
                f"evaluation ablation {name!r} references unknown subgroup {raw_subgroup!r}"
            )
        fill = _fill_value(raw.get("fill", 0), value.dtype, name)
        metrics = _analysis_metrics(raw.get("metrics"), f"ablation {name!r} metrics")
        if set(metrics) - set(resolved_metrics):
            raise EvaluationError(
                f"evaluation ablation {name!r} metric aliases must also be subgroup_metrics"
            )
        if any(resolved_metrics[alias] != metric for alias, metric in metrics.items()):
            raise EvaluationError(
                f"evaluation ablation {name!r} metric aliases must resolve to the same "
                "registered metric as subgroup_metrics"
            )
        for alias in metrics:
            _generated_metric(generated, f"{alias}.{name}_ablated")
            _generated_metric(generated, f"{alias}.{name}_ablation_delta")
        resolved_ablations[name] = {
            "input_field": field,
            "slice": [start, stop],
            "subgroup": raw_subgroup,
            "fill": fill,
            "metrics": metrics,
        }
    if len(generated) > _MLFLOW_MAX_METRICS_PER_BATCH:
        raise EvaluationError(
            f"evaluation analyses generate more than {_MLFLOW_MAX_METRICS_PER_BATCH} MLflow metrics"
        )
    identities = _input_identities(inputs, len(targets))
    parameters = (
        json.dumps(
            {
                name: identities[selected].tolist()
                for name, selected in sorted(resolved_subgroups.items())
            },
            sort_keys=True,
        ),
        json.dumps(resolved_metrics, sort_keys=True),
        json.dumps(resolved_ablations, sort_keys=True),
    )
    if any(len(value) > _MLFLOW_MAX_PARAM_VALUE_LENGTH for value in parameters):
        raise EvaluationError(
            f"evaluation analysis configuration exceeds MLflow's "
            f"{_MLFLOW_MAX_PARAM_VALUE_LENGTH}-character parameter limit"
        )
    return resolved_subgroups, resolved_metrics, resolved_ablations


def _analyze(
    *,
    model_uri: str,
    inputs: Mapping[str, Any],
    targets: np.ndarray[Any, Any],
    outputs: Mapping[str, np.ndarray[Any, Any]],
    prediction_field: str,
    score_field: str | None,
    subgroups: Mapping[str, np.ndarray[Any, Any]],
    subgroup_metrics: Mapping[str, str],
    ablations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, np.ndarray[Any, Any]]]:
    if not subgroups:
        return {}, {}
    identities = _input_identities(inputs, len(targets))
    _output_identities(outputs, identities, "full")
    predicted = _output(outputs, prediction_field, "prediction")
    score = None if score_field is None else _output(outputs, score_field, "score")
    values: dict[str, float] = {}
    evidence: dict[str, np.ndarray[Any, Any]] = {}
    subgroup_values: dict[tuple[str, str], float] = {}
    for subgroup, selected in subgroups.items():
        evidence[f"subgroup__{subgroup}"] = selected
        evidence[f"subgroup__{subgroup}__sample_id"] = identities[selected]
        for alias, metric in subgroup_metrics.items():
            value = _analysis_metric(
                metric,
                targets[selected],
                predicted[selected],
                None if score is None else score[selected],
                score_field,
            )
            values[f"{alias}.{subgroup}"] = value
            subgroup_values[(alias, subgroup)] = value
    for name, declaration in ablations.items():
        selected = subgroups[declaration["subgroup"]]
        ablated_inputs = _selected_inputs(inputs, selected, len(targets))
        field = declaration["input_field"]
        changed = np.asarray(ablated_inputs[field]).copy()
        start, stop = declaration["slice"]
        changed[..., start:stop] = declaration["fill"]
        ablated_inputs[field] = changed
        try:
            ablated_outputs = predict(model_uri, ablated_inputs)
        except InferenceError as error:
            raise EvaluationError(
                f"ablation {name!r} prediction is incompatible: {error}"
            ) from error
        selected_identities = identities[selected]
        _output_identities(ablated_outputs, selected_identities, f"ablation {name!r}")
        ablated_prediction = _output(ablated_outputs, prediction_field, "prediction")
        if ablated_prediction.shape != targets[selected].shape:
            raise EvaluationError(
                f"evaluation ablation {name!r} target shape {targets[selected].shape} does not "
                f"match prediction shape {ablated_prediction.shape}"
            )
        ablated_score = (
            None if score_field is None else _output(ablated_outputs, score_field, "score")
        )
        evidence[f"ablation__{name}__sample_id"] = selected_identities
        for field_name, output_value in ablated_outputs.items():
            evidence[f"ablation__{name}__prediction__{field_name}"] = output_value
        for alias, metric in declaration["metrics"].items():
            ablated_value = _analysis_metric(
                metric,
                targets[selected],
                ablated_prediction,
                ablated_score,
                score_field,
            )
            values[f"{alias}.{name}_ablated"] = ablated_value
            values[f"{alias}.{name}_ablation_delta"] = (
                subgroup_values[(alias, declaration["subgroup"])] - ablated_value
            )
    if any(not math.isfinite(value) for value in values.values()):
        raise EvaluationError("evaluation analysis metrics must all be finite")
    return values, evidence


def _analysis_name(name: object, role: str) -> None:
    if not isinstance(name, str) or not _TARGET_NAME.fullmatch(name) or name == "mean":
        raise EvaluationError(
            f"evaluation {role} names must start with a letter, contain only letters, "
            "numbers, underscores, or hyphens, and cannot use 'mean'"
        )


def _analysis_metrics(value: object, role: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise EvaluationError(f"evaluation {role} must be a non-empty alias-to-metric mapping")
    result: dict[str, str] = {}
    for alias, metric in value.items():
        _analysis_name(alias, "metric alias")
        if not isinstance(metric, str) or not metric:
            raise EvaluationError(f"evaluation {role} metric names must be non-empty strings")
        try:
            METRICS.get(metric)
        except Exception as error:
            raise EvaluationError(
                f"evaluation {role} contains unknown metric {metric!r}"
            ) from error
        result[alias] = metric
    return result


def _generated_metric(existing: set[str], name: str) -> None:
    if len(name) > _MLFLOW_MAX_KEY_LENGTH:
        raise EvaluationError(
            f"evaluation analyses generate an MLflow metric name longer than "
            f"{_MLFLOW_MAX_KEY_LENGTH} characters"
        )
    if name in existing:
        raise EvaluationError(f"evaluation metric name {name!r} is generated more than once")
    existing.add(name)


def _fill_value(value: object, dtype: np.dtype[Any], ablation: str) -> int | float:
    candidate = np.asarray(value)
    if candidate.ndim != 0 or candidate.dtype.kind not in "iuf" or not bool(np.isfinite(candidate)):
        raise EvaluationError(f"evaluation ablation {ablation!r} fill must be a finite scalar")
    if not np.can_cast(candidate.dtype, dtype, casting="same_kind"):
        raise EvaluationError(
            f"evaluation ablation {ablation!r} fill cannot be represented by input dtype {dtype}"
        )
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        numeric = candidate.item()
        if numeric < limits.min or numeric > limits.max:
            raise EvaluationError(
                f"evaluation ablation {ablation!r} fill cannot be represented by input dtype "
                f"{dtype}"
            )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = candidate.astype(dtype).item()
    if not math.isfinite(float(converted)):
        raise EvaluationError(
            f"evaluation ablation {ablation!r} fill cannot be represented by input dtype {dtype}"
        )
    return converted


def _input_identities(inputs: Mapping[str, Any], rows: int) -> np.ndarray[Any, Any]:
    identities = inputs.get("sample_id")
    if not isinstance(identities, np.ndarray) or identities.ndim != 1 or len(identities) != rows:
        raise EvaluationError(
            f"evaluation analyses require sample_id as a one-dimensional NumPy array "
            f"with {rows} rows"
        )
    if identities.dtype.kind != "U" or len(set(identities.tolist())) != rows:
        raise EvaluationError("evaluation analysis sample_id values must be unique strings")
    return identities


def _output_identities(
    outputs: Mapping[str, Any], expected: np.ndarray[Any, Any], role: str
) -> None:
    identities = outputs.get("sample_id")
    if not isinstance(identities, np.ndarray) or not np.array_equal(identities, expected):
        raise EvaluationError(f"evaluation {role} prediction changed sample identity order")


def _selected_inputs(
    inputs: Mapping[str, Any], selected: np.ndarray[Any, Any], rows: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in inputs.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == rows:
            result[name] = value[selected]
        else:
            result[name] = value
    return result


def _analysis_metric(
    metric: str,
    targets: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any] | None,
    score_field: str | None,
) -> float:
    _validate_metric_arrays(targets, predicted, score, score_field)
    try:
        value = compute((metric,), targets, predicted, score)[metric]
    except Exception as error:
        raise EvaluationError(
            f"evaluation analysis metric {metric!r} is incompatible: {error}"
        ) from error
    if not math.isfinite(value):
        raise EvaluationError(f"evaluation analysis metric {metric!r} must be finite")
    return value


def _attempt_run(client: MlflowClient, run_id: str) -> Run:
    try:
        run = client.get_run(run_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation attempt Run {run_id!r} cannot be resolved: {error}"
        ) from error
    if run.info.lifecycle_stage != "active" or run.info.status != "RUNNING":
        raise EvaluationError(f"evaluation attempt Run {run_id!r} must be active and RUNNING")
    if "mlflow.parentRunId" in run.data.tags or any(
        not run.data.tags.get(tag) for tag in _ATTEMPT_TAGS
    ):
        raise EvaluationError(
            f"evaluation Run {run_id!r} must be a top-level tracked Prefect attempt"
        )
    return run


def _require_clean_attempt(client: MlflowClient, run: Run) -> None:
    inputs = run.inputs
    has_inputs = inputs is not None and bool(inputs.dataset_inputs or inputs.model_inputs)
    evaluation_metadata = any(key.startswith("evaluation.") for key in run.data.params) or any(
        key.startswith("dsio.evaluation.") for key in run.data.tags
    )
    try:
        artifacts = client.list_artifacts(run.info.run_id, "evaluation")
    except Exception as error:
        raise EvaluationError(
            f"evaluation attempt Run {run.info.run_id!r} artifacts cannot be inspected: {error}"
        ) from error
    if run.data.metrics or has_inputs or evaluation_metadata or artifacts:
        raise EvaluationError(
            f"evaluation attempt Run {run.info.run_id!r} already contains evaluation evidence"
        )


def _model_evidence(client: MlflowClient, model_uri: str) -> tuple[str, str]:
    matched = _LOGGED_MODEL_URI.fullmatch(model_uri) if isinstance(model_uri, str) else None
    if matched is None:
        raise EvaluationError("evaluation model must use an immutable MLflow Logged Model URI")
    model_id = matched.group(1)
    try:
        model = client.get_logged_model(model_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation model {model_id!r} cannot be resolved: {error}"
        ) from error
    if (
        model.status != LoggedModelStatus.READY
        or model.model_uri != model_uri
        or model.model_type != "dsio.predictor"
        or model.tags.get("dsio.export_form") != "pyfunc"
        or not model.source_run_id
    ):
        raise EvaluationError(f"evaluation model {model_id!r} is not READY DSIO PyFunc evidence")
    source = _finished_run(client, model.source_run_id, "model source")
    return model_id, source.info.run_id


def _dataset_evidence(client: MlflowClient, run_id: str) -> DatasetInput:
    source = _finished_run(client, run_id, "dataset source")
    inputs = [] if source.inputs is None else source.inputs.dataset_inputs
    if len(inputs) != 1:
        raise EvaluationError(
            f"evaluation dataset source Run {run_id!r} must contain exactly one native "
            f"dataset input, found {len(inputs)}"
        )
    dataset_input = inputs[0]
    dataset = dataset_input.dataset
    if not dataset.name or not dataset.digest or not dataset.source_type or not dataset.source:
        raise EvaluationError("evaluation dataset input has incomplete native MLflow identity")
    return dataset_input


def _finished_run(client: MlflowClient, run_id: str, role: str) -> Run:
    try:
        run = client.get_run(run_id)
    except Exception as error:
        raise EvaluationError(
            f"evaluation {role} Run {run_id!r} cannot be resolved: {error}"
        ) from error
    if run.info.lifecycle_stage != "active" or run.info.status != "FINISHED":
        raise EvaluationError(f"evaluation {role} Run {run_id!r} must be active and FINISHED")
    return run


def _evaluation_dataset_input(source: DatasetInput, source_run_id: str) -> DatasetInput:
    tags = {tag.key: tag.value for tag in source.tags}
    tags.update(
        {
            "mlflow.data.context": "evaluation",
            "dsio.evaluation.source_run_id": source_run_id,
        }
    )
    return DatasetInput(
        source.dataset,
        [InputTag(key, value) for key, value in sorted(tags.items())],
    )


def _metric_names(metrics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(metrics, str) or not isinstance(metrics, Sequence) or not metrics:
        raise EvaluationError("evaluation requires at least one metric name")
    names = tuple(metrics)
    if any(not isinstance(name, str) or not name for name in names):
        raise EvaluationError("evaluation metric names must be non-empty strings")
    if len(set(names)) != len(names):
        raise EvaluationError("evaluation metric names must be unique")
    return names


def _field_name(name: str, role: str) -> None:
    if not isinstance(name, str) or not name or name == "sample_id":
        raise EvaluationError(f"evaluation {role} field must be a non-empty non-identity string")


def _target_names(
    names: Sequence[str] | None,
    targets: np.ndarray[Any, Any],
    metric_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    if names is None:
        return None
    if isinstance(names, str) or not isinstance(names, Sequence) or not names:
        raise EvaluationError("evaluation target_names must be a non-empty sequence")
    resolved = tuple(names)
    if any(
        not isinstance(name, str) or not _TARGET_NAME.fullmatch(name) or name == "mean"
        for name in resolved
    ):
        raise EvaluationError(
            "evaluation target_names must start with a letter, contain only letters, "
            "numbers, underscores, or hyphens, and cannot use 'mean'"
        )
    if len(set(resolved)) != len(resolved):
        raise EvaluationError("evaluation target_names must be unique")
    if targets.ndim < 2 or targets.shape[-1] != len(resolved):
        raise EvaluationError(
            f"evaluation target_names has {len(resolved)} names but target last axis "
            f"has shape {targets.shape}"
        )
    generated_names = [
        f"{metric_name}.{target_name}"
        for metric_name in metric_names
        for target_name in (*resolved, "mean")
    ]
    if any(len(name) > _MLFLOW_MAX_KEY_LENGTH for name in generated_names):
        raise EvaluationError(
            f"evaluation target_names generate an MLflow metric name longer than "
            f"{_MLFLOW_MAX_KEY_LENGTH} characters"
        )
    if len(generated_names) > _MLFLOW_MAX_METRICS_PER_BATCH:
        raise EvaluationError(
            f"evaluation target_names generate more than {_MLFLOW_MAX_METRICS_PER_BATCH} "
            "MLflow metrics"
        )
    if len(json.dumps(resolved)) > _MLFLOW_MAX_PARAM_VALUE_LENGTH:
        raise EvaluationError(
            f"evaluation target_names exceed MLflow's {_MLFLOW_MAX_PARAM_VALUE_LENGTH}-character "
            "parameter limit"
        )
    return resolved


def _mask(
    mask: np.ndarray[Any, Any] | None,
    targets: np.ndarray[Any, Any],
    target_names: tuple[str, ...] | None,
) -> np.ndarray[Any, Any] | None:
    if mask is None:
        return None
    if not isinstance(mask, np.ndarray) or mask.dtype.kind != "b":
        raise EvaluationError("evaluation mask must be a boolean NumPy array")
    expected = targets.shape[:-1] if target_names is not None else targets.shape
    if mask.shape != expected:
        raise EvaluationError(
            f"evaluation mask shape {mask.shape} does not match expected shape {expected}"
        )
    if not bool(mask.any()):
        raise EvaluationError("evaluation mask must select at least one target")
    return mask


def _metric_arrays(
    targets: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any] | None,
    mask: np.ndarray[Any, Any] | None,
    target_names: tuple[str, ...] | None,
    score_field: str | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    if target_names is not None and score is not None and score.shape != targets.shape:
        raise EvaluationError(
            f"evaluation score field {score_field!r} shape {score.shape} must match "
            f"named target shape {targets.shape}"
        )
    if mask is not None and score is not None and score.shape[: mask.ndim] != mask.shape:
        raise EvaluationError(
            f"evaluation score field {score_field!r} shape {score.shape} cannot use "
            f"mask shape {mask.shape}"
        )
    if mask is not None:
        return targets[mask], predicted[mask], None if score is None else score[mask]
    if target_names is not None:
        width = targets.shape[-1]
        return (
            targets.reshape(-1, width),
            predicted.reshape(-1, width),
            None if score is None else score.reshape(-1, width),
        )
    return targets, predicted, score


def _compute_metrics(
    names: tuple[str, ...],
    targets: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any] | None,
    target_names: tuple[str, ...] | None,
) -> dict[str, float]:
    if target_names is None:
        return compute(names, targets, predicted, score)
    values: dict[str, float] = {}
    for metric_name in names:
        per_target = []
        for index, target_name in enumerate(target_names):
            try:
                value = compute(
                    (metric_name,),
                    targets[:, index],
                    predicted[:, index],
                    None if score is None else score[:, index],
                )[metric_name]
            except Exception as error:
                raise EvaluationError(
                    f"evaluation metric {metric_name!r} failed for target {target_name!r}: {error}"
                ) from error
            values[f"{metric_name}.{target_name}"] = value
            per_target.append(value)
        values[f"{metric_name}.mean"] = float(np.mean(per_target))
    return values


def _output(
    outputs: Mapping[str, np.ndarray[Any, Any]], field: str, role: str
) -> np.ndarray[Any, Any]:
    value = outputs.get(field)
    if not isinstance(value, np.ndarray):
        raise EvaluationError(
            f"evaluation {role} field {field!r} is missing or is not a NumPy array"
        )
    return value


def _validate_metric_arrays(
    targets: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any] | None,
    score_field: str | None,
) -> None:
    target_family = _dtype_family(targets, "target")
    prediction_family = _dtype_family(predicted, "prediction")
    if target_family != prediction_family:
        raise EvaluationError(
            f"evaluation target dtype {targets.dtype} is incompatible with prediction "
            f"dtype {predicted.dtype}"
        )
    _require_finite(targets, "target")
    _require_finite(predicted, "prediction")
    if score is None:
        return
    if score.dtype.kind not in "buif":
        raise EvaluationError(
            f"evaluation score field {score_field!r} must contain real numeric values"
        )
    if targets.ndim != 1 or predicted.ndim != 1:
        raise EvaluationError(
            "evaluation targets and predictions must be one-dimensional when a score "
            "field is configured"
        )
    if score.ndim not in {1, 2} or score.shape[0] != targets.shape[0]:
        raise EvaluationError(
            f"evaluation score field {score_field!r} has incompatible shape {score.shape}"
        )
    if score.ndim == 2 and score.shape[1] != 2:
        raise EvaluationError(
            f"evaluation score field {score_field!r} must have one value or two class "
            "columns per target"
        )
    _require_finite(score, "score")


def _dtype_family(value: np.ndarray[Any, Any], role: str) -> str:
    if value.dtype.kind in "buif":
        return "numeric"
    if value.dtype.kind == "U":
        return "unicode"
    if value.dtype.kind == "S":
        return "bytes"
    raise EvaluationError(
        f"evaluation {role} dtype {value.dtype} is unsupported; object and complex "
        "arrays are not safe metric evidence"
    )


def _require_finite(value: np.ndarray[Any, Any], role: str) -> None:
    if value.dtype.kind in "buif" and not bool(np.isfinite(value).all()):
        raise EvaluationError(f"evaluation {role} values must all be finite")
