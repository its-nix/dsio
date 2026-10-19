"""Statistics fitted on one split role, recorded as evidence."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from dsio.data.adapters import SignalExamples
from dsio.data.splits.models import SplitError, SplitFile
from dsio.data.store import SignalStore
from dsio.data.views import assert_index_matches_store


def _training_assignment(
    store: SignalStore, manifest: SplitFile, partition: int
) -> tuple[str, list[str]]:
    store_name = str(store.path.name)
    if manifest.store != store_name:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset {manifest.store!r}, not {store_name!r}"
        )
    store_digest = store.identity
    if manifest.store_manifest_sha256 != store_digest:
        raise SplitError(
            f"split {manifest.name!r} was built for dataset digest "
            f"{manifest.store_manifest_sha256!r}, not {store_digest!r}"
        )
    if not manifest.required_roles:
        raise SplitError(f"split {manifest.name!r} declares no training role")
    role = manifest.required_roles[0]
    assignments = manifest.fold(partition).assignments
    try:
        sample_ids = list(assignments[role])
    except KeyError:
        raise SplitError(
            f"split {manifest.name!r} partition {partition} is missing declared training "
            f"role {role!r}"
        ) from None
    if not sample_ids:
        raise ValueError(
            f"split training role {role!r} in partition {partition} assigns no samples"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise SplitError(
            f"split training role {role!r} in partition {partition} repeats a sample identity"
        )
    return role, sample_ids


def _column_indexes(columns: slice | Sequence[int] | None, width: int, name: str) -> list[int]:
    if width <= 0:
        raise ValueError("standardization source must contain at least one feature")
    if columns is None:
        return list(range(width))
    if isinstance(columns, slice):
        indexes = list(range(width))[columns]
        if not indexes:
            raise ValueError(f"{name} must select at least one integer column")
        return indexes
    indexes = list(columns)
    if not indexes or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indexes
    ):
        raise ValueError(f"{name} must select at least one integer column")
    normalized = [index % width if index < 0 else index for index in indexes]
    if any(index < 0 or index >= width for index in normalized):
        raise ValueError(f"{name} contains a column outside [0, {width})")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not repeat a column")
    return normalized


def _window_entities(
    store: SignalStore, examples: SignalExamples | None, sample_ids: list[str], *, role: str
) -> list[str]:
    if not isinstance(examples, SignalExamples):
        raise ValueError("attribute fitting from window assignments requires SignalExamples")
    if examples.name != store.path.name or examples.digest != store.identity:
        raise ValueError("SignalExamples must describe the same store name and content identity")
    assert_index_matches_store(store, examples.index)
    governed_ids = list(examples.sample_ids)
    if len(set(governed_ids)) != len(governed_ids):
        raise ValueError("SignalExamples repeats a governed window identity")
    positions = {sample_id: position for position, sample_id in enumerate(governed_ids)}
    unknown = [sample_id for sample_id in sample_ids if sample_id not in positions]
    if unknown:
        raise SplitError(
            f"split training role {role!r} names unknown window identity {unknown[0]!r}"
        )
    entity_ids: list[str] = []
    seen: set[str] = set()
    for sample_id in sample_ids:
        position = positions[sample_id]
        code = int(examples.index.entity_codes[position])
        if code < 0 or code >= len(examples.index.entity_names):
            raise ValueError(f"window {sample_id!r} has invalid entity code {code}")
        entity_id = examples.index.entity_names[code]
        entity = store.entity(entity_id)
        indexed_group = examples.index.entity_groups[code]
        if entity.group != indexed_group:
            raise ValueError(
                f"window {sample_id!r} indexes group {indexed_group!r}, but entity "
                f"{entity_id!r} belongs to {entity.group!r}"
            )
        if entity_id not in seen:
            seen.add(entity_id)
            entity_ids.append(entity_id)
    return entity_ids


def fit_standardization(
    store: SignalStore,
    manifest: SplitFile,
    *,
    partition: int,
    observed: bool = False,
    value_columns: slice | Sequence[int] | None = None,
    observed_columns: slice | Sequence[int] | None = None,
    examples: SignalExamples | None = None,
    attribute: str | None = None,
    feature_names: Sequence[str] | None = None,
    minimum_scale: float = 0.0,
    reduction: Literal["numpy", "sum_squares"] = "numpy",
) -> dict[str, Any]:
    """Fit per-feature mean and scale over the samples assigned to training.

    The manifest's store binding and training assignment are validated before any sample
    is read. The training role is the first role declared by the manifest and cannot be
    overridden by the caller.

    Consumes:
        A ``SignalStore`` of ``[rows, features]`` samples and the replayed split manifest.

    Produces:
        Float64 ``mean``/``scale``, the governed role/partition/window identities, and,
        for selected sources, resolved entity identities plus the complete selection policy.
        Scale is the population standard deviation with configured tiny scales mapped to 1.

    Parameters:
        ``partition``: split partition index; ``observed``: omit NaNs; ``value_columns`` and
        ``observed_columns``: paired feature/mask selection; ``examples`` and ``attribute``:
        resolve window assignments to unique entity attributes; ``feature_names``: names for
        missing-feature errors; ``minimum_scale``: exclusive tiny-scale threshold;
        ``reduction``: NumPy-style centered or sum-of-squares population variance.

    Devices:
        CPU (NumPy); fitted once before training.

    Limitations:
        Loads the role's selected values into memory together. A feature with no observed
        value raises rather than inventing statistics. Explicit masks must be paired columns
        in the same stored matrix or entity attribute.

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> import numpy as np
        >>> from dsio.data.adapters import entity_examples
        >>> from dsio.data.splits import generate
        >>> path = Path(mkdtemp()) / "store"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     for index in range(4):
        ...         _ = builder.add(f"s{index}", np.array([[float(index)]]), group=f"g{index}")
        >>> store = SignalStore(path)
        >>> examples = entity_examples(store)
        >>> manifest = generate(
        ...     examples, "group_shuffle", name="demo", seed=1,
        ...     parameters={"test_size": 0.25},
        ... )
        >>> fitted = fit_standardization(store, manifest, partition=0)
        >>> role = manifest.required_roles[0]
        >>> sorted(fitted["sample_ids"]) == sorted(manifest.fold(0).assignments[role])
        True
    """
    role, sample_ids = _training_assignment(store, manifest, partition)
    if observed and observed_columns is not None:
        raise ValueError("observed=True and observed_columns are mutually exclusive")
    if isinstance(minimum_scale, bool) or not isinstance(minimum_scale, int | float):
        raise ValueError("minimum_scale must be finite and non-negative")
    minimum_scale = float(minimum_scale)
    if not math.isfinite(minimum_scale) or minimum_scale < 0:
        raise ValueError("minimum_scale must be finite and non-negative")
    if reduction not in {"numpy", "sum_squares"}:
        raise ValueError("reduction must be 'numpy' or 'sum_squares'")

    resolved_ids = sample_ids
    raw: list[np.ndarray] = []
    if attribute is None:
        if examples is not None:
            raise ValueError("examples is only valid with an entity attribute source")
        store_ids = {entity.entity_id for entity in store.entities}
        unknown = sorted(set(sample_ids) - store_ids)
        if unknown:
            raise SplitError(
                f"split training role {role!r} in partition {partition} names unknown sample "
                f"identity {unknown[0]!r}"
            )
        width = store.channels
    else:
        if not attribute:
            raise ValueError("attribute must be a non-empty name")
        resolved_ids = _window_entities(store, examples, sample_ids, role=role)
        for entity_id in resolved_ids:
            attrs = store.entity(entity_id).attrs
            if attribute not in attrs:
                raise ValueError(f"entity {entity_id!r} has no attribute {attribute!r}")
            value = np.asarray(attrs[attribute])
            raw.append(value[None] if value.ndim == 1 else value)
        if any(array.ndim != 2 for array in raw):
            raise ValueError("standardization source values must be rank two [rows, features]")
        widths = {array.shape[1] for array in raw}
        if len(widths) != 1:
            raise ValueError("standardization source values must have one consistent width")
        width = widths.pop()
    selected = _column_indexes(value_columns, width, "value_columns")
    if feature_names is not None and len(feature_names) != len(selected):
        raise ValueError("feature_names must match the selected feature count")
    mask_indexes = (
        None
        if observed_columns is None
        else _column_indexes(observed_columns, width, "observed_columns")
    )
    if mask_indexes is not None and len(mask_indexes) != len(selected):
        raise ValueError(
            "observed_columns must select the same number of features as value_columns"
        )
    if attribute is None:
        raw = [np.asarray(store.read_sample(sample_id)["data"]) for sample_id in sample_ids]
    packed = np.concatenate(raw, axis=0)
    values = np.asarray(packed[:, selected], dtype=np.float64)
    if mask_indexes is not None:
        mask_values = np.asarray(packed[:, mask_indexes])
        if not bool(np.isfinite(mask_values).all()) or not bool(
            ((mask_values == 0) | (mask_values == 1)).all()
        ):
            raise ValueError("observed columns must contain only finite zero or one values")
        validity = mask_values.astype(bool)
    elif observed:
        validity = ~np.isnan(values)
    else:
        validity = np.ones(values.shape, dtype=bool)

    uses_observation = mask_indexes is not None or observed
    if uses_observation:
        counts = validity.sum(axis=0)
        if bool((counts == 0).any()):
            missing_indexes = np.flatnonzero(counts == 0).tolist()
            if feature_names is None:
                missing: list[str | int] = [int(index) for index in missing_indexes]
            else:
                missing = [str(feature_names[index]) for index in missing_indexes]
            raise ValueError(f"features {missing} have no observed value in role {role!r}")
        safe = np.where(validity, values, 0.0)
        if not bool(np.isfinite(safe).all()):
            raise ValueError("observed values must be finite")
        working = safe
        count = counts.astype(np.float64)
    else:
        working = values
        count = np.full(values.shape[1], values.shape[0], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = working.sum(axis=0) / count
        if reduction == "sum_squares":
            variance = np.maximum(
                (np.square(working).sum(axis=0) / count) - mean**2,
                0.0,
            )
            scale = np.sqrt(variance)
        elif uses_observation:
            variance = np.square(np.where(validity, values - mean, 0.0)).sum(axis=0) / count
            scale = np.sqrt(variance)
        else:
            scale = values.std(axis=0)
    scale[scale < minimum_scale if minimum_scale > 0 else scale == 0] = 1.0
    if (uses_observation or bool(np.isfinite(values).all())) and (
        not bool(np.isfinite(mean).all()) or not bool(np.isfinite(scale).all())
    ):
        raise ValueError("fitted mean and scale must be finite; source values overflowed")
    result: dict[str, Any] = {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "role": role,
        "partition": partition,
        "sample_ids": sample_ids,
        "observed": observed,
    }
    records_policy = (
        observed
        or value_columns is not None
        or observed_columns is not None
        or attribute is not None
        or feature_names is not None
        or minimum_scale != 0
        or reduction != "numpy"
    )
    if records_policy:
        result.update(
            {
                "resolved_sample_ids": resolved_ids,
                "source": "data" if attribute is None else "entity_attribute",
                "attribute": attribute,
                "value_columns": selected,
                "observed_columns": mask_indexes,
                "observation": "mask" if mask_indexes is not None else "nan" if observed else "all",
                "minimum_scale": minimum_scale,
                "reduction": reduction,
            }
        )
        if feature_names is not None:
            result["feature_names"] = list(feature_names)
    return result


def fit_class_weights(
    store: SignalStore,
    manifest: SplitFile,
    classes: int,
    attribute: str,
    *,
    partition: int,
    examples: SignalExamples | None = None,
) -> dict[str, Any]:
    """Fit inverse-frequency class weights on the manifest's training role.

    Consumes:
        Zero-based integral labels stored in a sample/entity attribute. Window assignments
        require their bound ``SignalExamples`` and count each owning entity once.

    Produces:
        Native-loss weights ``N / (K * count)`` with counts, governed and resolved
        identities, role, partition, and source policy suitable for fitted evidence.

    Parameters:
        ``partition``; positive ``classes``; non-empty ``attribute``; optional window
        ``examples`` for entity-attribute resolution.

    Devices:
        CPU; fitted once before training.

    Limitations:
        Every declared class must occur in the training role. Labels must be zero-based.

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> from dsio.data.adapters import entity_examples
        >>> from dsio.data.splits import generate
        >>> path = Path(mkdtemp()) / "classes"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     for index in range(8):
        ...         _ = builder.add(
        ...             f"s{index}", np.zeros((1, 1)), group=f"g{index}",
        ...             attrs={"label": index % 4},
        ...         )
        >>> store = SignalStore(path)
        >>> manifest = generate(
        ...     entity_examples(store), "group_shuffle", name="classes", seed=3,
        ...     parameters={"test_size": 0.25},
        ... )
        >>> fit_class_weights(
        ...     store, manifest, partition=0, classes=4, attribute="label"
        ... )["counts"]
        [2, 1, 2, 1]
    """
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 1:
        raise ValueError("classes must be a positive integer")
    if not isinstance(attribute, str) or not attribute:
        raise ValueError("attribute must be a non-empty name")
    role, sample_ids = _training_assignment(store, manifest, partition)
    if examples is None:
        store_ids = {entity.entity_id for entity in store.entities}
        unknown = [sample_id for sample_id in sample_ids if sample_id not in store_ids]
        if unknown:
            raise SplitError(
                f"split training role {role!r} names unknown sample identity {unknown[0]!r}"
            )
        resolved_ids = sample_ids
        source = "sample_attribute"
    else:
        resolved_ids = _window_entities(store, examples, sample_ids, role=role)
        source = "entity_attribute"

    if classes > len(resolved_ids):
        raise ValueError(
            f"training role {role!r} with {len(resolved_ids)} resolved samples cannot "
            f"contain all {classes} classes"
        )
    counts = np.zeros(classes, dtype=np.int64)
    for sample_id in resolved_ids:
        attrs = store.entity(sample_id).attrs
        if attribute not in attrs:
            raise ValueError(f"entity {sample_id!r} has no attribute {attribute!r}")
        label = attrs[attribute]
        if isinstance(label, bool) or not isinstance(label, int | np.integer):
            raise ValueError(
                f"entity {sample_id!r} attribute {attribute!r} must be an integer class"
            )
        label = int(label)
        if label < 0 or label >= classes:
            raise ValueError(
                f"entity {sample_id!r} attribute {attribute!r} class {label} is outside "
                f"[0, {classes})"
            )
        counts[label] += 1
    if bool((counts == 0).any()):
        raise ValueError(
            f"training role {role!r} must contain all {classes} classes; counts={counts.tolist()}"
        )
    weights = counts.sum(dtype=np.int64) / (classes * counts.astype(np.float64))
    if not bool(np.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("fitted class weights must be finite and positive")
    return {
        "classes": classes,
        "counts": counts.tolist(),
        "weights": weights.tolist(),
        "role": role,
        "partition": partition,
        "sample_ids": sample_ids,
        "resolved_sample_ids": resolved_ids,
        "source": source,
        "attribute": attribute,
    }


def fit_group_weights(
    store: SignalStore,
    manifest: SplitFile,
    *,
    partition: int,
    examples: SignalExamples | None = None,
) -> dict[str, Any]:
    """Fit mean-one inverse-group-frequency weights on the training role.

    Consumes:
        Manifest-governed store samples, or a bound ``SignalExamples`` window index.

    Produces:
        Deterministic group counts, group weights, and identity-keyed sample weights using
        ``N / (G * group_count)``; their mean over the training role is one.

    Parameters:
        ``partition`` and optional governed window ``examples``.

    Devices:
        CPU; fitted once before training.

    Limitations:
        Window fits count every assignment and deliberately do not deduplicate entities.

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> from dsio.data.splits import generate
        >>> from dsio.data.views import WindowSpec, build_index
        >>> path = Path(mkdtemp()) / "groups"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     _ = builder.add("a", np.zeros((6, 1)), group="a")
        ...     _ = builder.add("b", np.zeros((4, 1)), group="b")
        >>> store = SignalStore(path)
        >>> examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
        >>> manifest = generate(
        ...     examples, "group_shuffle", name="groups", seed=3,
        ...     parameters={"test_size": 0.5},
        ... )
        >>> fit_group_weights(store, manifest, partition=0, examples=examples)["group_weights"]
        {'a': 1.0}
    """
    role, sample_ids = _training_assignment(store, manifest, partition)
    if examples is None:
        entities = {entity.entity_id: entity for entity in store.entities}
        unknown = [sample_id for sample_id in sample_ids if sample_id not in entities]
        if unknown:
            raise SplitError(
                f"split training role {role!r} names unknown sample identity {unknown[0]!r}"
            )
        groups = [str(entities[sample_id].group) for sample_id in sample_ids]
        source = "sample_group"
    else:
        if examples.name != store.path.name or examples.digest != store.identity:
            raise ValueError(
                "SignalExamples must describe the same store name and content identity"
            )
        assert_index_matches_store(store, examples.index)
        governed_ids = list(examples.sample_ids)
        if len(set(governed_ids)) != len(governed_ids):
            raise ValueError("SignalExamples repeats a governed window identity")
        positions = {sample_id: position for position, sample_id in enumerate(governed_ids)}
        unknown = [sample_id for sample_id in sample_ids if sample_id not in positions]
        if unknown:
            raise SplitError(
                f"split training role {role!r} names unknown window identity {unknown[0]!r}"
            )
        groups = []
        for sample_id in sample_ids:
            position = positions[sample_id]
            code = int(examples.index.entity_codes[position])
            if code < 0 or code >= len(examples.index.entity_names):
                raise ValueError(f"window {sample_id!r} has invalid entity code {code}")
            entity_id = examples.index.entity_names[code]
            entity = store.entity(entity_id)
            indexed_group = examples.index.entity_groups[code]
            if entity.group != indexed_group:
                raise ValueError(
                    f"window {sample_id!r} indexes group {indexed_group!r}, but entity "
                    f"{entity_id!r} belongs to {entity.group!r}"
                )
            groups.append(str(entity.group))
        source = "window_group"
    counts = Counter(groups)
    total = len(groups)
    group_count = len(counts)
    group_weights = {group: total / (group_count * count) for group, count in counts.items()}
    sample_weights = {
        sample_id: group_weights[group] for sample_id, group in zip(sample_ids, groups, strict=True)
    }
    values = np.asarray(list(sample_weights.values()), dtype=np.float64)
    if not bool(np.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError("fitted group weights must be finite and positive")
    if not math.isclose(float(values.mean()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("fitted group weights must have training-role mean one")
    return {
        "counts": dict(counts),
        "group_weights": group_weights,
        "sample_weights": sample_weights,
        "role": role,
        "partition": partition,
        "sample_ids": sample_ids,
        "source": source,
    }


def record_fitted(run_id: str, name: str, fitted: Mapping[str, Any]) -> str:
    """Log fitted values as an MLflow evidence artifact and return its run-relative path.

    The path is the same for every replay, so it can enter execution identity; the
    values themselves live in the artifact on the attempt's own run.

    Consumes:
        An active MLflow run ID and a JSON-serializable fitted mapping, e.g. from
        :func:`fit_standardization`.

    Produces:
        ``fitted/<name>.json`` on the run; returns that path.

    Parameters:
        ``run_id``; ``name`` (a plain file stem); ``fitted``.

    Devices:
        Device-independent.

    Limitations:
        Requires a writable MLflow run.

    Example:
        >>> from mlflow import MlflowClient
        >>> client = MlflowClient()
        >>> run = client.create_run(client.create_experiment("fitted-example"))
        >>> record_fitted(run.info.run_id, "standardization", {"mean": [0.0], "scale": [1.0]})
        'fitted/standardization.json'
    """
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid fitted artifact name {name!r}")
    try:
        json.dumps(dict(fitted), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fitted evidence must be finite canonical JSON: {exc}") from None
    from mlflow import MlflowClient

    path = f"fitted/{name}.json"
    MlflowClient().log_dict(run_id, dict(fitted), path)
    return path


__all__ = ["fit_class_weights", "fit_group_weights", "fit_standardization", "record_fitted"]
