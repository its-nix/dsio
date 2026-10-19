"""Identity-preserving windows over the canonical memory-mapped store.

Legacy experimental (docs/component-admission.md): no real downstream use yet, no
compatibility promise, deleted at 1.0 if still unproven.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dsio.batches import WindowItem
from dsio.data.adapters import SignalExamples
from dsio.data.loading.datasets import LoadingError
from dsio.data.store import SignalStore, StoreError
from dsio.data.views import WindowIndex, assert_index_matches_store
from dsio.experimental.data import items as item_mapping


class WindowItems:
    """Map governed windows into configured identity-bearing training items.

    Consumes:
        A ``SignalStore``, ``SignalExamples`` over a bound ``WindowIndex``, and governed
        window sample identities. Fields read window payloads, entity attributes, index
        labels or metrics. ``x`` may concatenate declared parts.

    Produces:
        CPU items containing ``sample_id``, configured tensor fields and optional ``group``.
        Fitted sample weights are identity-aligned; disjoint held-out assignments are neutral.

    Parameters:
        ``x`` is required; ``y`` and ``mask`` are optional field mappings. Sources are
        ``window``, ``entity_attribute``, ``index_label`` and ``index_metric`` and otherwise
        use ``StoredItems`` dtype/layout/transform/shape rules. ``x`` may instead declare a
        non-empty ``parts`` list, concatenated after flattening. ``sample_weight`` accepts
        ``{"from": "fitted"}`` with runtime ``fitted_sample_weights``. ``group`` emits
        group identity.

    Devices:
        CPU. Lightning moves collated tensors to the accelerator.

    Limitations:
        Fixed-length windows only. Weight fitting is a separate training-role component.

    Example:
        >>> from pathlib import Path
        >>> from tempfile import mkdtemp
        >>> import numpy as np
        >>> from dsio.data.adapters import SignalExamples
        >>> from dsio.data.store import SignalStore
        >>> from dsio.data.views import WindowSpec, build_index
        >>> path = Path(mkdtemp()) / "signals"
        >>> with SignalStore.builder(path, channels=1) as builder:
        ...     _ = builder.add("a", np.arange(4, dtype=np.float32)[:, None], group="g")
        >>> store = SignalStore(path)
        >>> examples = SignalExamples(store, build_index(store, WindowSpec(length=2, stride=2)))
        >>> factory = WindowItems(x={"from": "window"}, group=True)
        >>> item = factory(store, examples, [examples.index.sample_id(0)])[0]
        >>> item["group"], item["x"].tolist()
        ('g', [[0.0], [1.0]])
    """

    def __init__(
        self,
        *,
        x: Mapping[str, Any],
        y: Mapping[str, Any] | None = None,
        mask: Mapping[str, Any] | None = None,
        sample_weight: Mapping[str, Any] | None = None,
        fitted_sample_weights: Mapping[str, float] | None = None,
        group: bool = False,
    ) -> None:
        if not isinstance(group, bool):
            raise ValueError("group must be true or false")
        self.group = group
        self.fields: dict[str, Any] = {"x": _window_output_field("x", x, composite=True)}
        for name, spec in (("y", y), ("mask", mask)):
            if spec is not None:
                self.fields[name] = _window_output_field(name, spec)
        if sample_weight is not None:
            if not isinstance(sample_weight, Mapping) or dict(sample_weight) != {"from": "fitted"}:
                raise ValueError("field 'sample_weight': expected {'from': 'fitted'}")
            self.fields["sample_weight"] = {"source": "fitted"}
            if fitted_sample_weights is None:
                raise ValueError(
                    "field 'sample_weight' requires runtime fitted_sample_weights; "
                    "pass an empty mapping for an explicitly neutral dataset"
                )
        elif fitted_sample_weights is not None:
            raise ValueError("fitted_sample_weights requires a configured sample_weight field")
        self.fitted_sample_weights: dict[str, float] = {}
        for sample_id, raw_weight in (fitted_sample_weights or {}).items():
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("fitted sample-weight identities must be non-empty strings")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, int | float):
                raise ValueError("fitted sample weights must be finite and positive")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("fitted sample weights must be finite and positive")
            float32_min = float(np.nextafter(np.float32(0), np.float32(1)))
            float32_max = float(np.finfo(np.float32).max)
            if weight < float32_min or weight > float32_max:
                raise ValueError("fitted sample weights must remain finite and positive as float32")
            encoded = np.float32(weight)
            if not np.isfinite(encoded) or encoded <= 0:
                raise ValueError("fitted sample weights must remain finite and positive as float32")
            self.fitted_sample_weights[sample_id] = weight

    def __call__(
        self, store: SignalStore, examples: Any, sample_ids: Sequence[str]
    ) -> Dataset[dict[str, Any]]:
        if not isinstance(examples, SignalExamples):
            raise LoadingError("WindowItems requires SignalExamples with a WindowIndex")
        if examples.name != store.path.name or examples.digest != store.identity:
            raise LoadingError(
                "WindowItems requires SignalExamples describing the same store and content "
                f"identity; got {examples.name!r}/{examples.digest!r} for "
                f"{store.path.name!r}/{store.identity!r}"
            )
        try:
            assert_index_matches_store(store, examples.index)
        except ValueError as error:
            raise LoadingError(f"WindowItems index/store mismatch: {error}") from error
        governed = examples.sample_ids.tolist()
        governed_ids = set(governed)
        if len(governed_ids) != len(governed):
            raise LoadingError("WindowItems index repeats a governed window identity")
        unknown_weights = sorted(set(self.fitted_sample_weights) - governed_ids)
        if unknown_weights:
            raise LoadingError(
                f"fitted sample weights name unknown window identity {unknown_weights[0]!r}"
            )
        return _WindowItemsDataset(
            store,
            examples.index,
            sample_ids,
            self.fields,
            fitted_sample_weights=self.fitted_sample_weights,
            emit_group=self.group,
        )


class _WindowItemsDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: SignalStore,
        index: WindowIndex,
        sample_ids: Sequence[str],
        fields: Mapping[str, Any],
        *,
        fitted_sample_weights: Mapping[str, float],
        emit_group: bool,
    ) -> None:
        self.store = store
        self.index = index
        self.sample_ids = tuple(str(value) for value in sample_ids)
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise LoadingError("duplicate assigned window sample identity")
        positions = {index.sample_id(position): position for position in range(len(index))}
        try:
            self.positions = tuple(positions[sample_id] for sample_id in self.sample_ids)
        except KeyError as error:
            raise LoadingError(f"unknown window sample identity {error.args[0]!r}") from None
        self.fields = dict(fields)
        self.emit_group = emit_group
        self.entities = self._resolve_entities(store, index)
        self._validate_parallel_arrays(index)
        self._validate_assigned_windows(index)
        self._verified_entities: set[int] = set()
        self._verification_pid = os.getpid()
        covered = [sample_id in fitted_sample_weights for sample_id in self.sample_ids]
        if any(covered) and not all(covered):
            raise LoadingError("partial fitted sample-weight coverage for assigned windows")
        self._weights = {
            sample_id: fitted_sample_weights[sample_id] if all(covered) else 1.0
            for sample_id in self.sample_ids
        }

    @staticmethod
    def _resolve_entities(store: SignalStore, index: WindowIndex) -> tuple[Any, ...]:
        entities = []
        for code, (name, group) in enumerate(
            zip(index.entity_names, index.entity_groups, strict=True)
        ):
            try:
                entity = store.entity(name)
            except StoreError as error:
                raise LoadingError(
                    f"WindowItems index entity code {code} names unknown store entity {name!r}: "
                    f"{error}"
                ) from error
            if entity.group != group:
                raise LoadingError(
                    f"WindowItems index entity {name!r} has group {group!r}, but the store "
                    f"records {entity.group!r}"
                )
            entities.append(entity)
        return tuple(entities)

    @staticmethod
    def _validate_parallel_arrays(index: WindowIndex) -> None:
        if index.labels is not None and len(index.labels) != len(index):
            raise LoadingError(
                f"WindowItems index labels have {len(index.labels)} values for {len(index)} windows"
            )
        for name, values in index.metrics.items():
            if len(values) != len(index):
                raise LoadingError(
                    f"WindowItems index metric {name!r} has {len(values)} values for "
                    f"{len(index)} windows"
                )

    def _validate_assigned_windows(self, index: WindowIndex) -> None:
        length = index.spec.length
        for position, sample_id in zip(self.positions, self.sample_ids, strict=True):
            code = int(index.entity_codes[position])
            entity = self.entities[code]
            start = int(index.starts[position])
            if start < entity.start_row or start + length > entity.end_row:
                raise LoadingError(
                    f"window {sample_id!r} [{start}, {start + length}) is not owned by indexed "
                    f"entity {entity.entity_id!r} [{entity.start_row}, {entity.end_row})"
                )

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> dict[str, Any]:
        position = self.positions[item]
        sample_id = self.sample_ids[item]
        code = int(self.index.entity_codes[position])
        entity = self.entities[code]
        window = self._read_window(position, code, sample_id)
        sample = {"sample_id": sample_id, "data": window, "attrs": dict(entity.attrs)}
        result: dict[str, Any] = {"sample_id": sample_id}
        for name, spec in self.fields.items():
            if spec.get("source") == "fitted":
                result[name] = torch.tensor(self._weights[sample_id], dtype=torch.float32)
            elif "parts" in spec:
                parts = [
                    torch.from_numpy(self._mapped_value(name, part, sample, position)).reshape(-1)
                    for part in spec["parts"]
                ]
                try:
                    result[name] = torch.cat(parts)
                except RuntimeError as error:
                    raise LoadingError(
                        f"field {name!r} cannot concatenate window {sample_id!r}: {error}"
                    ) from error
            else:
                result[name] = torch.from_numpy(self._mapped_value(name, spec, sample, position))
        if self.emit_group:
            result["group"] = entity.group
        return result

    def _read_window(self, position: int, code: int, sample_id: str) -> np.ndarray[Any, Any]:
        entity = self.entities[code]
        start = int(self.index.starts[position])
        length = self.index.spec.length
        pid = os.getpid()
        if pid != self._verification_pid:
            self._verified_entities.clear()
            self._verification_pid = pid
        try:
            if code not in self._verified_entities:
                payload = self.store.read_sample(entity.entity_id)["data"]
                local_start = start - entity.start_row
                window = payload[local_start : local_start + length]
                self._verified_entities.add(code)
            else:
                window = self.store.read(start, length)
        except Exception as error:
            raise LoadingError(
                f"window {sample_id!r} could not verify/read owning sample "
                f"{entity.entity_id!r}: {error}"
            ) from error
        if len(window) != length:
            raise LoadingError(f"window {sample_id!r} expected {length} rows, got {len(window)}")
        return np.array(window, copy=True, order="C")

    def _mapped_value(
        self,
        name: str,
        spec: Mapping[str, Any],
        sample: Mapping[str, Any],
        position: int,
    ) -> np.ndarray[Any, Any]:
        source = spec.get("_window_source")
        if source == "index_label":
            attrs = (
                {}
                if self.index.labels is None
                else {spec["attribute"]: self.index.labels[position]}
            )
            sample = {**sample, "attrs": attrs}
        elif source == "index_metric":
            metric = spec["_window_metric"]
            values = self.index.metrics.get(metric)
            attrs = {} if values is None else {spec["attribute"]: values[position]}
            sample = {**sample, "attrs": attrs}
        return item_mapping._value(name, spec, sample)


def _window_output_field(
    name: str, spec: Mapping[str, Any], *, composite: bool = False
) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"field {name!r} must be a mapping, got {type(spec).__name__}")
    if "parts" in spec:
        if not composite or set(spec) != {"parts"}:
            raise ValueError(f"field {name!r}: parts is only supported as the complete x spec")
        parts = spec["parts"]
        if not isinstance(parts, Sequence) or isinstance(parts, str | bytes) or not parts:
            raise ValueError("field 'x': parts must be a non-empty sequence")
        return {"parts": [_window_field(name, part) for part in parts]}
    return _window_field(name, spec)


def _window_field(name: str, spec: object) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"field {name!r} must be a mapping, got {type(spec).__name__}")
    source = spec.get("from")
    translated = dict(spec)
    if source == "window":
        translated["from"] = "data"
    elif source == "entity_attribute":
        translated["from"] = "attribute"
    elif source == "index_label":
        if set(translated) & {"attribute", "metric"}:
            raise ValueError(f"field {name!r}: index_label accepts no attribute or metric")
        translated["from"] = "attribute"
        translated["attribute"] = "__index_label__"
    elif source == "index_metric":
        metric = translated.pop("metric", None)
        if not isinstance(metric, str) or not metric:
            raise ValueError(f"field {name!r}: index_metric must name its metric")
        translated["from"] = "attribute"
        translated["attribute"] = f"__index_metric__:{metric}"
    else:
        raise ValueError(f"field {name!r}: unsupported source {source!r}")
    field = item_mapping._field(name, translated)
    field["_window_source"] = source
    if source == "index_metric":
        field["_window_metric"] = metric
    return field


class WindowDataset(Dataset[WindowItem]):
    """Read selected index positions as raw, identity-bearing tensor windows.

    ``positions`` are offsets into ``index``. Each item reports its governed sample ID and
    original row position so batching never destroys identity.

    ``payload_dtype`` is explicit because storage dtype does not determine semantics. An
    int32 store may contain token IDs, which require an integer tensor, or quantised sensor
    values, which require the default float32 tensor. Only the caller knows which.

    Stochastic augmentation does not live here. Every phase receives the same raw sample;
    the Lightning module owns the train-only accelerator lane.
    """

    def __init__(
        self,
        store: SignalStore,
        index: WindowIndex,
        positions: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        *,
        channels_first: bool = True,
        payload_dtype: torch.dtype | None = None,
    ) -> None:
        assert_index_matches_store(store, index)
        self.store = store
        self.index = index
        self.positions = (
            np.arange(len(index), dtype=np.int64)
            if positions is None
            else np.asarray(positions, dtype=np.int64)
        )
        source = index.labels if labels is None else labels
        self.labels = None if source is None else np.asarray(source)
        if self.labels is not None and len(self.labels) != len(index):
            raise ValueError(
                f"labels has {len(self.labels)} entries for {len(index)} windows; they must "
                "be aligned with the whole index, not with this fold"
            )
        self.channels_first = channels_first
        self.payload_dtype = payload_dtype

    def __len__(self) -> int:
        return int(self.positions.size)

    def __getitem__(self, item: int) -> WindowItem:
        position = int(self.positions[item])
        start = int(self.index.starts[position])
        window = self.store.read(start, self.index.spec.length)
        array = np.ascontiguousarray(window.T if self.channels_first else window)
        raw = torch.from_numpy(array)
        payload = raw.float() if self.payload_dtype is None else raw.to(self.payload_dtype)
        result = WindowItem(
            sample_id=self.index.sample_id(position),
            x=payload,
            row=position,
        )
        if self.labels is not None:
            result["y"] = torch.as_tensor(self.labels[position])
        return result

    @property
    def groups(self) -> np.ndarray:
        """Group per item, for verifying a loader never crosses a split boundary."""
        return self.index.groups[self.positions]


__all__ = [
    "WindowDataset",
    "WindowItems",
]
