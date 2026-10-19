"""Named multimodal fusion over declared slices of one model input."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from dsio.config.components import resolve_component


class MultimodalFusion(nn.Module):
    """Encode declared input slices, gate optional modalities, and run a fusion head.

    Consumes:
        One tensor ``x`` shaped ``[batch, *input_shape]``. Modality and presence fields
        are positions in its flattened per-sample representation.

    Produces:
        The configured head's tensor output.

    Parameters:
        ``input_shape``: exact non-batch shape (only ``[F]`` and ``[1,F]`` are supported).
        ``allow_leading_singleton``: with ``input_shape=[F]``, also accept the explicitly
        compatible ``[B,1,F]`` form (default false).
        ``modalities``: identifier-keyed mappings with ``slice=[start, stop]``, an
        ``encoder`` ComponentConfig, optional scalar ``presence``, static ``enabled``
        (default true), and ``append_presence`` (default false). Slices are disjoint and,
        with unique presence indices, cover the flattened input. ``order``: every modality
        name exactly once, defining construction and concatenation order independently of
        mapping serialization. ``head``: ComponentConfig consuming concatenated
        embeddings/presences. ``encoder_parameters``: runtime-only parameters routed by
        modality name to nested encoder resolution.

    Devices:
        CPU and accelerators, as configured encoders and head support. Input selection
        preserves dtype/device; individual stages may cast explicitly.

    Limitations:
        Exactly one flat model input and rank-two encoder outputs. Domain placeholder
        validity belongs to ingestion/mapping. This block has no modality attention or
        stochastic modality dropout.

    Example:
        >>> import torch
        >>> model = MultimodalFusion(
        ...     input_shape=[3],
        ...     order=["a", "b"],
        ...     modalities={"a": {"slice": [0, 2], "encoder": {
        ...         "reference": "torch.nn:Linear",
        ...         "parameters": {"in_features": 2, "out_features": 2}}},
        ...         "b": {"slice": [2, 3], "encoder": {"reference": "torch.nn:Identity"}}},
        ...     head={"reference": "torch.nn:Linear",
        ...           "parameters": {"in_features": 3, "out_features": 1}},
        ... )
        >>> tuple(model(torch.ones(4, 3)).shape)
        (4, 1)
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        modalities: Mapping[str, Mapping[str, Any]],
        order: Sequence[str],
        head: Mapping[str, Any],
        encoder_parameters: Mapping[str, Mapping[str, Any]] | None = None,
        allow_leading_singleton: bool = False,
    ) -> None:
        super().__init__()
        shape = tuple(input_shape)
        if len(shape) not in (1, 2) or any(_invalid_extent(value) for value in shape):
            raise ValueError("input_shape must be [features] or [1, features]")
        if len(shape) == 2 and shape[0] != 1:
            raise ValueError("input_shape rank two supports only a leading singleton")
        if not isinstance(allow_leading_singleton, bool):
            raise ValueError("allow_leading_singleton must be true or false")
        if allow_leading_singleton and len(shape) != 1:
            raise ValueError("allow_leading_singleton requires input_shape=[features]")
        if not modalities:
            raise ValueError("modalities must contain at least one modality")
        for name in modalities:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"modality names must be identifiers, got {name!r}")
        if (
            isinstance(order, str | bytes)
            or len(order) != len(modalities)
            or len(set(order)) != len(order)
            or set(order) != set(modalities)
        ):
            raise ValueError("order must contain every modality name exactly once")
        runtime = dict(encoder_parameters or {})
        unknown = sorted(set(runtime) - set(modalities))
        if unknown:
            raise ValueError(f"encoder_parameters contain unknown modalities {unknown}")
        self.input_shape = shape
        self.allow_leading_singleton = allow_leading_singleton
        self.input_features = math.prod(shape)
        self.slices: dict[str, tuple[int, int]] = {}
        self.presence: dict[str, int | None] = {}
        self.enabled: dict[str, bool] = {}
        self.append_presence: dict[str, bool] = {}
        occupied: dict[int, str] = {}
        presence_indices: set[int] = set()
        encoders: dict[str, nn.Module] = {}
        for name in order:
            raw = modalities[name]
            spec = dict(raw)
            extra = sorted(
                set(spec) - {"slice", "encoder", "presence", "enabled", "append_presence"}
            )
            if extra:
                raise ValueError(f"modality {name!r} has unknown parameters {extra}")
            bounds = spec.get("slice")
            if (
                not isinstance(bounds, Sequence)
                or isinstance(bounds, str | bytes)
                or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
            ):
                raise ValueError(f"modality {name!r} slice must be [start, stop]")
            start, stop = bounds
            if start < 0 or stop <= start or stop > self.input_features:
                raise ValueError(
                    f"modality {name!r} slice [{start}, {stop}] must be non-empty and within "
                    f"[0, {self.input_features}]"
                )
            overlap = sorted(set(range(start, stop)) & set(occupied))
            if overlap:
                raise ValueError(
                    f"modality {name!r} slice overlaps {occupied[overlap[0]]!r} at {overlap}"
                )
            for index in range(start, stop):
                occupied[index] = name
            presence = spec.get("presence")
            if presence is not None and (
                isinstance(presence, bool)
                or not isinstance(presence, int)
                or not 0 <= presence < self.input_features
            ):
                raise ValueError(f"modality {name!r} presence must be a valid non-negative index")
            if presence is not None and start <= presence < stop:
                raise ValueError(f"modality {name!r} presence cannot be inside its slice")
            if presence is not None:
                presence_indices.add(presence)
            enabled = spec.get("enabled", True)
            append = spec.get("append_presence", False)
            if not isinstance(enabled, bool) or not isinstance(append, bool):
                raise ValueError(f"modality {name!r} enabled/append_presence must be boolean")
            if append and presence is None:
                raise ValueError(f"modality {name!r} cannot append presence without an index")
            encoder = spec.get("encoder")
            encoders[name] = resolve_component(
                encoder, expected=nn.Module, **dict(runtime.get(name, {}))
            )
            self.slices[name] = (start, stop)
            self.presence[name] = presence
            self.enabled[name] = enabled
            self.append_presence[name] = append
        collisions = sorted(set(occupied) & presence_indices)
        if collisions:
            raise ValueError(f"presence indices cannot occur inside modality slices: {collisions}")
        missing = sorted(set(range(self.input_features)) - set(occupied) - presence_indices)
        if missing:
            raise ValueError(
                f"modality slices and presence indices must cover input; missing {missing}"
            )
        self.encoders = nn.ModuleDict(encoders)
        self.head = resolve_component(head, expected=nn.Module)

    def forward(self, x: Tensor) -> Tensor:
        actual = tuple(x.shape[1:]) if isinstance(x, Tensor) else ()
        compatible = self.allow_leading_singleton and actual == (1, *self.input_shape)
        if not isinstance(x, Tensor) or (actual != self.input_shape and not compatible):
            rendered = ", ".join(str(value) for value in self.input_shape)
            shape = tuple(getattr(x, "shape", ()))
            alternative = f" or [batch, 1, {rendered}]" if self.allow_leading_singleton else ""
            raise ValueError(f"fusion input must be [batch, {rendered}]{alternative}, got {shape}")
        flat = x.reshape(x.shape[0], self.input_features)
        fused: list[Tensor] = []
        for name, encoder in self.encoders.items():
            start, stop = self.slices[name]
            presence_index = self.presence[name]
            if presence_index is None:
                active = torch.ones((len(flat), 1), dtype=torch.bool, device=flat.device)
                presence = None
            else:
                presence = flat[:, presence_index : presence_index + 1]
                torch._check_tensor_all(
                    torch.isfinite(presence) & ((presence == 0) | (presence == 1)),
                    lambda modality=name: (
                        f"modality {modality!r} presence must contain finite zero or one"
                    ),
                )
                active = presence.bool()
            if not self.enabled[name]:
                active = torch.zeros_like(active)
            values = torch.where(active, flat[:, start:stop], 0.0)
            encoded = encoder(values)
            if not isinstance(encoded, Tensor) or encoded.ndim != 2 or len(encoded) != len(flat):
                shape = tuple(getattr(encoded, "shape", ()))
                raise ValueError(
                    f"modality {name!r} encoder must return rank 2 [batch, features], got {shape}"
                )
            fused.append(torch.where(active, encoded, 0.0))
            if self.append_presence[name]:
                assert presence is not None
                fused.append(torch.where(active, presence, 0.0).to(encoded.dtype))
        result = self.head(torch.cat(fused, dim=1))
        if not isinstance(result, Tensor):
            raise ValueError(f"fusion head must return a Tensor, got {type(result).__name__}")
        return result


def _invalid_extent(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 1


__all__ = ["MultimodalFusion"]
