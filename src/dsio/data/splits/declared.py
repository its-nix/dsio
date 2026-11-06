"""Replay of a split that a data source publishes, rather than one DSIO computes.

Some corpora ship a fixed train/test division that every published result uses. Splitting
such a corpus afresh would no longer be the published experiment, and a hand-written
manifest is refused by ``validate``. ``declared`` reads each example's role from a named
per-example attribute, records where the division comes from, and is checked again on
every replay, so a manifest whose assignments drift from the attribute is refused.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from dsio.data.examples import Examples, ExamplesError
from dsio.data.splits.models import SplitError

ALGORITHM = "declared"
_SOURCE_FIELDS = ("citation", "url", "sha256")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalize_parameters(parameters: dict[str, Any], seed: int) -> dict[str, Any]:
    """The canonical ``declared`` parameters, or a ``SplitError`` naming what is wrong."""
    if seed != 0:
        raise SplitError(f"declared splits take no seed; pass seed=0, got {seed!r}")
    unknown = sorted(set(parameters) - {"attribute", "source"})
    if unknown:
        raise SplitError(
            f"declared does not accept parameter {unknown[0]!r}; allowed: attribute, source"
        )
    attribute = parameters.get("attribute")
    if not isinstance(attribute, str) or not attribute:
        raise SplitError("declared requires a non-empty 'attribute' naming each example's role")
    source = parameters.get("source")
    if not isinstance(source, dict):
        raise SplitError("declared requires 'source' with citation, url and sha256")
    extra = sorted(set(source) - set(_SOURCE_FIELDS))
    if extra:
        raise SplitError(f"declared source does not accept field {extra[0]!r}")
    for field in _SOURCE_FIELDS:
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise SplitError(f"declared source requires a non-empty {field!r}")
    if not _SHA256.match(source["sha256"]):
        raise SplitError("declared source 'sha256' must be 64 lowercase hex characters")
    return {
        "attribute": attribute,
        "source": {field: source[field] for field in _SOURCE_FIELDS},
    }


def declared_roles(examples: Examples, attribute: str, roles: tuple[str, ...]) -> np.ndarray:
    """Each example's role, read from ``attribute``; every value must be one of ``roles``."""
    try:
        values = np.asarray(examples.attribute(attribute)).astype(str)
    except ExamplesError as exc:
        raise SplitError(str(exc)) from exc
    unexpected = sorted(set(values.tolist()) - set(roles))
    if unexpected:
        raise SplitError(
            f"declared attribute {attribute!r} has value {unexpected[0]!r}, "
            f"which is not one of the roles {list(roles)}"
        )
    absent = [role for role in roles if not np.any(values == role)]
    if absent:
        raise SplitError(f"declared attribute {attribute!r} assigns no example to {absent[0]!r}")
    return values
