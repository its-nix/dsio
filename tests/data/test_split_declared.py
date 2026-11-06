from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsio.data.adapters import TableExamples
from dsio.data.splits import generate, validate
from dsio.data.splits.models import SplitError, SplitFile
from dsio.data.splits.resolve import resolve_masks

SOURCE = {
    "citation": "Author A. A published corpus. Journal 1:1-2, 2020.",
    "url": "https://example.org/corpus",
    "sha256": "0" * 64,
}


def _examples(released: list[str] | None = None, groups: list[str] | None = None) -> TableExamples:
    released = released or ["train", "test", "train", "train", "test", "train"]
    sample_ids = np.asarray([f"window-{index:02d}" for index in range(len(released))])
    return TableExamples(
        name="published",
        sample_ids=sample_ids,
        groups=np.asarray(groups) if groups else sample_ids.copy(),
        attributes={"released": np.asarray(released), "person": np.arange(len(released)) % 3},
        digest="fixed-dataset-identity",
    )


def _declared(examples: TableExamples, **overrides: object) -> SplitFile:
    parameters = {"attribute": "released", "source": dict(SOURCE)}
    parameters.update(overrides)
    return generate(examples, "declared", name="as-released", seed=0, parameters=parameters)


def test_declared_replays_the_published_roles_with_its_source() -> None:
    examples = _examples()

    manifest = _declared(examples)

    assert manifest.algorithm == "declared"
    assert manifest.seed == 0
    assert manifest.dependencies == {}
    assert manifest.coverage == "total"
    assert manifest.validations == (
        "identity_membership",
        "role_disjointness",
        "group_disjointness",
        "total_coverage",
    )
    assert manifest.parameters == {"attribute": "released", "source": SOURCE}
    assert len(manifest.folds) == 1
    fold = manifest.folds[0]
    assert fold.index == 0
    assert fold.assignments == {
        "train": ["window-00", "window-02", "window-03", "window-05"],
        "test": ["window-01", "window-04"],
    }
    validate(examples, manifest)
    masks = resolve_masks(examples, manifest, fold)
    assert examples.sample_ids[masks["test"]].tolist() == ["window-01", "window-04"]


def test_declared_digest_is_stable_and_records_the_source() -> None:
    first = _declared(_examples())
    second = _declared(_examples())
    assert first.digest == second.digest
    other = _declared(_examples(), source={**SOURCE, "sha256": "1" * 64})
    assert other.digest != first.digest


def test_declared_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    examples = _examples()
    manifest = _declared(examples)
    path = tmp_path / "split.yaml"
    path.write_text(manifest.to_yaml())
    loaded = SplitFile.load(path)
    assert loaded.digest == manifest.digest
    validate(examples, loaded)


def test_declared_honours_custom_role_names() -> None:
    examples = _examples(released=["fit", "val", "fit"])
    manifest = generate(
        examples,
        "declared",
        name="fit-val",
        seed=0,
        roles=("fit", "val"),
        parameters={"attribute": "released", "source": dict(SOURCE)},
    )
    assert manifest.folds[0].assignments["val"] == ["window-01"]


def test_declared_refuses_a_seed() -> None:
    with pytest.raises(SplitError, match="take no seed"):
        generate(
            _examples(),
            "declared",
            name="seeded",
            seed=3,
            parameters={"attribute": "released", "source": dict(SOURCE)},
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attribute": ""}, "non-empty 'attribute'"),
        ({"attribute": "missing"}, "missing"),
        ({"source": {**SOURCE, "url": " "}}, "non-empty 'url'"),
        ({"source": {"citation": "c", "url": "u"}}, "non-empty 'sha256'"),
        ({"source": {**SOURCE, "sha256": "ABC"}}, "64 lowercase hex"),
        ({"source": {**SOURCE, "licence": "none"}}, "does not accept field 'licence'"),
        ({"extra": 1}, "does not accept parameter 'extra'"),
    ],
)
def test_declared_refuses_incomplete_parameters(overrides: dict, message: str) -> None:
    with pytest.raises(SplitError, match=message):
        _declared(_examples(), **overrides)


def test_declared_refuses_a_value_that_is_not_a_role() -> None:
    with pytest.raises(SplitError, match="'holdout', which is not one of the roles"):
        _declared(_examples(released=["train", "test", "holdout"]))


def test_declared_refuses_an_empty_role() -> None:
    with pytest.raises(SplitError, match="no example to 'test'"):
        _declared(_examples(released=["train", "train"]))


def test_declared_refuses_a_group_that_straddles_the_published_roles() -> None:
    examples = _examples(released=["train", "test", "train"], groups=["a", "a", "b"])
    with pytest.raises(SplitError, match="leaks group 'a'"):
        _declared(examples)


def test_validate_refuses_assignments_that_drift_from_the_attribute() -> None:
    manifest = _declared(_examples())
    changed = _examples(released=["train", "train", "test", "train", "test", "train"])
    with pytest.raises(SplitError, match="disagree with the declared attribute"):
        validate(changed, manifest)


def test_validate_refuses_an_edited_declared_manifest() -> None:
    examples = _examples()
    manifest = _declared(examples)
    data = manifest.model_dump()
    fold = data["folds"][0]
    fold["assignments"] = {
        "train": ["window-00", "window-01", "window-02", "window-03"],
        "test": ["window-04", "window-05"],
    }
    fold["parts"] = {key: list(value) for key, value in fold["assignments"].items()}
    fold["counts"] = {"train": 4, "test": 2}
    edited = SplitFile.model_validate(data)
    with pytest.raises(SplitError, match="disagree with the declared attribute"):
        validate(examples, edited)


def test_validate_refuses_a_declared_manifest_with_a_seed() -> None:
    examples = _examples()
    data = _declared(examples).model_dump()
    data["seed"] = 7
    with pytest.raises(SplitError, match="take no seed"):
        validate(examples, SplitFile.model_validate(data))
