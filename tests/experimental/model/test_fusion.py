"""Multimodal fusion is configuration-only and absence cannot leak."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dsio.config.components import ComponentError, resolve_component
from dsio.experimental.model import MultimodalFusion


def _config(*, enabled: bool = True) -> dict[str, object]:
    return {
        "reference": "dsio.experimental.model.fusion:MultimodalFusion",
        "parameters": {
            "input_shape": [5],
            "order": ["primary", "optional"],
            "modalities": {
                "primary": {
                    "slice": [0, 2],
                    "encoder": {
                        "reference": "torch.nn:Linear",
                        "parameters": {"in_features": 2, "out_features": 2},
                    },
                },
                "optional": {
                    "slice": [2, 4],
                    "presence": 4,
                    "enabled": enabled,
                    "append_presence": True,
                    "encoder": {
                        "reference": "torch.nn:Linear",
                        "parameters": {"in_features": 2, "out_features": 2},
                    },
                },
            },
            "head": {
                "reference": "torch.nn:Linear",
                "parameters": {"in_features": 5, "out_features": 1},
            },
        },
    }


def test_fusion_resolves_and_absent_payload_cannot_reach_encoder_or_head() -> None:
    torch.manual_seed(4)
    model = resolve_component(_config(), expected=MultimodalFusion)
    baseline = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0], [1.0, 2.0, 3.0, 4.0, 1.0]])
    changed = baseline.clone()
    changed[0, 2:4] = torch.tensor([float("nan"), float("inf")])

    result = model(baseline)
    assert result.shape == (2, 1)
    assert torch.equal(result[0], model(changed)[0])
    present_changed = baseline.clone()
    present_changed[1, 2:4] += 10
    assert not torch.equal(result[1], model(present_changed)[1])


def test_absent_payload_isolated_from_batch_dependent_encoder_and_gradients() -> None:
    parameters = dict(_config()["parameters"])  # type: ignore[arg-type]
    modalities = dict(parameters["modalities"])  # type: ignore[arg-type]
    modalities["optional"] = {
        **modalities["optional"],
        "encoder": {
            "reference": "torch.nn:BatchNorm1d",
            "parameters": {"num_features": 2},
        },
    }
    parameters["modalities"] = modalities
    torch.manual_seed(19)
    baseline_model = MultimodalFusion(**parameters)  # type: ignore[arg-type]
    changed_model = MultimodalFusion(**parameters)  # type: ignore[arg-type]
    changed_model.load_state_dict(baseline_model.state_dict())
    baseline = torch.tensor([[1.0, 2.0, 4.0, 5.0, 1.0], [2.0, 3.0, 7.0, 8.0, 0.0]])
    changed = baseline.clone()
    changed[1, 2:4] = torch.tensor([float("nan"), float("inf")])
    assert torch.equal(baseline_model(baseline), changed_model(changed))
    assert torch.equal(
        baseline_model.encoders["optional"].running_mean,
        changed_model.encoders["optional"].running_mean,
    )

    differentiable = baseline.clone().requires_grad_()
    baseline_model(differentiable).sum().backward()
    assert differentiable.grad is not None
    assert differentiable.grad[1, 2:4].eq(0).all()


def test_static_disable_keeps_construction_but_forces_modality_and_presence_to_zero() -> None:
    torch.manual_seed(7)
    enabled = resolve_component(_config(), expected=MultimodalFusion)
    torch.manual_seed(7)
    disabled = resolve_component(_config(enabled=False), expected=MultimodalFusion)
    assert list(enabled.state_dict()) == list(disabled.state_dict())
    for left, right in zip(enabled.parameters(), disabled.parameters(), strict=True):
        assert torch.equal(left, right)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 1.0]])
    changed = x.clone()
    changed[:, 2:5] = torch.tensor([[99.0, -20.0, 0.0]])
    assert torch.equal(disabled(x), disabled(changed))


def test_fusion_supports_declared_singleton_shape_shared_presence_and_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusion(
        input_shape=[1, 3],
        order=["left", "right"],
        modalities={
            "left": {
                "slice": [0, 1],
                "presence": 2,
                "append_presence": True,
                "encoder": {"reference": "torch.nn:Identity"},
            },
            "right": {
                "slice": [1, 2],
                "presence": 2,
                "append_presence": True,
                "encoder": {"reference": "torch.nn:Identity"},
            },
        },
        head={"reference": "torch.nn:Identity"},
    ).to(device)
    result = model(torch.tensor([[[2.0, 3.0, 1.0]]], device=device))
    assert result.device == device
    assert torch.equal(result, torch.tensor([[2.0, 1.0, 3.0, 1.0]], device=device))


def test_fusion_accepts_only_declared_flat_and_compatible_singleton_layouts() -> None:
    parameters = dict(_config()["parameters"])  # type: ignore[arg-type]
    parameters["allow_leading_singleton"] = True
    model = MultimodalFusion(**parameters)  # type: ignore[arg-type]
    flat = torch.tensor([[1.0, 2.0, 3.0, 4.0, 1.0]])

    assert torch.equal(model(flat), model(flat.unsqueeze(1)))
    with pytest.raises(ValueError, match="fusion input"):
        model(flat.unsqueeze(-1))
    with pytest.raises(ValueError, match="requires input_shape"):
        MultimodalFusion(**{**parameters, "input_shape": [1, 5]})  # type: ignore[arg-type]


def test_fusion_routes_encoder_runtime_parameters_and_preserves_collisions() -> None:
    configured = _config()
    model = resolve_component(
        configured,
        encoder_parameters={"primary": {"bias": False}},
        expected=MultimodalFusion,
    )
    assert isinstance(model.encoders["primary"], nn.Linear)
    assert model.encoders["primary"].bias is None
    with pytest.raises(ComponentError, match="collide with runtime arguments"):
        resolve_component(
            configured,
            encoder_parameters={"primary": {"in_features": 2}},
            expected=MultimodalFusion,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"input_shape": []}, "input_shape"),
        ({"modalities": {}}, "modality"),
        ({"modalities": {"bad-name": {}}}, "identifier"),
    ],
)
def test_fusion_rejects_malformed_configuration(change: dict[str, object], message: str) -> None:
    parameters = dict(_config()["parameters"])  # type: ignore[arg-type]
    parameters.update(change)
    with pytest.raises((TypeError, ValueError), match=message):
        MultimodalFusion(**parameters)  # type: ignore[arg-type]


def test_fusion_names_runtime_shape_presence_and_encoder_contract_failures() -> None:
    model = resolve_component(_config(), expected=MultimodalFusion)
    with pytest.raises(ValueError, match=r"input.*\[batch, 5\]"):
        model(torch.ones(2, 1, 5))
    invalid = torch.ones(2, 5)
    invalid[:, 4] = 0.5
    with pytest.raises(RuntimeError, match="optional.*presence.*zero or one"):
        model(invalid)

    broken = MultimodalFusion(
        input_shape=[2],
        order=["broken"],
        modalities={
            "broken": {
                "slice": [0, 2],
                "encoder": {
                    "reference": "torch.nn:Unflatten",
                    "parameters": {"dim": 1, "unflattened_size": [1, 2]},
                },
            }
        },
        head={"reference": "torch.nn:Identity"},
    )
    with pytest.raises(ValueError, match="broken.*rank 2"):
        broken(torch.ones(3, 2))


def test_fusion_rejects_overlap_gaps_presence_in_slice_and_unknown_runtime_target() -> None:
    base = _config()["parameters"]  # type: ignore[index]
    modalities = dict(base["modalities"])  # type: ignore[index]
    modalities["optional"] = {**modalities["optional"], "slice": [1, 4]}
    with pytest.raises(ValueError, match="overlap"):
        MultimodalFusion(**{**base, "modalities": modalities})  # type: ignore[arg-type]
    modalities["optional"] = {**modalities["optional"], "slice": [2, 3]}
    with pytest.raises(ValueError, match="cover"):
        MultimodalFusion(**{**base, "modalities": modalities})  # type: ignore[arg-type]
    modalities["optional"] = {**modalities["optional"], "slice": [2, 5], "presence": 4}
    with pytest.raises(ValueError, match="presence.*slice"):
        MultimodalFusion(**{**base, "modalities": modalities})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown.*missing"):
        MultimodalFusion(**base, encoder_parameters={"missing": {"bias": False}})  # type: ignore[arg-type]
