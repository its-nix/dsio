"""Dense convolutional signal blocks that preserve every timestep."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU}


class PooledConv1d(nn.Module):
    """Apply a declared strided Conv1d stack and mean-pool over time.

    Consumes:
        A tensor ``[batch, channels, time]``.

    Produces:
        A tensor ``[batch, hidden]`` after the final activation and temporal mean.

    Parameters:
        ``channels``: input channels. ``hidden``: every convolution's output width
        (default 32). ``depth``: number of convolution/activation pairs (default 2).
        ``kernel_size``, ``stride``, and ``padding``: shared Conv1d parameters (defaults
        9, 4, and 4). ``activation``: ``"relu"`` or ``"gelu"`` (default ``"gelu"``).

    Devices:
        CPU and accelerators. Inputs are cast to the convolution parameters' dtype.

    Limitations:
        Uniform width and convolution parameters across stages; no normalization,
        residual, dilation, masked pooling, or learned projection.

    Example:
        >>> import torch
        >>> tuple(PooledConv1d(channels=4)(torch.ones(2, 4, 32)).shape)
        (2, 32)
    """

    def __init__(
        self,
        channels: int,
        hidden: int = 32,
        depth: int = 2,
        kernel_size: int = 9,
        stride: int = 4,
        padding: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.channels = _positive("channels", channels)
        hidden = _positive("hidden", hidden)
        depth = _positive("depth", depth)
        kernel_size = _positive("kernel_size", kernel_size)
        stride = _positive("stride", stride)
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a non-negative integer")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}"
            )
        layers: list[nn.Module] = []
        inputs = self.channels
        for _ in range(depth):
            layers.extend(
                (
                    nn.Conv1d(
                        inputs,
                        hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    _ACTIVATIONS[activation](),
                )
            )
            inputs = hidden
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"channel-first signal must have rank 3, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"channel axis 1 expected {self.channels}, got {x.shape[1]} in shape "
                f"{tuple(x.shape)}"
            )
        if x.shape[2] < 1:
            raise ValueError("time axis 2 must be non-empty")
        time = x.shape[2]
        stage = 0
        for layer in self.body:
            if not isinstance(layer, nn.Conv1d):
                continue
            kernel = layer.dilation[0] * (layer.kernel_size[0] - 1) + 1
            padding = cast(int, layer.padding[0])
            next_time = (time + 2 * padding - kernel) // layer.stride[0] + 1
            if next_time < 1:
                raise ValueError(
                    f"time axis 2 extent {time} is too short for convolution stage {stage}"
                )
            time = next_time
            stage += 1
        dtype = next(self.parameters()).dtype
        values = x if x.dtype == dtype else x.to(dtype)
        result = self.body(values)
        return result.mean(dim=2)


class DenseConv1d(nn.Module):
    """Apply two same-length convolutions without pooling.

    Consumes:
        A tensor ``[batch, channels, time]``. When ``observed_channel`` is set, one
        additional channel contains shared zero-or-one timestep validity.

    Produces:
        A tensor ``[batch, output, time]``; the temporal extent is unchanged.

    Parameters:
        ``channels``: input channels. ``output``: output channels. ``hidden``: width of
        the hidden convolution (default 16). ``kernel_size``: odd convolution kernel
        (default 5), padded symmetrically to preserve time. ``observed_channel``:
        optional index of the structural validity channel; it is not a learned feature.
        ``isolate_padding``: zero hidden and output padding so it cannot affect real
        predictions (default true). False preserves legacy unmasked boundary arithmetic.

    Devices:
        CPU and accelerators. Inputs are cast to the convolution parameters' dtype.

    Limitations:
        Exactly Conv1d -> ReLU -> Conv1d. With validity and isolated padding, hidden and
        output padding are zeroed so real predictions cannot depend on co-batched lengths.
        Disabling isolation is intended only for audited migration parity. There is no
        pooling, residual path, or causal padding.

    Example:
        >>> import torch
        >>> model = DenseConv1d(channels=3, output=2)
        >>> tuple(model(torch.ones(4, 3, 9)).shape)
        (4, 2, 9)
    """

    def __init__(
        self,
        channels: int,
        output: int,
        hidden: int = 16,
        kernel_size: int = 5,
        observed_channel: int | None = None,
        isolate_padding: bool = True,
    ) -> None:
        super().__init__()
        self.channels = _positive("channels", channels)
        hidden = _positive("hidden", hidden)
        output = _positive("output", output)
        if (
            isinstance(kernel_size, bool)
            or not isinstance(kernel_size, int)
            or kernel_size < 1
            or kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be an odd positive integer")
        if observed_channel is not None and (
            isinstance(observed_channel, bool) or not isinstance(observed_channel, int)
        ):
            raise ValueError("observed_channel must be an integer or None")
        if not isinstance(isolate_padding, bool):
            raise ValueError("isolate_padding must be true or false")
        input_channels = self.channels + 1
        if (
            observed_channel is not None
            and not -input_channels <= observed_channel < input_channels
        ):
            raise ValueError(
                f"observed_channel {observed_channel} must identify one of "
                f"{input_channels} input channels"
            )
        self.observed_channel = observed_channel
        self.isolate_padding = isolate_padding
        padding = kernel_size // 2
        self.input = nn.Conv1d(
            self.channels,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.output = nn.Conv1d(hidden, output, kernel_size=kernel_size, padding=padding)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"channel-first signal must have rank 3, got shape {tuple(x.shape)}")
        expected_channels = self.channels + (self.observed_channel is not None)
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"channel axis 1 expected {expected_channels}, got {x.shape[1]} "
                f"in shape {tuple(x.shape)}"
            )
        if x.shape[2] < 1:
            raise ValueError("time axis 2 must be non-empty")
        dtype = next(self.parameters()).dtype
        values = x if x.dtype == dtype else x.to(dtype)
        observed = None
        if self.observed_channel is not None:
            index = self.observed_channel % expected_channels
            validity = values[:, index : index + 1]
            torch._check_tensor_all(
                torch.isfinite(validity) & ((validity == 0) | (validity == 1)),
                lambda: "observed validity channel must contain only zero or one",
            )
            observed = validity.bool()
            values = torch.cat((values[:, :index], values[:, index + 1 :]), dim=1)
        hidden = self.activation(self.input(values))
        if observed is not None and self.isolate_padding:
            hidden = torch.where(observed, hidden, 0.0)
        result = self.output(hidden)
        if observed is None or not self.isolate_padding:
            return result
        return torch.where(observed, result, 0.0)


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["DenseConv1d", "PooledConv1d"]
