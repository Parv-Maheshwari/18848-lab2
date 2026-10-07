"""GRT spectrum convolutional tokenizer."""

from collections.abc import Sequence

import torch
from jaxtyping import Float
from nrdk.roverd import SpectrumData
from torch import Tensor, nn

from nrdk import modules


def _pad_doppler_range(
    x: Float[Tensor, "b c d r"], doppler: int, rng: int
) -> Float[Tensor, "b c d2 r2"]:
    """Pad the Doppler axis circularly, and the range axis with zeros.

    Doppler velocities alias (i.e., `+v_max` wraps around to `-v_max`), so the
    Doppler axis is periodic; range is not.
    """
    if doppler > 0:
        x = nn.functional.pad(x, (0, 0, doppler, doppler), mode="circular")
    if rng > 0:
        x = nn.functional.pad(x, (rng, rng, 0, 0), mode="constant", value=0.0)
    return x


class _ChannelNorm(nn.Module):
    """Layer norm over the channel axis of a channels-first image."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b c h w"]:
        return self.norm(
            x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2)


class DopplerRangeConvNext(nn.Module):
    """ConvNext residual block on a Doppler-range image.

    Identical to [`modules.ConvNextLayer`][nrdk.], except that the depthwise
    convolution uses circular padding along the (periodic) Doppler axis and
    zero padding along the range axis.

    Args:
        channels: number of input/output channels.
        kernel_size: depthwise convolution kernel size; must be odd.
        expansion_ratio: expansion ratio for the inverted bottleneck.
        layer_scale_init_value: initial value for layer scaling; if <= 0,
            layer scaling is disabled.
    """

    def __init__(
        self, channels: int, kernel_size: int = 7,
        expansion_ratio: float = 4.0, layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()

        if kernel_size % 2 != 1:
            raise ValueError(f"Kernel size must be odd: {kernel_size}")

        d = int(channels * expansion_ratio)
        self.pad = kernel_size // 2
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, groups=channels)
        self.norm = _ChannelNorm(channels)
        self.pw1 = nn.Conv2d(channels, d, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(d, channels, kernel_size=1)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(channels)
        ) if layer_scale_init_value > 0 else None

    def forward(self, x: Float[Tensor, "b c d r"]) -> Float[Tensor, "b c d r"]:
        x1 = self.dw(_pad_doppler_range(x, self.pad, self.pad))
        x1 = self.pw2(self.act(self.pw1(self.norm(x1))))
        if self.gamma is not None:
            x1 = self.gamma[None, :, None, None] * x1
        return x + x1


class ConvSpectrumTokenizer(nn.Module):
    """GRT 4D Radar Spectrum tokenizer with a convolutional stem.

    Instead of a single linear projection of each patch (as in
    [`SpectrumTokenizer`][grt.tokenizer.]), the spectrum is first processed by
    a small convolutional network over the Doppler-range image, which lets
    each token aggregate context from neighboring patches:

    1. The elevation and azimuth axes are squeezed into the channel axis.
    2. A `3x3` convolution stem projects the input to `d_conv` channels.
    3. `depth` ConvNext blocks (`7x7` depthwise) extract local features.
    4. A strided convolution with kernel and stride equal to the
       (Doppler, range) patch size projects each patch to `d_model`, yielding
       exactly the same token grid as the baseline tokenizer.
    5. The same sinusoidal positional embedding and readout token are added.

    All convolutions use circular padding along the Doppler axis (which is
    periodic due to velocity aliasing) and zero padding along range.

    !!! warning

        The convolutions operate on each frame independently, so the time
        patch size must be `1`. Additionally, `squeeze` must remove all axes
        other than (time, doppler, range).

    Args:
        d_model: model feature dimension.
        patch: input (time, doppler, elevation, azimuth, range) patch size.
        squeeze: eliminate these axes by moving them to the channel axis prior
            to patching; specified by index.
        n_channels: number of input channels; see [`xwr.nn`][xwr.nn].
        scale: position embedding scale.
        w_min: minimum frequency for sinusoidal position embeddings.
        d_conv: hidden channels of the convolutional stem.
        depth: number of ConvNext blocks.
        kernel_size: depthwise kernel size of the ConvNext blocks.
    """

    def __init__(
        self, d_model: int = 768, patch: Sequence[int] = (1, 2, 2, 8, 4),
        squeeze: Sequence[int] = (2, 3), n_channels: int = 2,
        scale: Sequence[float] | float | None = None,
        w_min: Sequence[float] | float | None = 0.2,
        d_conv: int = 128, depth: int = 2, kernel_size: int = 7,
    ) -> None:
        super().__init__()

        if len(patch) != 5:
            raise ValueError(
                f"Invalid patch size: {patch}; expected 5 dims "
                f"(time, doppler, elevation, azimuth, range)")
        if sorted(squeeze) != [2, 3]:
            raise ValueError(
                f"ConvSpectrumTokenizer requires squeeze=[2, 3] (elevation, "
                f"azimuth); got {squeeze}.")
        if patch[0] != 1:
            raise ValueError(
                f"ConvSpectrumTokenizer requires a time patch size of 1; got "
                f"{patch[0]}.")

        self.squeeze = modules.Squeeze(dim=squeeze, size=patch)
        c_in = n_channels * self.squeeze.n_channels
        self.patch_size = (patch[1], patch[4])

        self.stem = nn.Conv2d(c_in, d_conv, kernel_size=3)
        self.stem_norm = _ChannelNorm(d_conv)
        self.stem_act = nn.GELU()
        self.blocks = nn.Sequential(*[
            DopplerRangeConvNext(d_conv, kernel_size=kernel_size)
            for _ in range(depth)])
        self.proj = nn.Conv2d(
            d_conv, d_model, kernel_size=self.patch_size,
            stride=self.patch_size)

        self.pos = modules.Sinusoid(scale=scale, w_min=w_min)
        self.readout = modules.Readout(d_model=d_model)

    def forward(
        self, spectrum: SpectrumData
    ) -> Float[Tensor, "n s c"]:
        """Apply tokenizer.

        Args:
            spectrum: input batch spectrum data.

        Returns:
            Tokenized output, with the readout token appended last.
        """
        x = self.squeeze(spectrum.spectrum)  # n t d r c
        n, t, d, r, c = x.shape

        # Crop to a multiple of the patch size, matching `PatchMerge`.
        pd, pr = self.patch_size
        x = x[:, :, :d - d % pd, :r - r % pr]

        img = x.reshape(n * t, *x.shape[2:]).permute(0, 3, 1, 2)
        img = self.stem_act(self.stem_norm(
            self.stem(_pad_doppler_range(img, 1, 1))))
        img = self.proj(self.blocks(img))  # (n t) c d2 r2

        tokens = img.permute(0, 2, 3, 1).reshape(
            n, t, img.shape[2], img.shape[3], img.shape[1])
        embedded = self.pos(tokens)
        flat = embedded.reshape(embedded.shape[0], -1, embedded.shape[-1])

        return self.readout(flat)
