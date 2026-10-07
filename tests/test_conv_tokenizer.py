"""Tests for the convolutional GRT spectrum tokenizer."""

import pytest
import torch
from nrdk.roverd import SpectrumData

from grt.conv_tokenizer import ConvSpectrumTokenizer, DopplerRangeConvNext
from grt.tokenizer import SpectrumTokenizer

D_MODEL = 32
D_CONV = 8
N_CHANNELS = 3
PATCH = (1, 2, 2, 8, 4)  # (time, doppler, elevation, azimuth, range)
# (time, doppler, elevation, azimuth, range); 16 x 8 token grid.
SHAPE = (1, 32, 2, 8, 32)


def make_spectrum(
    batch: int = 2, shape: tuple[int, ...] = SHAPE
) -> SpectrumData:
    """Create random spectrum data with the given (t, d, el, az, rng) shape."""
    return SpectrumData(
        spectrum=torch.randn(batch, *shape, N_CHANNELS),
        timestamps=torch.zeros(batch, shape[0], dtype=torch.float64),
        range_resolution=torch.full((batch,), 0.1),
        doppler_resolution=torch.full((batch,), 0.05))


def make_tokenizer(**kwargs) -> ConvSpectrumTokenizer:
    """Create a small convolutional tokenizer."""
    return ConvSpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, squeeze=[2, 3],
        n_channels=N_CHANNELS, d_conv=D_CONV, **kwargs).eval()


def token_grid(tokens: torch.Tensor, shape: tuple[int, ...] = SHAPE):
    """Reshape tokens (excluding readout) to a (n, t, d, r, c) grid."""
    t, d, _, _, r = shape
    return tokens[:, :-1].reshape(
        tokens.shape[0], t, d // PATCH[1], r // PATCH[4], -1)


def test_same_shape_as_baseline() -> None:
    """Produces the same token sequence shape as the linear tokenizer."""
    spectrum = make_spectrum(batch=2)
    baseline = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, squeeze=[2, 3], n_channels=N_CHANNELS)

    assert make_tokenizer()(spectrum).shape == baseline(spectrum).shape
    # (1, 32, 32) / (1, 2, 4) = 16 * 8 = 128 patches + readout.
    assert make_tokenizer()(spectrum).shape == (2, 128 + 1, D_MODEL)


def test_history() -> None:
    """Multiple frames are tokenized independently along time."""
    shape = (3, 8, 2, 8, 16)
    tokens = make_tokenizer()(make_spectrum(batch=2, shape=shape))
    assert tokens.shape == (2, 3 * 4 * 4 + 1, D_MODEL)


def test_crops_remainder() -> None:
    """Non-divisible inputs are cropped, like `PatchMerge`."""
    tokens = make_tokenizer()(make_spectrum(shape=(1, 9, 2, 8, 18)))
    assert tokens.shape == (2, 4 * 4 + 1, D_MODEL)


def test_readout_token_is_appended_last() -> None:
    """The last token of every sequence is the (shared) readout parameter."""
    tokenizer = make_tokenizer()
    tokens = tokenizer(make_spectrum(batch=3))
    assert torch.equal(
        tokens[:, -1], tokenizer.readout.readout.expand(3, D_MODEL))


@torch.no_grad()
def test_receptive_field_crosses_patches() -> None:
    """Unlike the linear tokenizer, tokens see neighboring patches."""
    torch.manual_seed(0)
    spectrum = make_spectrum(batch=1)
    perturbed = make_spectrum(batch=1)
    perturbed.spectrum[:] = spectrum.spectrum
    # Last bin of the token at (doppler=7, range=3).
    perturbed.spectrum[:, :, 15, :, :, 15] += 10.0

    conv = make_tokenizer()
    linear = SpectrumTokenizer(
        d_model=D_MODEL, patch=PATCH, squeeze=[2, 3],
        n_channels=N_CHANNELS).eval()

    for tokenizer, crosses in [(conv, True), (linear, False)]:
        a = token_grid(tokenizer(spectrum))
        b = token_grid(tokenizer(perturbed))
        assert not torch.allclose(a[:, :, 7, 3], b[:, :, 7, 3])
        # Neighbor at (doppler=8, range=4).
        assert (not torch.allclose(a[:, :, 8, 4], b[:, :, 8, 4])) == crosses


@torch.no_grad()
def test_doppler_is_circular_range_is_not() -> None:
    """Doppler wraps around; range does not."""
    torch.manual_seed(0)
    tokenizer = make_tokenizer()
    spectrum = make_spectrum(batch=1)

    # Doppler bin 0 -> wraps to the last Doppler token, but not the middle.
    perturbed = make_spectrum(batch=1)
    perturbed.spectrum[:] = spectrum.spectrum
    perturbed.spectrum[:, :, 0, :, :, 16] += 10.0
    a = token_grid(tokenizer(spectrum))
    b = token_grid(tokenizer(perturbed))
    assert not torch.allclose(a[:, :, -1, 4], b[:, :, -1, 4])
    assert torch.allclose(a[:, :, 8, 4], b[:, :, 8, 4])

    # Range bin 0 -> does not wrap to the last range token.
    perturbed.spectrum[:] = spectrum.spectrum
    perturbed.spectrum[:, :, 16, :, :, 0] += 10.0
    b = token_grid(tokenizer(perturbed))
    assert not torch.allclose(a[:, :, 8, 0], b[:, :, 8, 0])
    assert torch.allclose(a[:, :, 8, -1], b[:, :, 8, -1])


def test_gradients_reach_all_parameters() -> None:
    """Every parameter receives a gradient."""
    tokenizer = make_tokenizer().train()
    tokenizer(make_spectrum()).square().mean().backward()
    for name, p in tokenizer.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_convnext_block_preserves_shape() -> None:
    """ConvNext block is shape-preserving and starts close to identity."""
    block = DopplerRangeConvNext(D_CONV)
    x = torch.randn(2, D_CONV, 6, 10)
    y = block(x)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=1e-3)


@pytest.mark.parametrize("kwargs", [
    {"patch": (1, 2, 2, 8)},
    {"patch": (2, 2, 2, 8, 4)},
    {"squeeze": [2]},
])
def test_invalid_arguments(kwargs: dict) -> None:
    """Invalid configurations are rejected."""
    args = {
        "d_model": D_MODEL, "patch": PATCH, "squeeze": [2, 3],
        "n_channels": N_CHANNELS, **kwargs}
    with pytest.raises(ValueError):
        ConvSpectrumTokenizer(**args)


def test_invalid_kernel_size() -> None:
    """Even depthwise kernels are rejected."""
    with pytest.raises(ValueError):
        DopplerRangeConvNext(D_CONV, kernel_size=4)
