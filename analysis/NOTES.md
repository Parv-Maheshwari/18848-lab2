# Lab 2 Notes: Convolutional Tokenizer

## Proposed change

**What.** Replace GRT's linear patch embedding with a convolutional tokenizer
(`grt/conv_tokenizer.py::ConvSpectrumTokenizer`, selected with
`model/tokenizer=conv`). The baseline flattens each 2 (Doppler) x 4 (range)
patch of the 48-channel spectrum (2 elevation x 8 azimuth x {amplitude, sin,
cos}) and applies a single linear layer. The conv tokenizer instead runs:

1. a 3x3 convolution stem (48 -> 128 channels) + LayerNorm + GELU,
2. two ConvNeXt blocks (7x7 depthwise conv, LayerNorm, 4x inverted-bottleneck
   MLP, layer scale),
3. a strided 2x4 convolution (128 -> 512), i.e. the same patch grid as the
   baseline,

over the full-resolution 64 x 256 Doppler-range image of each frame. Doppler
uses circular padding (velocities alias, so the Doppler axis is periodic);
range uses zero padding. The token count (32 x 64 = 2048 + readout),
sinusoidal positional encoding, encoder, decoder, objective and all training
hyperparameters are unchanged.

**Why it could help.**

- *Local context per token.* A linear patch embedding sees only its own 8
  cells. Radar returns are spread across neighboring range-Doppler bins
  (sidelobes, extended targets, Doppler spread from rotating/moving parts), and
  whether a cell is a true peak or a sidelobe depends on its neighbors. With a
  receptive field of ~15 x 15 bins, each token can encode local structure
  (peak vs. sidelobe, local noise floor) before attention runs.
- *Inductive bias.* Convolutions bake in locality and translation
  equivariance, which a transformer otherwise has to learn from data. This
  should matter most at small training set sizes (p10, p20); prior work on
  ViTs ("Early Convolutions Help Transformers See Better", Xiao et al. 2021;
  hybrid ViT, Dosovitskiy et al. 2021) finds conv stems improve optimization
  stability and sample efficiency.
- *Physically-correct boundary handling* along Doppler (circular padding).
- *Cheap.* About +20% FLOPs per sample, dominated by the full-resolution
  ConvNeXt pointwise layers; the transformer is unchanged.

**Implementation.** New component + config, baseline untouched:

- `grt/conv_tokenizer.py`: `ConvSpectrumTokenizer` (same constructor surface as
  `SpectrumTokenizer`, plus `d_conv`, `depth`, `kernel_size`) and
  `DopplerRangeConvNext`, a copy of `nrdk.modules.ConvNextLayer` with mixed
  circular/zero padding. Reuses `nrdk.modules.Squeeze`, `Sinusoid`, `Readout`.
- `config/model/tokenizer/conv.yaml`.
- `tests/test_conv_tokenizer.py`: same output shape as the baseline;
  multi-frame input; remainder cropping; readout token last; receptive field
  crosses patch boundaries (and the baseline's does not); Doppler wraps and
  range does not; gradients reach every parameter; ConvNeXt block initializes
  near identity; invalid arguments are rejected.
- Sweep: `scripts/run_sweep.sh conv model/tokenizer=conv`.

## Conceptual questions

**Why not std(X)/sqrt(n)?** Test samples are consecutive radar frames from a
small number of continuous recordings, so per-sample losses are strongly
autocorrelated in time: neighboring frames see nearly the same scene and have
nearly the same loss. std/sqrt(n) assumes n independent samples. With
positively correlated samples, the actual information content is much smaller.
`nrdk.tss` estimates this as the effective sample size (ESS) from the
autocorrelation. The standard error is then std/sqrt(ESS), which is much
larger. Using sqrt(n) underestimates the standard error by a factor of
sqrt(n/ESS), giving error bars that are far too narrow and overconfident
"significant" results. See the middle panel of `scaling_ci_methods.png`, and
the ESS/n column of the table.

**Why a paired comparison?** Most of the variance in per-sample loss comes from
the sample itself: some scenes are hard for every model, others are easy.
Both models are evaluated on exactly the same frames, so their errors are
highly correlated. Differencing per sample cancels this shared variance:
Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B), which is much smaller than
Var(A) + Var(B) when Cov is large. Computing each method's CI independently
ignores the covariance. The resulting intervals are dominated by
scene-difficulty variance, overlap heavily, and can hide a real, consistent
improvement (right panel of `scaling_ci_methods.png`). The paired test asks
the right question: on the same input, is model A better than model B?

## Results

_TODO: fill in from `results_table.md` and `scaling.png` once runs finish._
