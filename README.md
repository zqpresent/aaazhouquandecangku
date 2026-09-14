# aaazhouquandecangku

Inference-only package for a scene-text visual-style encoder. Given a tightly
cropped text-region image, the encoder describes how the text looks while
trying to suppress the literal text content and surrounding background.

The repository and model names are intentionally opaque. All three artifacts
are public, so opaque names must not be treated as an access-control mechanism.

## What the encoder returns

| Output | Shape per image | Intended meaning |
|---|---:|---|
| `z_global` | `[256]` | Overall text style: typography plus appearance |
| `z_font` | `[128]` | Font/typography-oriented representation |
| `z_appearance` | `[128]` | Fill, stroke, shadow, opacity-oriented representation |
| `style_tokens` | `[8, 384]` | Local query tokens for downstream research |

The three `z_*` vectors are L2-normalized, so cosine similarity is simply their
dot product. `style_tokens` are not normalized and have not yet been validated
as conditioning tokens for a generator.

The released weights include the complete fine-tuned DINOv2 ViT-S/14 backbone.
Loading them does not download separate backbone weights.

## Installation

Python 3.10 is the tested version. Python 3.11 is supported by the package
metadata but has not been used for the release validation.

```bash
git clone https://github.com/zqpresent/aaazhouquandecangku.git
cd aaazhouquandecangku
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The pinned inference stack is:

```text
torch==2.5.0
torchvision==0.20.0
timm==1.0.20
Pillow==10.3.0
numpy==1.24.4
huggingface-hub==0.35.0
safetensors==0.6.2
```

For a particular CUDA build, install the matching PyTorch 2.5.0 wheel using
the official PyTorch selector first, then install this repository. CPU
inference is supported. A GPU is useful for larger batches but is not required.

## Download the public weights

Choose the model for the intended script coverage:

| Public model ID | Scope | Weight SHA-256 |
|---|---|---|
| [`zqpresent/m0a3cebee1aaac92424b0`](https://huggingface.co/zqpresent/m0a3cebee1aaac92424b0) | Original English model | `edb52cbd56667a5f317271b31f329858500bb2ee20e791c631f43321b42fd544` |
| [`zqpresent/m2d8e6c4f1a9b7350e42`](https://huggingface.co/zqpresent/m2d8e6c4f1a9b7350e42) | Chinese-specialized model | `eb9a09231f43dbb69e4399e81eb2c0802e83b3948f241dcd0bb6576bf57e51f2` |
| [`zqpresent/m7c1f9a2e4b8d6035a71`](https://huggingface.co/zqpresent/m7c1f9a2e4b8d6035a71) | Chinese/English model, including cross-language matching | `cd4ae96fa1c3b2d26389c48e83dafac82901189ea90d2b3fee7744b01681bb8a` |

For general Chinese/English use, download the bilingual artifact:

```bash
export OPAQUE_MODEL_ID="zqpresent/m7c1f9a2e4b8d6035a71"
opaque-encoder download "$OPAQUE_MODEL_ID" --local-dir weights
```

The downloader requests only `config.json` and `model.safetensors`, uses
anonymous access, and checks the SHA-256 digest declared in the config.

After downloading, `weights/` is fully self-contained and can be moved to an
offline machine.

## Encode a directory from the command line

Put tightly cropped text images in `crops/`, then run:

```bash
opaque-encoder encode \
  --model weights \
  --input crops \
  --output outputs/styles.npz \
  --device cpu \
  --local-files-only
```

For a GPU, replace `--device cpu` with a device visible to that process, such
as `--device cuda:0`. The command writes:

- `styles.npz`: `z_global`, `z_font`, `z_appearance`, and `style_tokens` arrays;
- `styles.json`: relative input names, output shapes, and the weight digest.

Existing output files are never overwritten. Add `--recursive` to scan input
subdirectories.

## Python API

```python
from pathlib import Path
import torch

from opaque_encoder import StyleEncoder

encoder = StyleEncoder.from_pretrained(
    "weights",                 # or the public Hugging Face model ID
    device="cpu",
    local_files_only=True,
)
vectors = encoder.encode(
    [Path("crops/a.png"), Path("crops/b.png")],
    batch_size=2,
)

# Cosine similarity because z_global is L2-normalized.
score = vectors["z_global"] @ vectors["z_global"].T
print(vectors["z_global"].shape)      # torch.Size([2, 256])
print(vectors["z_font"].shape)        # torch.Size([2, 128])
print(vectors["z_appearance"].shape)  # torch.Size([2, 128])
print(vectors["style_tokens"].shape)  # torch.Size([2, 8, 384])
print(score)
```

`encode` accepts paths, Pillow RGB images, or any iterable containing a mixture
of the two. It streams one batch at a time and returns CPU `float32` tensors.

## Input contract

- Supply one tightly cropped text instance or line per image.
- Arbitrary aspect ratios are accepted.
- Preprocessing preserves aspect ratio, uses LANCZOS resizing, pads with middle
  gray to `112 x 448`, creates a valid-region mask, and applies ImageNet
  normalization.
- Do not stretch images to `112 x 448` before calling the package.
- The model captures visual similarity; it does not recognize or transcribe the
  characters.

## Original English model evidence

The original English checkpoint was selected with one fixed development seed for each
protocol. Its training used counterfactual synthetic groups in which typography,
appearance, content, and background could be varied independently. The final
three stages exposed the model to 4.4 million rendered scene views. Test font
families and test COCO background sources were held out from training.

Three-seed synthetic tests produced the following percentages (mean +/- sample
standard deviation). Each row contains the development seed used during model
selection plus two seeds evaluated after selection; it is therefore not a
three-seed selection-independent estimate.

| Protocol | Global R@1 | Global mAP | Font R@1 | Appearance R@1 | Content mAP (lower) | Background accuracy (lower) |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 77.33 +/- 3.72 | 84.10 +/- 2.81 | 29.65 +/- 1.89 | 44.79 +/- 1.11 | 9.69 +/- 0.46 | 2.14 +/- 0.97 |
| Geometry | 72.20 +/- 2.10 | 80.14 +/- 1.96 | 26.22 +/- 2.59 | 42.25 +/- 2.34 | 9.73 +/- 0.43 | 2.32 +/- 1.09 |
| Hard | 71.88 +/- 3.04 | 80.43 +/- 2.20 | 29.83 +/- 3.57 | 41.44 +/- 3.10 | 9.94 +/- 0.32 | 2.66 +/- 0.84 |

For global and appearance retrieval, an eligible positive had a different text
content and a different background source. Font retrieval also required
different content/style examples, but did not always exclude a shared
background; it should therefore not be presented as an equally strict
cross-background metric. Retrieval results are not classification accuracy and
do not establish performance on real photographs.

## Chinese and bilingual model evidence

Both additions were selected on development data before final held-out tests.
The Chinese-specialized stage used 40,000 groups / 640,000 rendered views. The
bilingual stage used 60,000 groups / 1,920,000 rendered views, including about
25% original-English font replay. On the controlled three-render-seed test, the
following global-style R@1 values were observed:

| Model | Chinese to Chinese | English to English | Chinese to English | English to Chinese |
|---|---:|---:|---:|---:|
| Chinese-specialized | 80.21% | 62.55% | 61.91% | 58.12% |
| Chinese/English | 80.97% | 74.98% | 79.26% | 79.56% |

Use the Chinese-specialized artifact for the strongest observed strict Chinese
font retrieval (`33.48%` R@1 versus `28.74%` for the bilingual artifact). Use
the bilingual artifact when English retention or cross-language style matching
matters. These are synthetic retrieval measurements, not real-scene accuracy.

## Known limitations

- Evaluation is on controlled synthetic data; no real-world paired style test
  set has been used.
- The original artifact is English-first. The two new artifacts cover Chinese
  and/or English only; the Chinese character scope is GB2312 level 1, not all
  Han characters or multilingual shaping.
- Letter-spacing sensitivity is weaker than font, fill, stroke, and shadow.
- Low text/background contrast and visually similar font faces remain difficult.
- Content/background leakage is low under the reported probes, not mathematically
  zero.
- `style_tokens` are an interface for future work, not proof that a downstream
  editor will use them successfully.

## Reproducibility and release safety

The public artifacts contain only inference parameters: the backbone, style
query head, and font/appearance projection heads. Training-only classifiers,
adversaries, optimizer state, scheduler state, data manifests, and local paths
are not included. `safetensors` is used instead of a pickle-based checkpoint.

Each public model was compared numerically with its source checkpoint on four
input aspect ratios. For the original English model, maximum absolute differences were `5.59e-8` for
`z_global`, `1.16e-7` for `z_font`, `7.46e-8` for `z_appearance`, and `2.39e-6`
for `style_tokens`; all outputs were finite and normalized outputs had unit norm.
The two newer model repositories include their own `verification.json` records.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for backbone and library
attribution and license scope. Public read access does not itself grant a
separate license for repository-specific source code.
