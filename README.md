# aaazhouquandecangku

Inference-only package for a scene-text visual-style encoder. Given a tightly
cropped text-region image, the encoder describes how the text looks while
trying to suppress the literal text content and surrounding background.

The repository name is intentionally opaque. The repository and model are
public, so the name must not be treated as an access-control mechanism.

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

Set the model ID once the model repository is published:

```bash
export OPAQUE_MODEL_ID="zqpresent/m0a3cebee1aaac92424b0"
opaque-encoder download "$OPAQUE_MODEL_ID" --local-dir weights
```

The downloader requests only `config.json` and `model.safetensors`, uses
anonymous access, and checks the SHA-256 digest declared in the config. The
published weight digest is:

```text
edb52cbd56667a5f317271b31f329858500bb2ee20e791c631f43321b42fd544
```

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

## Training evidence

The released checkpoint was selected with one fixed development seed for each
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

## Known limitations

- Evaluation is on controlled synthetic data; no real-world paired style test
  set has been used.
- Training and evaluation are English-first and cover 74 sampled characters,
  not multilingual shaping.
- Letter-spacing sensitivity is weaker than font, fill, stroke, and shadow.
- Low text/background contrast and visually similar font faces remain difficult.
- Content/background leakage is low under the reported probes, not mathematically
  zero.
- `style_tokens` are an interface for future work, not proof that a downstream
  editor will use them successfully.

## Reproducibility and release safety

The public artifact contains only inference parameters: the backbone, style
query head, and font/appearance projection heads. Training-only classifiers,
adversaries, optimizer state, scheduler state, data manifests, and local paths
are not included. `safetensors` is used instead of a pickle-based checkpoint.

The public model was compared numerically with the source checkpoint on four
input aspect ratios. Maximum absolute differences were `5.59e-8` for
`z_global`, `1.16e-7` for `z_font`, `7.46e-8` for `z_appearance`, and `2.39e-6`
for `style_tokens`; all outputs were finite and normalized outputs had unit norm.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for backbone and library
attribution and license scope. Public read access does not itself grant a
separate license for repository-specific source code.
