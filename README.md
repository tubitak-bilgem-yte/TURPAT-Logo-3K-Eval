# Trademark Triplet Benchmark

Benchmark image embedding models on the private Hugging Face dataset
`Bilgem-YTE-AI/turpat-marka-eval`.

The dataset contains triplets:

- `anchor_img`
- `positive_img`
- `negative_img`

The benchmark embeds unique images once per model, computes cosine similarities
for anchor-positive and anchor-negative pairs, and reports accuracy plus
PSR and TSTC metrics for both CLS/global and patch-pooled embeddings.

## Setup

Use an environment with PyTorch, Datasets, Pillow, NumPy, Pandas, and
`transformers>=5.8.0`. DINOv3 support requires the newer Transformers release.

Authentication is read from the local Hugging Face login:

```bash
huggingface-cli whoami
```

## Run

Smoke test:

```bash
python benchmark.py --limit 8 --models facebook/dinov2-large
```

Default benchmark:

```bash
python benchmark.py \
  --dataset Bilgem-YTE-AI/turpat-marka-eval \
  --split train
```

The built-in registry intentionally contains only these 512-resolution models:

- `facebook/dinov2-large`
- `facebook/ijepa_vith16_1k`
- `facebook/dinov3-vith16plus-pretrain-lvd1689m`
- `google/siglip2-so400m-patch16-512`
- `google/siglip2-large-patch16-512`

Useful options:

```bash
python benchmark.py --batch-size 16 --device cuda:0 --force-recompute
```

Set the TSTC temperature:

```bash
python benchmark.py --tstc-temperature 0.5
```

List available CLI options:

```bash
python benchmark.py --help
```

List registered model keys:

```bash
python benchmark.py --list-models
```

## Outputs

Outputs are written under `outputs/` by default:

- `outputs/embeddings/{model}.npz`: cached embeddings.
- `outputs/results/{model}.csv`: per-triplet metrics.
- `outputs/results/summary.csv`: model summary table.
- `outputs/results/summary.json`: model summary records.

Per-triplet rows include:

- image ids for anchor, positive, and negative images.
- positive and negative cosine similarities.
- per-triplet similarity ratio (`PSR`).
- temperature-scaled triplet confidence (`TSTC`).
- correctness flag for each embedding type.

## Metrics

All embeddings are L2-normalized before cosine similarity. For each triplet,
`s_pos = cos(anchor, positive)` and `s_neg = cos(anchor, negative)`. Accuracy is
the fraction of triplets where `s_pos > s_neg`.

The benchmark reports:

- PSR: `s_pos / (s_neg + 1e-6)`.
- TSTC: `sigmoid((s_pos - s_neg) / tau)`, with `tau` controlled by `--tstc-temperature`.
- TSS: `(s_pos - s_neg) / (1 - min(s_pos, s_neg))`.
- Mean summaries for PSR, TSTC, and TSS, plus global PSR: `mean(s_pos) / (mean(s_neg) + 1e-6)`.

## Adding New Models

There are two supported extension paths.

### 1. Add a Hugging Face / Transformers model to the registry

Edit `models/registry.py` and add a `VisionModelSpec` entry to
`MODEL_REGISTRY`.

```python
MODEL_REGISTRY["my-dinov2-384"] = VisionModelSpec(
    key="my-dinov2-384",
    model_id="facebook/dinov2-base",
    model_class="auto",
    preprocess_kwargs={
        "size": {"height": 384, "width": 384},
        "crop_size": {"height": 384, "width": 384},
    },
    cls_strategy="token",
    cls_token_index=0,
    patch_start_index=1,
)
```

Important fields:

- `model_id`: checkpoint loaded with `AutoModel.from_pretrained`.
- `processor_id`: optional separate image processor id.
- `model_class`: use `"auto"` for most vision encoders or `"vision_submodel"` for multimodal models where only `model.vision_model` should run.
- `processor_kwargs`: passed to `AutoImageProcessor.from_pretrained`.
- `preprocess_kwargs`: passed every time images are preprocessed.
- `forward_kwargs`: passed to the model forward call, useful for resized ViT-style models.
- `model_kwargs`: passed to `from_pretrained`, for example `{"add_pooling_layer": False}`.
- `trust_remote_code`: set to `True` only for checkpoints whose code you trust.
- `cls_strategy`: one of `"token"`, `"pooler"`, `"mean_tokens"`, or `"same_as_patch"`.
- `cls_token_index`: token index used when `cls_strategy="token"`.
- `patch_start_index`: first patch token index, or `"after_register_tokens"` for DINOv3-style register tokens.

Only set `forward_kwargs` that the selected model actually supports. For
example, the installed `Dinov2Model` works with resized preprocessing without an
`interpolate_pos_encoding` argument, while ViT and I-JEPA expose that argument.

The generic Transformers embedder supports common output shapes:

- `[batch, tokens, dim]`: `cls` and `patch` are pooled according to the model spec.
- `[batch, channels, height, width]`: both embeddings are global-average-pooled.
- `[batch, dim]`: the same vector is used for both `cls` and `patch`.

Current built-in pooling choices:

- DINOv2: `cls` token 0, `patch` mean from token 1 onward.
- I-JEPA: `cls` uses token mean and `patch` uses all-token mean, matching the local experiment style.
- DINOv3: `cls` token 0, `patch` mean starts after CLS plus `model.config.num_register_tokens`.
- SigLIP2: `cls` uses the vision pooler output, `patch` uses all vision patch tokens.

### 2. Load your own custom inference code at runtime

Use this path for trained local checkpoints, custom preprocessing, non-HF
models, unusual pooling, or architectures that Transformers cannot load yet.

Create a Python file with a factory that returns an object with
`embed_batch(images)`.

```python
import numpy as np


class MyEmbedder:
    def __init__(self, device="cuda"):
        self.device = device
        # Load your checkpoint and build your preprocessing pipeline here.

    def embed_batch(self, images):
        # images is a list of RGB PIL.Image objects.
        # Run any preprocessing you want here.
        batch_size = len(images)
        embeddings = np.ones((batch_size, 768), dtype=np.float32)

        # Return both keys. If your model only has one descriptor, reuse it.
        return {"cls": embeddings, "patch": embeddings}


def create_embedder(device="cuda", cache_dir=None, model_key=None):
    return MyEmbedder(device=device)
```

Run it with:

```bash
python benchmark.py \
  --custom-model my-trained=/path/to/my_model.py:create_embedder \
  --models my-trained \
  --limit 8
```

The custom factory may accept any of these keyword arguments:

- `device`
- `cache_dir`
- `model_key`

Custom output contract:

- Return a dict with `cls` and `patch`.
- Each value must be shaped `[batch, dim]`.
- NumPy arrays and CPU/GPU torch tensors are accepted.
- The benchmark normalizes rows before computing cosine similarity.
- Use `--force-recompute` after changing model code or preprocessing, because embeddings are cached by model key.

See `examples/custom_embedder_template.py` for a runnable template.

## Tests

The test suite uses Python's standard library:

```bash
python -m unittest discover -s tests
```
