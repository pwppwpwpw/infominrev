# InfoMinRev (Refactor)

This repository is a full refactor of the original codebase to match the paper **"Towards compact reversible image representations for neural style transfer"**. It provides a clean, config-driven training and inference workflow, and implements the paper’s InfoMinRev architecture and losses (content loss, JSD style loss, Barlow Twins loss, and mutual information minimization).

## What’s Implemented

- **InfoMinRev architecture**: reversible modules with PAC-based coupling, squeeze layers, and AdaIN transfer in latent space.
- **Losses aligned to the paper**:
  - Content loss (Eq. 11)
  - Jensen–Shannon divergence style loss (Eq. 9) with warmup
  - Barlow Twins channel-wise redundancy loss (Eq. 7–8)
  - Mutual information minimization via KDE/NMI (Eq. 3–6)
- **Config-driven training** with deterministic seeding and reproducible settings.
- **Clean package layout** with reusable components and an explicit training engine.

## Repository Structure

- `infominrev/` core package
- `infominrev/models/` InfoMinRev model + VGG encoder
- `infominrev/losses/` content, JSD, Barlow, NMI losses
- `infominrev/engine.py` training engine (Trainer)
- `infominrev/data.py` dataset and dataloaders
- `configs/` JSON configs
- `scripts/` CLI entrypoints
- `legacy/` original unrefactored code snapshot

## Environment Setup

Recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow numpy
```

Notes:
- This repo does not pin exact versions; align with your CUDA + PyTorch environment.
- If you want to reproduce the paper’s experiments, match the environment used in your lab’s existing setup.

## Quick Start (Training)

1. Update dataset paths in `/Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/configs/default.json`:

```json
"data": {
  "content_dir": "/path/to/content",
  "style_dir": "/path/to/style"
}
```

2. Set the VGG weights path in the same config:

```json
"vgg": {
  "weights": "model/vgg_normalised.pth"
}
```

3. Run training:

```bash
python3 /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/scripts/train.py \
  --config /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/configs/default.json
```

Checkpoints and samples will be written to the configured output directory:

```json
"train": {
  "output_dir": "./runs/infominrev"
}
```

## Quick Start (Stylization / Inference)

```bash
python3 /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/scripts/stylize.py \
  --config /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/configs/default.json \
  --checkpoint /path/to/checkpoint.pth \
  --content /path/to/content_dir_or_image \
  --style /path/to/style_dir_or_image \
  --output /path/to/output_dir
```

## Configuration Reference

All training and inference behavior is controlled by JSON config. Key fields:

```json
{
  "seed": 42,
  "device": "cuda",
  "data": {
    "content_dir": "/path/to/content",
    "style_dir": "/path/to/style",
    "image_size": 512,
    "crop_size": 256,
    "batch_size": 4,
    "num_workers": 8,
    "pin_memory": true
  },
  "model": {
    "in_channels": 3,
    "pad_channels": 5,
    "num_blocks": 20,
    "num_squeeze": 2,
    "hidden_channels": 512,
    "blocks_per_stage": [7, 7, 6]
  },
  "loss": {
    "content_weight": 1.0,
    "style_weight": 10.0,
    "barlow_weight": 0.02,
    "mi_weight": 0.1,
    "jsd": {"warmup_steps": 2000, "eps": 1e-8},
    "barlow": {
      "lambda_offdiag": 0.0051,
      "eps": 1e-6,
      "layer": "relu4_1",
      "projector_dims": [512, 256, 128]
    },
    "mi": {"bandwidth": 1.0, "pool": 4, "max_dim": 1024}
  },
  "optim": {
    "lr": 0.0001,
    "betas": [0.9, 0.999],
    "weight_decay": 0.0
  },
  "train": {
    "max_iter": 160000,
    "log_every": 50,
    "save_every": 10000,
    "sample_every": 500,
    "output_dir": "./runs/infominrev",
    "resume": null
  },
  "vgg": {
    "weights": "model/vgg_normalised.pth",
    "layers": ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu4_2"]
  }
}
```

## Architecture Notes

- **InfoMinRev modules** are reversible blocks with PAC-based coupling. The coupling follows Eq. (1–2) of the paper.
- **Squeeze layers** are inserted between stages. `blocks_per_stage` controls how the 20 blocks are distributed across 3 stages.
- **Transfer module** uses AdaIN in latent space, as described in the paper.

## Loss Notes

- **Content loss** uses normalized features (Eq. 11) on `relu4_2` by default.
- **JSD style loss** is computed on mean and variance of feature maps (Eq. 9). A warmup stage uses MSE before switching to JSD.
- **Barlow Twins loss** uses the normalized cross-correlation matrix as in Eq. (7–8). An optional MLP projector is supported.
- **MI minimization** uses KDE-based NMI between consecutive module inputs/outputs (Eq. 3–6).

## Output Artifacts

- `runs/infominrev/checkpoint_*.pth` model checkpoints
- `runs/infominrev/samples/sample_*.jpg` training snapshots
- `runs/infominrev/config.json` resolved config copy

## Legacy Code

The original unrefactored scripts are archived under `legacy/`. If you need to compare behavior, check:

- `legacy/train.py`
- `legacy/Eval.py`
- `legacy/net.py`

## Troubleshooting

- **Slow training or OOM**: reduce `batch_size`, increase `loss.mi.pool`, or reduce `loss.mi.max_dim`.
- **JSD loss unstable early**: increase `loss.jsd.warmup_steps`.
- **Barlow loss too strong**: reduce `loss.barlow_weight` or `barlow.lambda_offdiag`.
- **Dataset loading errors**: ensure your directories contain only image files.

## Reproducibility

- The trainer sets `seed` for Python, NumPy, and Torch.
- Determinism across CUDA kernels is not enforced by default. If you need strict determinism, we can add it.

## Citation

If you use this code, cite the original paper:

```
@inproceedings{infominrev,
  title={Towards compact reversible image representations for neural style transfer},
  author={Liu, Xiyao and Yang, Siyu and Zhang, Jian and Schaefer, Gerald and Li, Jiya and Fan, Xunli and Wu, Songtao and Fang, Hui},
  booktitle={ECCV},
  year={2024}
}
```
