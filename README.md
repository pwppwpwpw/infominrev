# InfoMinRev: Towards Compact Reversible Image Representations for Neural Style Transfer

**Paper** • **Model** • **Dataset** • **Citation**

本仓库是论文 *"Towards Compact Reversible Image Representations for Neural Style Transfer"* 的**规范化重构实现**，对齐论文提出的 InfoMinRev 架构与损失设计，并提供配置化的训练与推理入口。

## News and Updates

- 2026.03.11：完成全量重构，结构与损失对齐论文，提供 config-driven 训练/推理流程。

## Paper Overview

论文从信息论视角解释风格迁移中的过度风格化与纹理不足问题，并提出以下核心设计：

- **InfoMinRev 模块**：可逆流式架构，通过模块间互信息最小化压缩冗余特征。
- **Barlow Twins 通道去相关**：减少通道耦合，增强内容表达能力。
- **Jensen–Shannon 风格分布对齐**：替代传统均值/方差 L2 风格损失，缓解过/欠风格化。

## Installation

推荐 Python 3.8+：

```bash
pip install torch torchvision pillow numpy
```

如需复现论文环境，可根据你的 GPU/PyTorch 版本补充依赖。

## Repository Structure

- `infominrev/`：核心包
- `infominrev/models/`：InfoMinRev + VGG 编码器
- `infominrev/losses/`：Content / JSD / Barlow / NMI-KDE
- `infominrev/engine.py`：训练引擎（Trainer）
- `infominrev/data.py`：数据与 DataLoader
- `scripts/`：训练与推理入口
- `configs/`：JSON 配置
- `legacy/`：原始代码快照（保留以便对照）

## Model Architecture

- **20 个 InfoMinRev 模块 + 2 次 squeeze**（默认分配为 `[7, 7, 6]`）
- **PAC 耦合层 + 可逆变换**对齐论文 Eq.(1)(2)
- **AdaIN** 作为 latent space transfer 模块

如需结构示意图，请参考论文 Fig. 3 和 Fig. 4。

## Training

1. 修改配置：

```json
"data": {
  "content_dir": "/path/to/content",
  "style_dir": "/path/to/style"
},
"vgg": {
  "weights": "model/vgg_normalised.pth"
}
```

2. 启动训练：

```bash
python3 /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/scripts/train.py \
  --config /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/configs/default.json
```

## Inference / Stylization

```bash
python3 /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/scripts/stylize.py \
  --config /Users/yixiao/Desktop/杨思宇-备份/infominrev_v3/configs/default.json \
  --checkpoint /path/to/checkpoint.pth \
  --content /path/to/content_dir_or_image \
  --style /path/to/style_dir_or_image \
  --output /path/to/output_dir
```

## Loss Design (Aligned to Paper)

- **Content Loss**：Eq.(11)，使用 relu4_2 特征并归一化。
- **JSD Style Loss**：Eq.(9)，均值/方差分布 JSD，含 warmup 阶段。
- **Barlow Twins Loss**：Eq.(7)(8)，按相关系数矩阵定义。
- **Mutual Information Loss**：Eq.(3)(6)，KDE + NMI。

## Citation

如果你使用本代码，请引用论文：

```
@inproceedings{infominrev,
  title={Towards Compact Reversible Image Representations for Neural Style Transfer},
  author={Liu, Xiyao and Yang, Siyu and Zhang, Jian and Schaefer, Gerald and Li, Jiya and Fan, Xunli and Wu, Songtao and Fang, Hui},
  booktitle={ECCV},
  year={2024}
}
```

## Acknowledgments

感谢原论文作者及相关开源基线（AdaIN / WCT / ArtFlow 等）为本项目提供理论与实现参考。
