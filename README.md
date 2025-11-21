# TRANSNAR
[![中关村学院 GitHub 组织](https://img.shields.io/badge/Linked%20to-bjzgcai%20Org-blue?logo=github)](https://github.com/bjzgcai)

简要说明
参照文章《Transformers meet Neural Algorithmic Reasoners》 文章链接如下：[https://arxiv.org/abs/2406.09308v1](https://arxiv.org/abs/2406.09308v1) 。这里依据文章的思路尝试进行简单的复现与读者理解。

TRANSNAR 是一个轻量级的序列到序列/无自回归（NAR）训练/实验目录，基于本工作区中的模型与训练脚本（例如 `train_transnar.py`、`train_nar.py`）。本项目提供用于快速跑通训练流程的脚本、模型保存与检查点管理目录结构，适合做快速原型、调参与小规模复现实验。

**目录结构（选取重要项）**

- `train_transnar.py`：TRANSNAR 训练入口脚本。
- `train_nar.py`：NAR（非自回归）训练入口脚本。
- `checkpoints/`：训练过程中的模型检查点与权重。
- `data/`、`datasets/`：用于训练/验证/测试的数据目录（若存在）。
- `models/`：模型源码与定义。
- `scripts/`、`utils/`：辅助脚本与工具函数。

**依赖**

- Python 3.8+（建议使用虚拟环境）
- 推荐安装项目顶层 `requirements.txt`（如果仓库根目录有）或 `SALSA-CLRS/requirements.txt`。

示例（PowerShell）:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\TransNAR\requirements.txt
```

如果你在一个独立环境下运行 `TRANSNAR`，也可以在本目录下添加或维护单独的 `requirements.txt`。

**快速开始**

1. 激活虚拟环境并安装依赖（见上）
2. 进入项目目录并运行训练脚本

PowerShell 运行示例：

```powershell
cd .\TransNAR

# 优先运行 NAR 训练
python .\train_nar.py #会在 /data/raw/ 文件夹下生成图数据，然后训练NAR
#并将NAR模型保存到 /checkpoints/nar_bfs.pt

#之后运行
python .\scripts\make_text_dataset.py 
#会在 /data/text/ 文件夹下生成文本数据 bfs_text_train.jsonl

#最后运行 TRANSNAR ，这里NAR模型的参数设置为不变！
python .\train_transnar.py
#模型的参数会保存到  /checkpoints/文件夹下
```

运行脚本通常会在标准输出打印超参数、训练进度以及保存检查点的路径。若脚本接受命令行参数（如 `--config`、`--epochs`、`--batch-size` 等），请参考脚本顶部或源码注释以了解可用参数。

**常见位置**

- 日志/监控：若使用 `tensorboard` 或类似工具，检查脚本输出或 `checkpoints/` 附带的日志路径。
- 检查点：模型和优化器状态通常保存在 `checkpoints/`，可用于恢复训练或评估。

**开发与调试建议**

- 小数据集先跑通一轮以确认配置正确。
- 使用较小 batch 和少量 epoch 做快速回归测试。
- 若出现依赖或路径错误，检查当前 Python 环境与 `PYTHONPATH`（必要时使用 `pip install -e .` 或在运行脚本前将仓库根加入 `sys.path`）。

**贡献与联系方式**

如有任何疑问可联系：
BingzhengYan 
Email：s-ybz25@bjzgca.edu.cn

