# transnar-mini

简要说明

transnar-mini 是一个轻量级的序列到序列/无自回归（NAR）训练/实验目录，基于本工作区中的模型与训练脚本（例如 `train_transnar.py`、`train_nar.py`）。本项目提供用于快速跑通训练流程的脚本、模型保存与检查点管理目录结构，适合做快速原型、调参与小规模复现实验。

**目录结构（选取重要项）**

- `train_transnar.py`：TransNAR 训练入口脚本。
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
pip install -r ..\SALSA-CLRS\requirements.txt
```

如果你在一个独立环境下运行 `transnar-mini`，也可以在本目录下添加或维护单独的 `requirements.txt`。

**快速开始**

1. 激活虚拟环境并安装依赖（见上）
2. 进入项目目录并运行训练脚本

PowerShell 示例：

```powershell
cd .\transnar-mini
python .\train_transnar.py
# 或者运行 NAR 训练
python .\train_nar.py
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

如需我帮你把 README 翻译成英文、扩展为更详细的使用示例，或者把运行参数自动抽取到文档中，请告诉我具体需求。

---
（本文件根据工作区中 `transnar-mini` 目录的现有脚本与结构生成；如需把示例命令改为精确参数，请允许我读取 `train_transnar.py` 与 `train_nar.py` 的入口参数并更新 README。）
