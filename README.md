# CLIPTrace 2026 Challenge – Starter Kit

**后门模型检测与反演挑战赛**官方选手起步仓库。

**官网：** [https://cliptrace-2026.github.io](https://cliptrace-2026.github.io)  
**竞赛平台：** 即将公布  
**Hugging Face：** [https://huggingface.co/cliptrace-2026](https://huggingface.co/cliptrace-2026)

---

## 赛题概览

参赛者会获得一组顺序随机打乱的 CLIP 模型检查点，需要完成两个关联任务：

- **任务一：后门模型检测。** 为每个模型输出 `0`（正常）或 `1`（后门）。
- **任务二：目标特征反演。** 对每个预测为后门的模型，恢复一个 768 维、L2 归一化的目标特征向量。

最终成绩由检测准确率（30%）和目标特征反演得分（70%）组成。完整规则与评分方式请查看[赛事官网](https://cliptrace-2026.github.io/challenge/)。

## 官方提供内容

- 完整的 Hugging Face `CLIPModel` 检查点；
- 用于触发器反演的公开图像数据；
- DECREE-style 检测与目标特征恢复 baseline；
- 提交文件生成与本地格式检查脚本。

模型固定为 `openai/clip-vit-large-patch14-336`，图像输入为 336 × 336，投影特征维度为 768。参赛者拥有模型权重和中间特征的白盒访问权限。

---

## 快速开始

### 1 · 安装依赖

建议使用 Python 3.10–3.12 和带 CUDA 的 PyTorch 环境。

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell

pip install -r requirements.txt
```

### 2 · 下载模型和数据

先完成 `hf auth login`，或设置有权访问竞赛资源的 `HF_TOKEN`：

```bash
python download_resources.py
```

默认下载开发阶段资源。阶段切换方式与参考仓库一致：修改 `download_resources.py` 顶部的 `PHASE`，或运行：

```bash
python download_resources.py --phase final
```

下载后的目录为：

```text
resources/
├── model-repository/models/development/model_0001/...
└── data/imagenet/val/...
```

### 3 · 运行 baseline

```bash
cd detection-recovery-track
bash baseline_decree.sh
```

默认命令会依次处理阶段目录下的全部模型，并把结果直接写入仓库根目录的 `submission/`。也可以先对单个模型做快速冒烟测试：

```bash
MODEL_ID=model_0001 EPOCHS=1 MAX_SAMPLES=12 BATCH_SIZE=2 bash baseline_decree.sh
```

常用参数可以通过环境变量覆盖：

```bash
DEVICE=cuda:0 EPOCHS=100 MAX_SAMPLES=785 BATCH_SIZE=12 bash baseline_decree.sh
MODELS_DIR=/path/to/models DATA_DIR=/path/to/data/imagenet bash baseline_decree.sh
```

Windows 用户可以直接运行等价的 Python 命令：

```powershell
python detection-recovery-track/scripts/baseline_decree.py `
  --models-dir resources/model-repository/models/development `
  --data-dir resources/data/imagenet `
  --submission-dir submission
```

### 4 · 创建提交包

```bash
bash create_submission.sh
```

或在任意系统上运行：

```bash
python create_submission.py
```

脚本会先严格检查 JSON、模型标识、特征形状、数据类型、有限值和 L2 范数，然后生成 `submission.zip`。

---

## Baseline 方法

本仓库提供一个便于修改的 DECREE-style baseline：它为每个模型优化一个通用掩码和图像补丁，使不同输入的 CLIP 图像特征趋于相同；若优化得到的掩码足够稀疏，则将模型判定为后门模型，并将触发输入的平均投影特征作为目标特征估计。

默认判定规则为：

```text
PL1 = L1(mask) / (336 × 336 × 3)
label = 1 if PL1 < 0.10 else 0
```

这只是官方参考方法，不保证特定排行榜成绩。完整运行计算量较大，建议先用单模型、少轮数确认环境和数据路径，再开始正式运行。

如果使用该 baseline，请引用：

> Shiwei Feng, Guanhong Tao, Siyuan Cheng, Guangyu Shen, Xiangzhe Xu, and Zhangyang Wang. *Detecting Backdoors in Pre-trained Encoders*. CVPR 2023.

---

## 提交目录

Baseline 会生成：

```text
submission/
├── predictions.json
├── embeddings/
│   ├── model_0001.pt
│   └── ...
└── code/
    └── README.md
```

`predictions.json` 示例：

```json
{
  "version": "1.0",
  "predictions": [
    {
      "model_id": "model_0001",
      "label": 1,
      "embedding_file": "embeddings/model_0001.pt"
    },
    {
      "model_id": "model_0002",
      "label": 0,
      "embedding_file": null
    }
  ]
}
```

每个 `.pt` 文件必须只包含一个位于 CPU 上的 `torch.float32` 张量，形状严格为 `[768]`，所有元素有限且 L2 范数与 1 的误差不超过 `1e-4`。

## 开发自己的方法

选手可以直接修改 `detection-recovery-track/scripts/baseline_decree.py`，也可以完全替换 baseline。只要最终按上述格式写入 `submission/`，就能复用官方检查和打包脚本。

需要提交复现代码的阶段，请把方法代码放入 `submission/code/`，并补充运行环境、命令和硬件说明。

## 联系方式

- 官网：[https://cliptrace-2026.github.io](https://cliptrace-2026.github.io)
- 邮箱：[wangzhongqi23s@ict.ac.cn](mailto:wangzhongqi23s@ict.ac.cn)
- QQ 群：906907183

祝各位参赛顺利！

