# CLIPTrace 2026 Challenge – Starter Kit

**后门模型检测与反演挑战赛**官方选手起步仓库。

- 官网：[https://cliptrace-2026.github.io](https://cliptrace-2026.github.io)
- 比赛模型：[RobinWZQ/cliptrace-2026-models](https://huggingface.co/RobinWZQ/cliptrace-2026-models)
- 竞赛平台：[https://www.codabench.org/competitions/17511/](https://www.codabench.org/competitions/17511/)

## 任务

参赛者需要对每个 CLIP 模型完成：

1. 后门检测：输出 `0`（正常）或 `1`（后门）；
2. 目标特征反演：对预测为后门的模型恢复一个 768 维、L2 归一化的目标特征。

官方检查点兼容 Hugging Face `CLIPModel`，基础架构为
`openai/clip-vit-large-patch14-336`：输入大小 336 × 336，patch 大小 14，
投影特征维度 768。

## 仓库结构

```text
cliptrace-challenge/
├── Baseline/
│   ├── main.py                 # DECREE-style 检测与反演
│   ├── imagenet.py             # 数据集与 CLIP 图像预处理
│   ├── utils.py                # 相似度等工具函数
│   ├── trigger/
│   │   └── trigger_clip_l.npz  # baseline 初始化所需触发器文件
│   └── data/                   # 下载后的 baseline 数据
├── submission/
│   ├── code/
│   └── embeddings/
├── download_resources.py       # Hugging Face 下载器
├── create_submission.py        # 校验并生成 submission.zip
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

建议使用 Python 3.10–3.12 和带 CUDA 的 PyTorch。

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
```

### 2. 登录 Hugging Face

模型仓库是私有仓库，数据仓库需要接受访问条款。请先在对应页面取得权限，
然后登录或设置 token：

```bash
hf auth login
# 或：export HF_TOKEN=hf_...
```

不要把 token 写入代码或提交到 Git。

### 3. 下载数据与模型

默认只下载一个模型：

```bash
# 只下载 baseline 数据
python download_resources.py --data-only

# 下载指定模型
python download_resources.py --models-only --model-id model_0002

# 显式下载当前阶段全部模型
python download_resources.py --models-only --all-models
```

下载后的目录与 `Starter Kit` 保持一致：

```text
cliptrace-2026-models/models/development/model_0001/...
Baseline/data/imagenet/{train,val}/...
```

### 4. 运行 baseline

```bash
python Baseline/main.py --model_id model_0001
```

如果本地缺少该模型，`Baseline/main.py` 会自动从 Hugging Face **只下载当前
`model_id`**，然后运行。关闭自动下载：

```bash
python Baseline/main.py --model_id model_0001 --no-auto_download_model
```

也可以使用本地或自行训练的检查点：

```bash
python Baseline/main.py \
  --encoder_path /path/to/model \
  --model_id my_model \
  --imagenet_root /path/to/imagenet
```

快速冒烟测试可减少轮数和数据比例：

```bash
python Baseline/main.py \
  --model_id model_0001 \
  --epochs 1 \
  --sample_ratio 0.01 \
  --batch_size 2
```

默认输出：

```text
submission/predictions.json
submission/embeddings/<model_id>.pt
outputs/baseline-artifacts/<model_id>/...
outputs/baseline_metrics.jsonl
```

每次运行处理一个模型，并增量更新 `predictions.json`。批量运行时，对模型目录
逐个调用上述命令即可。

## Baseline 方法

它优化通用掩码和图像补丁，使不同输入的 CLIP 图像特征趋于相同；满足相似度条件后，以成功触发器中最小的掩码 L1 计算：

```text
PL1 = L1(mask) / (336 × 336 × 3)
label = 1 if PL1 < 0.01 else 0
```

若判定为后门模型，baseline 会对触发输入的投影特征求均值并 L2 归一化，作为
目标特征估计。

如果使用该 baseline，请引用：

> Shiwei Feng, Guanhong Tao, Siyuan Cheng, Guangyu Shen, Xiangzhe Xu, and Zhangyang Wang. *Detecting Backdoors in Pre-trained Encoders*. CVPR 2023.

## 创建提交包

Baseline 生成的 `predictions.json` 格式如下：

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

每个 embedding 必须是位于 CPU 的 `torch.float32` 张量，形状严格为 `[768]`，
元素全部有限且 L2 范数与 1 的误差不超过 `1e-4`。

完成所有模型后运行：

```bash
python create_submission.py
```

脚本会校验提交并生成仓库根目录下的 `submission.zip`。若只想测试格式、尚未
下载完整模型集，可使用：

```bash
python create_submission.py --skip-model-id-check
```

## 开发自己的方法

可以直接修改 `Baseline/`，也可以完全替换 baseline。只要最终按上述格式写入
`submission/`，就能复用官方校验与打包脚本。需要提交复现代码时，请把方法代码
放入 `submission/code/`，并补充环境、命令和硬件说明。

## 联系方式

- 官网：[https://cliptrace-2026.github.io](https://cliptrace-2026.github.io)
- 邮箱：[wangzhongqi23s@ict.ac.cn](mailto:wangzhongqi23s@ict.ac.cn)
- QQ 群：906907183
