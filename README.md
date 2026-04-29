# lerobot_policy_openvla

**OpenVLA-OFT** 作为 [LeRobot](https://github.com/huggingface/lerobot) Policy 插件的完整实现。

将 [OpenVLA-OFT](https://openvla-oft.github.io/)（Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success）无缝集成进 LeRobot 的训练/评估/部署流水线。

## 核心特性

| 特性 | 说明 |
|------|------|
| **连续动作输出** | MLP action head (L1 回归)，无离散化，更高精度 |
| **Action Chunking** | 单次前向输出 N 步动作，推理速度 25-50x 提升 |
| **多相机输入** | 支持 1-3 路相机（第三人称 + 腕部）|
| **本体感知** | 可选 proprioception 输入 |
| **LoRA 微调** | 通过 `peft` 高效微调，仅需 ~25GB GPU |
| **量化推理** | 4-bit/8-bit 量化，~16GB GPU 可运行推理 |
| **LeRobot 原生** | 完全兼容 `lerobot-train` / `lerobot-eval` 工具链 |

## 安装

```bash
# 1. 安装 lerobot
pip install lerobot

# 2. 安装本插件
git clone <this-repo>
cd lerobot_policy_openvla
pip install -e .

# 可选：flash-attention（推理加速）
pip install flash-attn --no-build-isolation

# 可选：量化支持
pip install bitsandbytes
```

## 快速开始

### 推理（Real Robot）

```python
import torch
from lerobot_policy_openvla import OpenVLAPolicy, OpenVLAConfig
from lerobot_policy_openvla.processor_openvla import make_openvla_pre_post_processors

# 加载微调后的 checkpoint
policy = OpenVLAPolicy.from_pretrained(
    "moojink/openvla-7b-oft-finetuned-libero-spatial"
).eval().cuda()

# 初始化 processor
config = policy.config
pre_proc, post_proc = make_openvla_pre_post_processors(config)

# Episode 开始时重置 action queue
policy.reset()

# 推理循环
for step in range(max_steps):
    obs = robot.get_observation()          # 从机器人获取观测
    processed_obs = pre_proc(obs)          # 预处理
    action = policy.select_action(processed_obs)  # 推理
    action = post_proc(action)             # 反归一化
    robot.send_action(action)              # 执行
```

### 训练（LoRA 微调）

```bash
lerobot-train \
    --policy.type openvla \
    --policy.pretrained_backbone openvla/openvla-7b \
    --policy.use_lora true \
    --policy.lora_rank 32 \
    --policy.lora_alpha 64 \
    --policy.action_dim 7 \
    --policy.action_chunk_size 8 \
    --policy.num_images_in_input 1 \
    --policy.center_crop true \
    --dataset.repo_id your-hf-name/your-lerobot-dataset \
    --batch_size 8 \
    --steps 50000 \
    --output_dir ./outputs/openvla_lora
```

### 双臂任务（ALOHA，14-DOF）

```bash
lerobot-train \
    --policy.type openvla \
    --policy.pretrained_backbone openvla/openvla-7b \
    --policy.use_lora true \
    --policy.action_dim 14 \
    --policy.action_chunk_size 25 \
    --policy.num_images_in_input 3 \
    --policy.use_proprio true \
    --policy.proprio_dim 14 \
    --policy.use_film true \
    --dataset.repo_id your-hf-name/aloha-dataset \
    --batch_size 4
```

### 量化推理（低显存）

```python
# 4-bit 量化，~16GB GPU 可运行
from lerobot_policy_openvla import OpenVLAConfig

config = OpenVLAConfig(
    pretrained_backbone="moojink/openvla-7b-oft-finetuned-libero-spatial",
    load_in_4bit=True,  # 或 load_in_8bit=True
)
policy = OpenVLAPolicy(config).eval().cuda()
```

## 项目结构

```
lerobot_policy_openvla/
├── pyproject.toml
├── README.md
└── src/
    └── lerobot_policy_openvla/
        ├── __init__.py                  # 包入口，暴露公开 API
        ├── configuration_openvla.py     # OpenVLAConfig（所有超参数）
        ├── modeling_openvla.py          # OpenVLAPolicy + 子模块
        └── processor_openvla.py         # 预处理/后处理 pipeline
```

## 模型架构

```
输入
├── 图像 x N (224x224)  →  SigLIP + DINOv2 视觉编码器
├── 语言指令 (str)       →  Tokenizer
└── 本体状态 (可选)      →  ProprioProjector (MLP)
                                 ↓
                    Llama-2 7B LLM 骨干（并行解码）
                                 ↓
                    MLPActionHead (L1 回归)
                                 ↓
输出：连续动作 chunk (action_chunk_size × action_dim)
```

## GPU 显存需求

| 场景 | 配置 | 显存 |
|------|------|------|
| 推理（单相机） | bfloat16 | ~16 GB |
| 推理（双相机+proprio） | bfloat16 | ~16.2 GB |
| LoRA 训练（单相机，batch=1） | bfloat16 | ~25.6 GB |
| LoRA 训练（双相机+proprio，batch=8） | bfloat16 | ~62.5 GB |

## 参考

- 论文：[Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://openvla-oft.github.io/)
- 原始代码：[moojink/openvla-oft](https://github.com/moojink/openvla-oft)
- LeRobot 插件文档：[Bring Your Own Policies](https://huggingface.co/docs/lerobot/en/bring_your_own_policies)
- OpenVLA 预训练权重：[openvla/openvla-7b](https://huggingface.co/openvla/openvla-7b)
