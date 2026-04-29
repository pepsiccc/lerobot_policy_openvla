"""
OpenVLA-OFT Policy — LeRobot 集成实现。

架构概览（OpenVLA-OFT）：
┌─────────────────────────────────────────────────────────────────────┐
│  输入                                                                 │
│  ├── 图像 x N (224x224, RGB)   ─→  SigLIP + DINOv2 视觉编码器        │
│  ├── 语言指令 (str)             ─→  Tokenizer                         │
│  └── 本体状态 (可选)            ─→  ProprioProjector (MLP)            │
│                                         ↓                             │
│                              Llama-2 7B LLM 骨干                     │
│                         (并行解码，非自回归生成)                       │
│                                         ↓                             │
│                           MLPActionHead (L1 回归)                     │
│                                         ↓                             │
│  输出：连续动作 chunk  (action_chunk_size × action_dim)                │
└─────────────────────────────────────────────────────────────────────┘

关键设计决策：
1. 使用 HuggingFace transformers AutoModelForVision2Seq 加载模型，
   无需依赖原版 prismatic 包，便于集成。
2. 并行解码：一次前向传播输出完整 action chunk（非自回归），实现高速推理。
3. LeRobot 接口：实现 forward()（训练）和 select_action()（推理）。
4. LoRA：通过 peft 库无缝注入，训练时启用，推理时可合并权重。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.nn as nn
# Prismatic VLM（OpenVLA 的 backbone）使用自定义架构，在 config.json 的 auto_map 中
# 注册的是 "AutoModelForVision2Seq"。transformers 5.x 虽然将该类改名为
# AutoModelForImageTextToText，但 trust_remote_code=True 的加载路径走的是
# auto_map，会直接实例化 modeling_prismatic.py 里的类，绕过 Auto 注册表。
# 因此这里统一使用 AutoConfig + 手动加载，兼容 4.x 和 5.x。
from transformers import AutoConfig, AutoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_openvla import OpenVLAConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 子模块：MLP Action Head
# ─────────────────────────────────────────────────────────────────────────────

class MLPActionHead(nn.Module):
    """将 LLM 最后一层隐藏状态映射到连续动作 chunk。

    OpenVLA-OFT 的核心创新之一：用 MLP 回归头替代 token 生成，
    实现并行解码，大幅降低推理延迟。

    输入：LLM 最后 action_chunk_size 个 token 的隐藏状态
          形状：(batch, action_chunk_size, llm_dim)
    输出：连续动作 chunk
          形状：(batch, action_chunk_size, action_dim)
    """

    def __init__(
        self,
        llm_dim: int,
        action_dim: int,
        action_chunk_size: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size

        # 构建 MLP：llm_dim → hidden_dim × N → action_dim * action_chunk_size
        layers: list[nn.Module] = []
        in_dim = llm_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        # 输出层：预测 action_chunk_size 步动作（展平后预测，然后 reshape）
        layers.append(nn.Linear(in_dim, action_dim * action_chunk_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: LLM 输出的隐藏状态。
                并行解码时取最后 action_chunk_size 个位置的均值，
                形状 (batch, llm_dim)；
                或直接传入 (batch, action_chunk_size, llm_dim) 逐步预测。

        Returns:
            actions: (batch, action_chunk_size, action_dim)
        """
        # 若输入是 3D (batch, seq, dim)，取最后 action_chunk_size 步的均值
        # 以获得全局 context 表示
        if hidden_states.dim() == 3:
            # 取最后 action_chunk_size 个位置的均值作为 action context
            hidden_states = hidden_states[:, -self.action_chunk_size:, :].mean(dim=1)
        # hidden_states: (batch, llm_dim)
        out = self.mlp(hidden_states)  # (batch, action_dim * action_chunk_size)
        return out.view(-1, self.action_chunk_size, self.action_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 子模块：Proprio Projector
# ─────────────────────────────────────────────────────────────────────────────

class ProprioProjector(nn.Module):
    """将本体感知状态投影到 LLM 输入空间，作为软提示 (soft prompt) 注入。

    与 OpenVLA-OFT 原版实现对齐：
    proprio → LayerNorm → Linear → GELU → Linear → LLM embedding 空间
    """

    def __init__(self, proprio_dim: int, llm_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (batch, proprio_dim)
        Returns:
            tokens: (batch, 1, llm_dim) — 作为 1 个额外 token 注入
        """
        return self.projector(proprio).unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# 主 Policy 类
# ─────────────────────────────────────────────────────────────────────────────

class OpenVLAPolicy(PreTrainedPolicy):
    """OpenVLA-OFT Policy，符合 LeRobot PreTrainedPolicy 接口。

    使用示例（推理）：
        policy = OpenVLAPolicy.from_pretrained("your-hf-repo/openvla-oft")
        # observation 格式见 select_action() 的 docstring
        action = policy.select_action(observation)

    使用示例（训练）：
        lerobot-train \\
            --policy.type openvla \\
            --policy.pretrained_backbone openvla/openvla-7b \\
            --policy.use_lora true \\
            --dataset.repo_id your/dataset
    """

    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(
        self,
        config: OpenVLAConfig,
        dataset_stats: dict[str, Any] | None = None,
    ):
        super().__init__(config, dataset_stats)
        self.config: OpenVLAConfig = config

        # ── 加载 Prismatic/OpenVLA backbone ──────────────────────────────────
        logger.info(f"加载 OpenVLA backbone：{config.pretrained_backbone}")
        torch_dtype = getattr(torch, config.torch_dtype)

        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=config.load_in_4bit,
                    load_in_8bit=config.load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                raise ImportError(
                    "量化需要安装 bitsandbytes：pip install bitsandbytes"
                )

        # Prismatic/OpenVLA 使用 trust_remote_code 加载自定义模型类。
        # auto_map 中注册的是 "AutoModelForVision2Seq"，但 transformers 5.x
        # 已删除该名称。解决方案：先用 AutoConfig 读取配置，再用配置类上的
        # from_pretrained 直接加载，完全绕过 Auto 注册表的名称查找。
        model_config = AutoConfig.from_pretrained(
            config.pretrained_backbone,
            trust_remote_code=True,
        )
        model_cls = model_config.__class__  # OpenVLAConfig (prismatic)
        # 从 auto_map 找到模型类并动态加载
        auto_map = getattr(model_config, "auto_map", {})
        model_cls_path = (
            auto_map.get("AutoModelForVision2Seq")
            or auto_map.get("AutoModelForImageTextToText")
        )
        if model_cls_path:
            # model_cls_path 格式："modeling_prismatic.OpenVLAForActionPrediction"
            module_name, cls_name = model_cls_path.split(".")
            import importlib, sys, os

            # 将模型目录加入 sys.path，使 modeling_prismatic.py 可被直接 import
            # 无论是本地路径还是 HuggingFace 缓存路径，这种方式都适用
            model_dir = str(config.pretrained_backbone)
            if model_dir not in sys.path:
                sys.path.insert(0, model_dir)

            # 如果已经加载过（import 缓存），先清除避免使用旧版本
            if module_name in sys.modules:
                del sys.modules[module_name]

            mod = importlib.import_module(module_name)
            ModelClass = getattr(mod, cls_name)
        else:
            raise ValueError(
                f"无法从 config.json 的 auto_map 找到模型类。"
                f"auto_map={auto_map}"
            )

        load_kwargs = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # transformers 5.x 用 dtype，4.x 用 torch_dtype
        try:
            import transformers
            from packaging.version import Version
            if Version(transformers.__version__) >= Version("5.0.0"):
                load_kwargs["dtype"] = torch_dtype
            else:
                load_kwargs["torch_dtype"] = torch_dtype
        except Exception:
            load_kwargs["torch_dtype"] = torch_dtype

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.vla = ModelClass.from_pretrained(
            config.pretrained_backbone,
            **load_kwargs,
        )

        # 获取 LLM 隐藏维度（用于初始化 action head 和 proprio projector）
        self.llm_dim: int = self._get_llm_dim()

        # ── 可选：注入 LoRA ────────────────────────────────────────────────
        if config.use_lora:
            self._inject_lora()

        # ── MLP Action Head ───────────────────────────────────────────────
        self.action_head = MLPActionHead(
            llm_dim=self.llm_dim,
            action_dim=config.action_dim,
            action_chunk_size=config.action_chunk_size,
            hidden_dim=config.action_head_hidden_dim,
            num_hidden_layers=config.action_head_num_hidden_layers,
        )

        # ── Proprio Projector（可选）──────────────────────────────────────
        self.proprio_projector: ProprioProjector | None = None
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(
                proprio_dim=config.proprio_dim,
                llm_dim=self.llm_dim,
                hidden_dim=config.proprio_projector_hidden_dim,
            )

        # ── 推理用 action chunk 队列 ──────────────────────────────────────
        # 实现 action chunking：一次预测多步，按需从队列中取
        self._action_queue: deque[torch.Tensor] = deque()

        logger.info(
            f"OpenVLAPolicy 初始化完成。"
            f"LLM dim={self.llm_dim}, action_dim={config.action_dim}, "
            f"chunk_size={config.action_chunk_size}, "
            f"use_lora={config.use_lora}, use_proprio={config.use_proprio}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 内部辅助方法
    # ─────────────────────────────────────────────────────────────────────────

    def _get_llm_dim(self) -> int:
        """从 backbone 中提取 LLM 隐藏层维度。"""
        # OpenVLA 使用 Llama-2 7B，hidden_size=4096
        # 尝试多种常见属性路径
        for attr_path in [
            "language_model.config.hidden_size",
            "llm.config.hidden_size",
            "config.text_config.hidden_size",
            "config.hidden_size",
        ]:
            obj = self.vla
            try:
                for attr in attr_path.split("."):
                    obj = getattr(obj, attr)
                return int(obj)
            except AttributeError:
                continue
        # 兜底：Llama-2 7B 的标准维度
        logger.warning("无法自动检测 LLM hidden_size，使用默认值 4096。")
        return 4096

    def _inject_lora(self) -> None:
        """向 VLA backbone 注入 LoRA 适配器。"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("LoRA 需要安装 peft：pip install peft")

        cfg = self.config
        target_modules = cfg.lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.vla = get_peft_model(self.vla, lora_config)
        self.vla.print_trainable_parameters()

    def _build_prompt(
        self,
        task_description: str,
        num_images: int,
    ) -> str:
        """构造 OpenVLA 标准提示词格式。

        OpenVLA 使用固定的提示词模板：
        "In: What action should the robot take to {task}?\nOut:"
        多图像时在提示词前插入 <image> token（processor 处理）。
        """
        return f"In: What action should the robot take to {task_description}?\nOut:"

    def _get_hidden_states(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        proprio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """一次前向传播，提取 LLM 最后一层隐藏状态（并行解码）。

        Args:
            pixel_values: 预处理后的图像 (batch, C, H, W) 或多图像
            input_ids: tokenized 提示词 (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)
            proprio_tokens: 本体感知 token (batch, 1, llm_dim) [可选]

        Returns:
            hidden_states: LLM 最后一层输出 (batch, seq_len, llm_dim)
        """
        # 利用 output_hidden_states=True 获取所有层的隐藏状态
        outputs = self.vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        # 取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, llm_dim)

        # 若有 proprio token，在序列末尾追加并重新过最后一个 transformer block
        # 简化实现：直接将 proprio embedding 加到最后 token 的 hidden state 上
        if proprio_tokens is not None:
            # proprio_tokens: (batch, 1, llm_dim)
            # 加到最后一个 token 位置的 hidden state
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + proprio_tokens

        return hidden_states

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 标准接口：forward()（训练）
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """训练前向传播。

        Args:
            batch: LeRobot DataLoader 输出的批次数据，包含：
                - "observation.images.<cam_name>": (B, C, H, W) 图像张量
                - "observation.state": (B, proprio_dim) 本体状态 [可选]
                - "input_ids": (B, seq_len) tokenized 输入
                - "attention_mask": (B, seq_len) 注意力掩码
                - "action": (B, action_chunk_size, action_dim) 目标动作

        Returns:
            (loss, info)：符合 PreTrainedPolicy 抽象接口规范。
        """
        # ── 收集图像 ──────────────────────────────────────────────────────
        pixel_values = self._collect_pixel_values(batch)

        # ── 本体感知 token ─────────────────────────────────────────────────
        proprio_tokens = None
        if self.config.use_proprio and "observation.state" in batch:
            proprio = batch["observation.state"]
            proprio_tokens = self.proprio_projector(proprio)  # (B, 1, llm_dim)

        # ── LLM 前向 ──────────────────────────────────────────────────────
        hidden_states = self._get_hidden_states(
            pixel_values=pixel_values,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            proprio_tokens=proprio_tokens,
        )

        # ── Action Head 预测 ───────────────────────────────────────────────
        pred_actions = self.action_head(hidden_states)
        # pred_actions: (B, action_chunk_size, action_dim)

        # ── 计算损失 ──────────────────────────────────────────────────────
        target_actions = batch["action"]  # (B, action_chunk_size, action_dim)

        if self.config.use_l1_regression:
            loss = nn.functional.l1_loss(pred_actions, target_actions)
        else:
            loss = nn.functional.mse_loss(pred_actions, target_actions)

        info = {"loss_action": loss.item()}
        return loss, info

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot 标准接口：select_action()（推理）
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """单步推理，返回下一个要执行的动作。

        实现 Action Chunking：队列为空时触发一次前向传播，预测整个 chunk；
        然后逐步从队列中取出动作执行，直到队列再次为空。

        Args:
            observation: 来自机器人/仿真器的当前观测，包含：
                - "observation.images.<cam_name>": (1, C, H, W) 图像
                - "observation.state": (1, proprio_dim) 本体状态 [可选]
                - "input_ids": (1, seq_len) tokenized 提示词
                - "attention_mask": (1, seq_len) 注意力掩码
                （上述 tokenized 输入由 processor 在 pre_process 阶段生成）

        Returns:
            action: (action_dim,) 当前步要执行的连续动作
        """
        self.eval()

        # 若 action queue 已空，重新推理填充
        if len(self._action_queue) == 0:
            chunk = self._predict_action_chunk(observation)
            # chunk: (1, action_chunk_size, action_dim) → 逐步入队
            for step in range(self.config.num_open_loop_steps):
                self._action_queue.append(chunk[0, step])  # (action_dim,)

        return self._action_queue.popleft()

    def _predict_action_chunk(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """执行一次完整的前向传播，返回 action chunk。

        Returns:
            chunk: (1, action_chunk_size, action_dim)
        """
        pixel_values = self._collect_pixel_values(observation)

        proprio_tokens = None
        if self.config.use_proprio and "observation.state" in observation:
            proprio = observation["observation.state"]
            proprio_tokens = self.proprio_projector(proprio)

        hidden_states = self._get_hidden_states(
            pixel_values=pixel_values,
            input_ids=observation["input_ids"],
            attention_mask=observation["attention_mask"],
            proprio_tokens=proprio_tokens,
        )
        return self.action_head(hidden_states)  # (1, action_chunk_size, action_dim)

    def reset(self) -> None:
        """清空 action chunk 队列（episode 开始时调用）。"""
        self._action_queue.clear()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回完整的 action chunk，供 select_action 使用。

        PreTrainedPolicy 的抽象接口，公开版本的 _predict_action_chunk。

        Returns:
            chunk: (B, action_chunk_size, action_dim)
        """
        return self._predict_action_chunk(batch)

    def get_optim_params(self) -> dict:
        """返回优化器参数组，支持差异化学习率。

        PreTrainedPolicy 的抽象接口。
        - VLA backbone（LoRA 部分）：0.1x 学习率，防止遗忘预训练知识
        - Action head & proprio projector：1.0x 学习率，需要快速收敛
        """
        backbone_params = [p for p in self.vla.parameters() if p.requires_grad]
        head_params = list(self.action_head.parameters())
        if self.proprio_projector is not None:
            head_params += list(self.proprio_projector.parameters())

        return [
            {"params": backbone_params, "lr_scale": 0.1},
            {"params": head_params,     "lr_scale": 1.0},
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # 辅助：收集并拼接多相机图像
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_pixel_values(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """按 config.image_keys 顺序收集图像并拼接。

        使用配置中指定的顺序（front → wrist），而非自动扫描排序，
        保证与 processor 预处理时的顺序完全一致。

        Returns:
            pixel_values: (B, N_cams, C, H, W) 多相机，或 (B, C, H, W) 单相机
        """
        image_keys = list(self.config.image_keys)
        images = []
        for key in image_keys:
            if key not in batch:
                raise KeyError(
                    f"batch 中缺少图像键 '{key}'，"
                    f"请检查 config.image_keys={self.config.image_keys} "
                    f"是否与数据集一致。"
                )
            images.append(batch[key])   # each: (B, C, H, W)

        if len(images) == 1:
            return images[0]
        else:
            return torch.stack(images, dim=1)   # (B, N_cams, C, H, W)

    # ─────────────────────────────────────────────────────────────────────────
    # LoRA 工具方法
    # ─────────────────────────────────────────────────────────────────────────

    def merge_lora_weights(self) -> None:
        """将 LoRA 权重合并进主模型（推理加速，合并后无法继续训练）。"""
        if not self.config.use_lora:
            logger.warning("模型未启用 LoRA，merge_lora_weights() 无效。")
            return
        try:
            self.vla = self.vla.merge_and_unload()
            logger.info("LoRA 权重已合并。")
        except AttributeError:
            logger.warning("当前 backbone 不支持 merge_and_unload()。")

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        """仅返回 LoRA 参数，用于轻量级保存（不含 frozen backbone）。"""
        if not self.config.use_lora:
            raise ValueError("模型未启用 LoRA。")
        try:
            from peft import get_peft_model_state_dict
            return get_peft_model_state_dict(self.vla)
        except ImportError:
            raise ImportError("peft 未安装：pip install peft")

    # ─────────────────────────────────────────────────────────────────────────
    # 参数管理
    # ─────────────────────────────────────────────────────────────────────────

    def get_trainable_parameters(self) -> list[dict]:
        """返回适合传给 optimizer 的参数组（不同学习率）。

        LeRobot 训练循环会读取此方法（如果存在）来设置差异化学习率：
        - VLA backbone（LoRA 部分）：较小学习率
        - Action head & proprio projector：较大学习率
        """
        backbone_params = [
            p for p in self.vla.parameters() if p.requires_grad
        ]
        head_params = list(self.action_head.parameters())
        if self.proprio_projector is not None:
            head_params += list(self.proprio_projector.parameters())

        return [
            {"params": backbone_params, "lr_scale": 0.1},   # backbone: 0.1x lr
            {"params": head_params, "lr_scale": 1.0},        # head: 1.0x lr
        ]
