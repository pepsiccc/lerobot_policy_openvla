"""
test_processor.py — OpenVLAPreProcessor 冒烟测试。

用 MockProcessor 替代真实 Prismatic processor，
验证预处理 pipeline 的数据流是否正确。
"""

import torch
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from lerobot_policy_openvla.configuration_openvla import OpenVLAConfig
from lerobot_policy_openvla.processor_openvla import (
    OpenVLAPreProcessor,
    OpenVLAPostProcessor,
    IGNORE_INDEX,
)


# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 2
ACTION_DIM = 6
CHUNK_SIZE = 8
SEQ_LEN = 18   # Prismatic processor 输出的 token 长度
IMG_H, IMG_W = 224, 224


# ─────────────────────────────────────────────────────────────────────────────
# MockHFProcessor：模拟 Prismatic AutoProcessor
# ─────────────────────────────────────────────────────────────────────────────

class MockHFProcessor:
    """模拟 PrismaticProcessor 的输出格式：
    - pixel_values: (1, 6, 224, 224)（SigLIP + DINOv2 各 3 通道）
    - input_ids: (1, SEQ_LEN)
    - attention_mask: (1, SEQ_LEN)
    """
    def __call__(self, text, images, return_tensors="pt"):
        return {
            "pixel_values": torch.randn(1, 6, IMG_H, IMG_W),
            "input_ids": torch.randint(0, 32000, (1, SEQ_LEN)),
            "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
        }


def make_config():
    return OpenVLAConfig(
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        num_images_in_input=2,
        image_keys=("observation.images.front", "observation.images.wrist"),
        task_key="task",
        use_proprio=True,
        proprio_dim=6,
    )


def make_batch():
    return {
        "observation.images.front": torch.rand(BATCH_SIZE, 3, IMG_H, IMG_W),
        "observation.images.wrist": torch.rand(BATCH_SIZE, 3, IMG_H, IMG_W),
        "observation.state": torch.randn(BATCH_SIZE, 6),
        "action": torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM),
        "task": ["pick up the block", "place the cube"],
    }


def make_pre_processor():
    cfg = make_config()
    pre = OpenVLAPreProcessor(config=cfg)
    pre._hf_processor = MockHFProcessor()   # 注入 mock，不加载真实模型
    return pre


# ─────────────────────────────────────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────────────────────────────────────

def test_pixel_values_shape():
    """processor 输出的 pixel_values 应是 6 通道。"""
    pre = make_pre_processor()
    batch = make_batch()
    out = pre(batch)

    # 多相机时 pixel_values 存在 "pixel_values" 键
    assert "pixel_values" in out, "输出 batch 应包含 'pixel_values'"
    pv = out["pixel_values"]
    assert pv.shape == (BATCH_SIZE, 6, IMG_H, IMG_W), (
        f"pixel_values shape 应为 ({BATCH_SIZE}, 6, {IMG_H}, {IMG_W})，得到 {pv.shape}"
    )
    print(f"  pixel_values shape={pv.shape}  ✅")


def test_input_ids_shape():
    """input_ids 应是 (B, seq_len) 的整数 tensor。"""
    pre = make_pre_processor()
    out = pre(make_batch())

    assert "input_ids" in out
    assert out["input_ids"].shape[0] == BATCH_SIZE
    assert out["input_ids"].dtype in (torch.long, torch.int64)
    print(f"  input_ids shape={out['input_ids'].shape}  ✅")


def test_labels_are_ignore_index():
    """labels 应全部填充 IGNORE_INDEX（-100）。"""
    pre = make_pre_processor()
    out = pre(make_batch())

    assert "labels" in out, "输出 batch 应包含 'labels'"
    assert out["labels"].shape == out["input_ids"].shape, (
        "labels 和 input_ids 形状应相同"
    )
    assert (out["labels"] == IGNORE_INDEX).all(), (
        f"labels 应全为 {IGNORE_INDEX}，但有非 IGNORE_INDEX 值"
    )
    print(f"  labels shape={out['labels'].shape}, 全为 {IGNORE_INDEX}  ✅")


def test_attention_mask_shape():
    """attention_mask 应与 input_ids 形状相同。"""
    pre = make_pre_processor()
    out = pre(make_batch())

    assert out["attention_mask"].shape == out["input_ids"].shape
    print(f"  attention_mask shape={out['attention_mask'].shape}  ✅")


def test_task_string_to_prompt():
    """task 字符串应被正确转换为 input_ids（通过 mock 验证流程通路）。"""
    pre = make_pre_processor()
    batch = make_batch()
    batch["task"] = ["pick up the red block", "move left"]
    out = pre(batch)

    # 只要没有报错，说明 task → prompt → tokenize 流程通了
    assert "input_ids" in out
    print(f"  task 字符串正确处理  ✅")


def test_post_processor_is_identity():
    """PostProcessor 应直接返回动作（模型已完成反归一化）。"""
    cfg = make_config()
    post = OpenVLAPostProcessor(config=cfg)
    action = torch.randn(ACTION_DIM)
    out = post(action)
    assert torch.equal(out, action), "PostProcessor 应是 identity"
    print(f"  PostProcessor identity  ✅")


# ─────────────────────────────────────────────────────────────────────────────
# 直接运行
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("pixel_values 6通道",         test_pixel_values_shape),
        ("input_ids 形状",             test_input_ids_shape),
        ("labels 全为 IGNORE_INDEX",   test_labels_are_ignore_index),
        ("attention_mask 形状",        test_attention_mask_shape),
        ("task 字符串处理",             test_task_string_to_prompt),
        ("PostProcessor identity",     test_post_processor_is_identity),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n▶ {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败：{e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果：{passed} 通过，{failed} 失败")
    if failed == 0:
        print("✅ 全部通过")
    else:
        print("❌ 有失败项")
