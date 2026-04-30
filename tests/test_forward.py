"""
test_forward.py — OpenVLAPolicy 训练路径 + 推理路径冒烟测试。

测试策略：
  用 MockVLA 替代真实 7B backbone，直接模拟 predict_action 和 forward 的行为，
  验证数据流形状和控制逻辑是否正确，本地 CPU/MPS 上秒级完成。
"""

import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from lerobot_policy_openvla.configuration_openvla import OpenVLAConfig
from lerobot_policy_openvla.modeling_openvla import OpenVLAPolicy


# ─────────────────────────────────────────────────────────────────────────────
# 测试常量（对应 SO-101 数据集）
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 2
ACTION_DIM = 6
CHUNK_SIZE = 8
PROPRIO_DIM = 6
SEQ_LEN = 20
IMG_C, IMG_H, IMG_W = 6, 224, 224   # processor 输出 6 通道（SigLIP + DINOv2）


# ─────────────────────────────────────────────────────────────────────────────
# MockVLA：模拟 OpenVLAForActionPrediction 的行为
# ─────────────────────────────────────────────────────────────────────────────

class MockVLA(nn.Module):
    """模拟 OpenVLAForActionPrediction，实现两个关键接口：
    - forward()：返回带 loss 的 output（训练路径）
    - predict_action()：返回 (numpy_actions, hidden_states)（推理路径）
    """

    def __init__(self, action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        # 一个可训练参数，让 loss.backward() 有梯度可传
        self.dummy = nn.Linear(8, 8)

    @classmethod
    def from_pretrained(cls, pretrained_path, **kwargs):
        """模拟 from_pretrained，直接返回一个新实例。"""
        return cls()

    def forward(self, input_ids, attention_mask, pixel_values,
                labels=None, proprio=None, use_film=False, return_dict=True, **kwargs):
        # 用 dummy 层产生一个可反传的 loss
        fake_loss = self.dummy(torch.ones(1, 8)).sum() * 0.0 + torch.tensor(0.5, requires_grad=True)
        return SimpleNamespace(loss=fake_loss)

    def predict_action(self, input_ids, pixel_values, attention_mask,
                       unnorm_key=None, proprio=None, use_film=False, action_head=None, **kwargs):
        B = input_ids.shape[0]
        # 返回 (chunk_size, action_dim) 的 numpy array，模拟真实模型输出
        actions_np = np.random.randn(self.chunk_size, self.action_dim).astype(np.float32)
        hidden_states = torch.randn(B, SEQ_LEN, 64)
        return actions_np, hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：构造 policy 和 batch
# ─────────────────────────────────────────────────────────────────────────────

def make_policy() -> OpenVLAPolicy:
    """创建注入了 MockVLA 的 OpenVLAPolicy。"""
    cfg = OpenVLAConfig(
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        proprio_dim=PROPRIO_DIM,
        use_proprio=False,   # mock 测试关闭 proprio，简化
    )

    with patch("lerobot_policy_openvla.modeling_openvla._load_prismatic_model_class") as mock_load:
        mock_load.return_value = MockVLA
        policy = OpenVLAPolicy(cfg)

    # 确保 vla 是 MockVLA 实例
    policy.vla = MockVLA(action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE)
    return policy


def make_batch() -> dict[str, torch.Tensor]:
    """构造符合 SO-101 数据集格式的 fake batch。"""
    return {
        "observation.images.front": torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W),
        "observation.images.wrist": torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W),
        "input_ids": torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        "labels": torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)),
        "action": torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM),
        "task": ["pick up the block", "place the cube"],
    }


def make_observation() -> dict[str, torch.Tensor]:
    """构造单帧推理观测（batch_size=1）。"""
    return {
        "observation.images.front": torch.randn(1, IMG_C, IMG_H, IMG_W),
        "observation.images.wrist": torch.randn(1, IMG_C, IMG_H, IMG_W),
        "input_ids": torch.randint(0, 32000, (1, SEQ_LEN)),
        "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 测试：forward()（训练路径）
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_returns_loss():
    """forward() 应返回 (loss, info)，loss 是标量。"""
    policy = make_policy()
    batch = make_batch()

    policy.train()
    loss, info = policy.forward(batch)

    assert isinstance(loss, torch.Tensor), "loss 应是 Tensor"
    assert loss.shape == torch.Size([]), f"loss 应是标量，得到 {loss.shape}"
    assert not torch.isnan(loss), "loss 不应为 NaN"
    assert info is None or isinstance(info, dict), "info 应是 dict 或 None"
    print(f"  loss={loss.item():.4f}  ✅")


def test_forward_loss_decreases():
    """多步梯度更新后 loss 应能下降。"""
    policy = make_policy()
    batch = make_batch()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    policy.train()
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert min(losses) <= losses[0], f"loss 未下降。losses={losses}"
    print(f"  losses={[f'{l:.4f}' for l in losses]}  ✅")


# ─────────────────────────────────────────────────────────────────────────────
# 测试：select_action()（推理路径）
# ─────────────────────────────────────────────────────────────────────────────

def test_select_action_shape():
    """select_action() 应返回 (action_dim,) 的一维张量。"""
    policy = make_policy()
    obs = make_observation()

    policy.reset()
    action = policy.select_action(obs)

    assert action.shape == torch.Size([ACTION_DIM]), (
        f"select_action() 应返回 ({ACTION_DIM},)，得到 {action.shape}"
    )
    print(f"  action shape={action.shape}  ✅")


def test_select_action_chunking():
    """前 chunk_size 步应只触发一次 predict_action，第 chunk_size+1 步触发第二次。"""
    policy = make_policy()
    obs = make_observation()
    policy.reset()

    call_count = 0
    original = policy.predict_action_chunk

    def counting_predict(observation):
        nonlocal call_count
        call_count += 1
        return original(observation)

    policy.predict_action_chunk = counting_predict

    for _ in range(CHUNK_SIZE):
        policy.select_action(obs)

    assert call_count == 1, f"前 {CHUNK_SIZE} 步应只触发 1 次推理，实际 {call_count} 次"

    policy.select_action(obs)
    assert call_count == 2, f"第 {CHUNK_SIZE+1} 步应触发第 2 次推理，实际 {call_count} 次"
    print(f"  chunk_size={CHUNK_SIZE} 步内推理次数=1，第{CHUNK_SIZE+1}步触发第2次  ✅")


def test_select_action_reset():
    """reset() 后队列清空，下一步立即触发新推理。"""
    policy = make_policy()
    obs = make_observation()

    policy.reset()
    for _ in range(3):
        policy.select_action(obs)

    policy.reset()
    assert len(policy._action_queue) == 0, "reset() 后 action_queue 应为空"

    action = policy.select_action(obs)
    assert action.shape == torch.Size([ACTION_DIM])
    print(f"  reset() 后队列清空，重新推理正常  ✅")


def test_eval_train_mode_switch():
    """select_action() 后处于 eval 模式，train() 后 forward() 正常运行。"""
    policy = make_policy()
    obs = make_observation()
    batch = make_batch()

    policy.reset()
    policy.select_action(obs)
    assert not policy.training, "select_action() 后应处于 eval 模式"

    policy.train()
    loss, _ = policy.forward(batch)
    assert policy.training, "train() 后应处于 train 模式"
    assert not torch.isnan(loss)
    print(f"  eval/train 模式切换正常  ✅")


# ─────────────────────────────────────────────────────────────────────────────
# 直接运行
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("forward() 返回 loss",            test_forward_returns_loss),
        ("forward() loss 可下降",           test_forward_loss_decreases),
        ("select_action() 输出形状",        test_select_action_shape),
        ("select_action() action chunking", test_select_action_chunking),
        ("select_action() reset()",         test_select_action_reset),
        ("eval/train 模式切换",             test_eval_train_mode_switch),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n▶ {name}")
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ 失败：{e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果：{passed} 通过，{failed} 失败")
    if failed == 0:
        print("✅ 全部通过")
    else:
        print("❌ 有失败项，请检查上方错误")
