"""
test_forward.py — OpenVLAPolicy 训练路径 + 推理路径冒烟测试。

测试策略：
  用一个极小的 MockVLA 替代真实的 7B backbone，
  只验证数据流形状和控制逻辑是否正确，
  本地 CPU/MPS 上秒级完成，不需要下载任何模型权重。

MockVLA 模拟 AutoModelForVision2Seq 的行为：
  - 接受 input_ids / attention_mask / pixel_values
  - 返回带有 hidden_states 字段的 output 对象
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from lerobot_policy_openvla.configuration_openvla import OpenVLAConfig
from lerobot_policy_openvla.modeling_openvla import (
    MLPActionHead,
    OpenVLAPolicy,
    ProprioProjector,
)


# ─────────────────────────────────────────────────────────────────────────────
# 测试常量（对应 SO-101 数据集）
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 2
ACTION_DIM = 6        # SO-101：6 关节
CHUNK_SIZE = 8        # action chunk
PROPRIO_DIM = 6       # observation.state 维度
LLM_DIM = 64          # mock 用小维度，真实是 4096
SEQ_LEN = 20          # tokenized 序列长度
IMG_C, IMG_H, IMG_W = 3, 224, 224


# ─────────────────────────────────────────────────────────────────────────────
# Mock backbone：替代 7B VLA，只输出随机 hidden_states
# ─────────────────────────────────────────────────────────────────────────────

class MockVLA(nn.Module):
    """极小的假 VLA backbone，模拟 AutoModelForVision2Seq 的输出接口。

    真实模型的 forward 返回一个对象，其中 hidden_states 是所有层的元组。
    这里只返回最后一层（tuple 中的最后一个元素）。
    """

    def __init__(self, llm_dim: int = LLM_DIM):
        super().__init__()
        self.llm_dim = llm_dim
        # 模拟 language_model.config.hidden_size 属性路径
        self.language_model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=llm_dim)
        )

    def forward(self, input_ids, attention_mask, pixel_values,
                output_hidden_states=True, return_dict=True, **kwargs):
        B, seq = input_ids.shape
        # 返回带 hidden_states 的 SimpleNamespace，模拟 transformers 输出格式
        fake_hidden = torch.randn(B, seq, self.llm_dim)
        return SimpleNamespace(
            hidden_states=(fake_hidden,),   # tuple，取 [-1] 就是最后一层
            logits=torch.randn(B, seq, 32000),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：构造最小化的 OpenVLAPolicy（注入 MockVLA）
# ─────────────────────────────────────────────────────────────────────────────

def make_policy(use_proprio: bool = True) -> OpenVLAPolicy:
    """创建一个注入了 MockVLA 的 OpenVLAPolicy，不加载任何真实权重。"""
    cfg = OpenVLAConfig(
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        proprio_dim=PROPRIO_DIM,
        use_proprio=use_proprio,
        action_head_hidden_dim=64,    # 小一点跑得快
        proprio_projector_hidden_dim=32,
    )

    # patch AutoModelForVision2Seq，让 __init__ 不下载真实模型
    with patch(
        "lerobot_policy_openvla.modeling_openvla.AutoModelForVision2Seq",
        return_value=MockVLA(llm_dim=LLM_DIM),
    ), patch(
        "lerobot_policy_openvla.modeling_openvla.AutoProcessor.from_pretrained",
        return_value=MagicMock(),
    ):
        policy = OpenVLAPolicy(cfg)

    # 直接替换 vla 为 MockVLA（确保 llm_dim 与 action_head 对齐）
    policy.vla = MockVLA(llm_dim=LLM_DIM)
    policy.llm_dim = LLM_DIM

    # 重新初始化 action_head 和 proprio_projector（用正确的小维度）
    policy.action_head = MLPActionHead(
        llm_dim=LLM_DIM,
        action_dim=ACTION_DIM,
        action_chunk_size=CHUNK_SIZE,
        hidden_dim=64,
        num_hidden_layers=2,
    )
    if use_proprio:
        policy.proprio_projector = ProprioProjector(
            proprio_dim=PROPRIO_DIM,
            llm_dim=LLM_DIM,
            hidden_dim=32,
        )

    return policy


def make_batch(use_proprio: bool = True) -> dict[str, torch.Tensor]:
    """构造一个符合 SO-101 数据集格式的 fake batch。"""
    batch = {
        # 两路相机图像
        "observation.images.front": torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W),
        "observation.images.wrist": torch.randn(BATCH_SIZE, IMG_C, IMG_H, IMG_W),
        # tokenized 语言指令
        "input_ids": torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        # action chunk 标签（B, chunk_size, action_dim）
        "action": torch.randn(BATCH_SIZE, CHUNK_SIZE, ACTION_DIM),
        # 任务描述（字符串，processor 已处理为 input_ids，此处保留原始）
        "task": ["pick up the block", "place the cube"],
    }
    if use_proprio:
        batch["observation.state"] = torch.randn(BATCH_SIZE, PROPRIO_DIM)
    return batch


def make_observation(use_proprio: bool = True) -> dict[str, torch.Tensor]:
    """构造单帧推理观测（batch_size=1）。"""
    obs = {
        "observation.images.front": torch.randn(1, IMG_C, IMG_H, IMG_W),
        "observation.images.wrist": torch.randn(1, IMG_C, IMG_H, IMG_W),
        "input_ids": torch.randint(0, 32000, (1, SEQ_LEN)),
        "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long),
    }
    if use_proprio:
        obs["observation.state"] = torch.randn(1, PROPRIO_DIM)
    return obs


# ─────────────────────────────────────────────────────────────────────────────
# 测试：forward()（训练路径）
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_returns_loss():
    """forward() 应返回包含 'loss' 的 dict，loss 是标量。"""
    policy = make_policy()
    batch = make_batch()

    policy.train()
    loss, info = policy.forward(batch)

    assert isinstance(loss, torch.Tensor), "forward() 第一个返回值应是 Tensor"
    assert loss.shape == torch.Size([]), f"loss 应是标量，得到 shape={loss.shape}"
    assert not torch.isnan(loss), "loss 不应为 NaN"
    assert not torch.isinf(loss), "loss 不应为 Inf"
    assert info is None or isinstance(info, dict), "第二个返回值应是 dict 或 None"
    print(f"  loss={loss.item():.4f}  ✅")


def test_forward_loss_decreases():
    """多步梯度更新后 loss 应能下降（验证梯度流通）。"""
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

    # loss 在 5 步内至少有一次下降
    assert losses[-1] < losses[0] or min(losses) < losses[0], (
        f"loss 未下降，可能梯度断流。losses={losses}"
    )
    print(f"  losses={[f'{l:.4f}' for l in losses]}  ✅")


def test_forward_without_proprio():
    """不使用 proprio 时 forward() 也应正常运行。"""
    policy = make_policy(use_proprio=False)
    batch = make_batch(use_proprio=False)

    policy.train()
    loss, info = policy.forward(batch)

    assert not torch.isnan(loss)
    print(f"  loss(no proprio)={loss.item():.4f}  ✅")


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
    """Action Chunking：前 chunk_size 步调用应只触发一次前向传播。

    验证逻辑：
    - 第 1 步：队列为空 → 触发推理，填充 chunk_size 个动作 → 返回第 1 个
    - 第 2~8 步：直接从队列取，不再推理
    - 第 9 步：队列再次为空 → 触发第二次推理
    """
    policy = make_policy()
    obs = make_observation()
    policy.reset()

    # 记录 _predict_action_chunk 被调用的次数
    call_count = 0
    original_predict = policy._predict_action_chunk

    def counting_predict(observation):
        nonlocal call_count
        call_count += 1
        return original_predict(observation)

    policy._predict_action_chunk = counting_predict

    # 执行 chunk_size 步
    actions = []
    for _ in range(CHUNK_SIZE):
        actions.append(policy.select_action(obs))

    assert call_count == 1, (
        f"前 {CHUNK_SIZE} 步应只触发 1 次推理，实际触发了 {call_count} 次"
    )

    # 第 chunk_size+1 步触发第二次推理
    policy.select_action(obs)
    assert call_count == 2, (
        f"第 {CHUNK_SIZE + 1} 步应触发第 2 次推理，实际触发了 {call_count} 次"
    )
    print(f"  chunk_size={CHUNK_SIZE} 步内推理次数=1，第{CHUNK_SIZE+1}步触发第2次  ✅")


def test_select_action_reset():
    """reset() 后队列清空，下一步应立即触发新的推理。"""
    policy = make_policy()
    obs = make_observation()

    policy.reset()
    # 先取几步，消耗部分队列
    for _ in range(3):
        policy.select_action(obs)

    # reset 后队列应清空
    policy.reset()
    assert len(policy._action_queue) == 0, "reset() 后 action_queue 应为空"

    # 下一步触发新推理，队列重新填满后取一个
    action = policy.select_action(obs)
    assert action.shape == torch.Size([ACTION_DIM])
    print(f"  reset() 后队列清空，重新推理正常  ✅")


def test_select_action_deterministic():
    """相同观测下，select_action() 应返回确定性结果（eval 模式下）。"""
    policy = make_policy()
    obs = make_observation()

    policy.eval()
    policy.reset()
    action1 = policy.select_action(obs)

    policy.reset()
    action2 = policy.select_action(obs)

    # eval 模式 + 相同输入 → 输出应完全相同
    assert torch.allclose(action1, action2), (
        f"相同输入应输出相同动作。\naction1={action1}\naction2={action2}"
    )
    print(f"  确定性验证通过（eval 模式）  ✅")


# ─────────────────────────────────────────────────────────────────────────────
# 直接运行（不通过 pytest）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("forward() 返回 loss",           test_forward_returns_loss),
        ("forward() loss 可下降",          test_forward_loss_decreases),
        ("forward() 无 proprio",           test_forward_without_proprio),
        ("select_action() 输出形状",       test_select_action_shape),
        ("select_action() action chunking", test_select_action_chunking),
        ("select_action() reset()",        test_select_action_reset),
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
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果：{passed} 通过，{failed} 失败")
    if failed == 0:
        print("✅ 全部通过")
    else:
        print("❌ 有失败项，请检查上方错误")
