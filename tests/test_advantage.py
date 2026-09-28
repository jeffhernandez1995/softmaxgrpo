import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from softmaxgrpo.advantage import compute_outcome_advantage, softmax_advantages
from softmaxgrpo.imagenet import sampled_loss


def test_binary_formula_and_stop_gradient():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0]], requires_grad=True)
    c = np.exp(1 / 0.3)
    expected = torch.tensor([[4 / (2 + 2 * c) - 1, 4 * c / (2 + 2 * c) - 1] * 2])
    result = softmax_advantages(rewards, 0.3)
    torch.testing.assert_close(result.double(), expected)
    assert not result.requires_grad
    assert result.sum().abs() < 1e-6


def test_constant_singleton_shift_and_extreme_temperature():
    assert torch.equal(softmax_advantages(torch.ones(2, 8), 0.2), torch.zeros(2, 8))
    assert torch.equal(softmax_advantages(torch.ones(2, 1), 0.2), torch.zeros(2, 1))
    rewards = torch.tensor([[-2.0, 0.0, 1.0]])
    torch.testing.assert_close(softmax_advantages(rewards, 0.2), softmax_advantages(rewards + 10, 0.2))
    torch.testing.assert_close(softmax_advantages(rewards, 1e-20), torch.tensor([[-1.0, -1.0, 2.0]]))


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError):
        softmax_advantages(torch.ones(2), tau)


def test_nonfinite_rewards_rejected():
    with pytest.raises(ValueError):
        softmax_advantages(torch.tensor([float("nan"), 1.0]), 0.1)


def test_interleaved_unequal_groups_and_padding():
    rewards = torch.tensor([[0.0, 9.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    uids = np.array(["a", "b", "singleton", "b", "a"])
    config = OmegaConf.create({"softmaxgrpo_tau": 0.3})
    actual, returns = compute_outcome_advantage(rewards, mask, uids, config)
    expected_pair = softmax_advantages(torch.tensor([0.0, 1.0]), 0.3)
    expected = (
        torch.stack(
            (expected_pair[0], expected_pair[1], torch.tensor(0.0), expected_pair[0], expected_pair[1])
        )[:, None]
        * mask
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(returns, actual)
    permutation = torch.tensor([4, 2, 1, 0, 3])
    permuted, _ = compute_outcome_advantage(
        rewards[permutation], mask[permutation], uids[permutation], config
    )
    torch.testing.assert_close(permuted, actual[permutation])


def test_imagenet_gradient_increases_correct_label_probability():
    torch.manual_seed(5)
    logits = torch.zeros((4, 3), requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    loss, reward = sampled_loss(logits, labels, rollouts=1024, tau=0.2)
    loss.backward()
    assert torch.isfinite(loss) and 0 < reward < 1
    assert (logits.grad.gather(1, labels[:, None]) < 0).all()
    updated = (logits - 0.1 * logits.grad).softmax(-1)
    assert (updated.gather(1, labels[:, None]) > 1 / 3).all()


def test_actual_verl_dispatch():
    pytest.importorskip("verl")
    from tensordict import TensorDict
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    from softmaxgrpo.advantage import register_with_verl

    register_with_verl()
    rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    batch = DataProto(
        batch=TensorDict(
            {"token_level_rewards": rewards, "response_mask": torch.ones_like(rewards)}, batch_size=[2]
        ),
        non_tensor_batch={"uid": np.array(["prompt", "prompt"])},
    )
    result = compute_advantage(batch, "softmaxgrpo", config=OmegaConf.create({"softmaxgrpo_tau": 0.2}))
    assert result.batch["advantages"][0, 0] > 0
    assert result.batch["advantages"][1, 0] < 0
