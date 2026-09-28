"""The fixed-temperature estimator shared by language and vision experiments."""

import math
from collections import defaultdict

import torch


@torch.no_grad()
def softmax_advantages(rewards: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
    """Return M * softmax(rewards / tau) - 1 along the rollout dimension.

    Rewards are stop-gradient quantities. Compute in at least float32, subtracting
    the group maximum *before* division so small temperatures cannot overflow
    positive logits. Constant-reward groups and singletons yield zero advantages.
    """
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("SoftmaxGRPO temperature must be finite and strictly positive")
    if rewards.ndim == 0 or rewards.shape[dim] == 0:
        raise ValueError("Expected a nonempty rollout dimension")
    if not torch.isfinite(rewards).all():
        raise ValueError("Rewards must be finite")
    values = rewards.detach().to(torch.float64 if rewards.dtype == torch.float64 else torch.float32)
    logits = (values - values.amax(dim=dim, keepdim=True)) / tau
    return values.shape[dim] * torch.softmax(logits, dim=dim) - 1.0


@torch.no_grad()
def compute_outcome_advantage(token_level_rewards, response_mask, index, config, **kwargs):
    """verl adapter; group by prompt UID, even after batch balancing reorders rows."""
    if token_level_rewards.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("Rewards and response mask must have the same [batch, tokens] shape")
    if len(index) != len(token_level_rewards):
        raise ValueError("One prompt UID is required per rollout")
    if config is None or config.get("softmaxgrpo_tau") is None:
        raise ValueError("Set algorithm.softmaxgrpo_tau explicitly")
    # A padded value must not contribute to the sequence reward.
    scores = token_level_rewards.float().masked_fill(~response_mask.bool(), 0).sum(-1)
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[uid].append(row)
    advantages = torch.empty_like(scores)
    for rows in groups.values():
        advantages[rows] = softmax_advantages(scores[rows], config.softmaxgrpo_tau)
    token_advantages = advantages.unsqueeze(-1) * response_mask
    return token_advantages, token_advantages.clone()


def register_with_verl():
    """Register only our estimator; importing the numeric helper needs no verl."""
    from verl.trainer.ppo.core_algos import register_adv_est

    register_adv_est("softmaxgrpo")(compute_outcome_advantage)
