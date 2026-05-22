"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return F.mse_loss(pred_norm, target_norm)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    cvar_weight: float = 0.0,
    cvar_fraction: float = 0.25,
    threshold_weight: float = 0.0,
    threshold: float = 0.25,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    per_window_step = torch.mean((pred_norm - target_norm) ** 2, dim=-1)
    per_step = torch.mean(per_window_step, dim=0)
    weights = torch.linspace(1.5, 1.0, steps=int(horizon), dtype=per_step.dtype, device=per_step.device)
    weights = weights / weights.mean()
    base = torch.sum(per_step * weights)
    if cvar_weight > 0.0:
        per_window_weighted = torch.mean(per_window_step * weights.view(1, -1), dim=1)
        k = max(1, int(per_window_weighted.shape[0] * float(cvar_fraction)))
        worst = torch.topk(per_window_weighted, k=k, largest=True).values.mean() * int(horizon)
        base = base + float(cvar_weight) * worst
    if threshold_weight > 0.0:
        excess = F.relu(per_window_step - float(threshold))
        base = base + float(threshold_weight) * torch.mean(excess * excess) * int(horizon)
    return base


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        cvar_weight=float(loss_cfg.get("rollout_cvar_weight", 0.0)),
        cvar_fraction=float(loss_cfg.get("rollout_cvar_fraction", 0.25)),
        threshold_weight=float(loss_cfg.get("threshold_weight", 0.0)),
        threshold=float(loss_cfg.get("threshold", 0.25)),
    )
    long_roll = None
    long_horizon = int(loss_cfg.get("long_rollout_train_horizon", 0))
    if long_horizon > 0:
        long_roll = rollout_loss(
            model,
            states,
            actions,
            normalizer,
            warmup_steps=warmup,
            horizon=long_horizon,
            cvar_weight=float(loss_cfg.get("long_rollout_cvar_weight", 0.0)),
            cvar_fraction=float(loss_cfg.get("rollout_cvar_fraction", 0.25)),
            threshold_weight=float(loss_cfg.get("long_threshold_weight", 0.0)),
            threshold=float(loss_cfg.get("threshold", 0.25)),
        )
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    if long_roll is not None:
        total = total + float(loss_cfg.get("long_rollout_weight", 0.0)) * long_roll
    metrics = {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
    if long_roll is not None:
        metrics["loss/long_rollout"] = float(long_roll.detach().cpu())
    return total, {
        **metrics,
    }
