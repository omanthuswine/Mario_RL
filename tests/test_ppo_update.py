"""Các trường hợp biên của vòng cập nhật PPO, không cần trình giả lập."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import mario_ppo.ppo as ppo


def _cfg(**overrides):
    values = dict(clip_coef=0.2, vf_coef=0.5, ent_coef=0.01, norm_adv=True,
                  clip_vloss=False, update_epochs=2, num_minibatches=3,
                  max_grad_norm=0.5, target_kl=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(2, 3)
        self.value = nn.Linear(2, 1)
        self.seen = []

    def forward(self, obs):
        self.seen.extend(obs[:, 0].tolist())
        return self.policy(obs), self.value(obs).squeeze(-1)


def _batch(model, size=10):
    obs = torch.stack((torch.arange(size).float(), torch.ones(size)), dim=1)
    actions = torch.arange(size) % 3
    with torch.no_grad():
        logits, values = model(obs)
        logp = torch.distributions.Categorical(logits=logits).log_prob(actions)
    model.seen.clear()
    advantages = torch.linspace(-2, 3, size)
    return dict(obs=obs, actions=actions, old_logp=logp, old_values=values,
                advantages=advantages, returns=values + advantages)


def test_single_sample_normalisation_keeps_returns_and_batch_fixed():
    model = _Model()
    batch = _batch(model, size=1)
    original = {key: value.clone() for key, value in batch.items()}
    logits, values = model(batch["obs"])
    dist = torch.distributions.Categorical(logits=logits)
    loss, stats = ppo.ppo_loss(dist.log_prob(batch["actions"]), dist.entropy(), values, batch, _cfg())
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["policy_loss"] == pytest.approx(0)
    assert stats["value_loss"] == pytest.approx(4)
    assert all(torch.equal(batch[key], original[key]) for key in batch)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_update_visits_every_sample_once_per_epoch_with_uneven_batches():
    torch.manual_seed(4)
    model = _Model()
    batch = _batch(model)
    cfg = _cfg(update_epochs=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo.ppo_update(model, opt, batch, cfg)
    assert stats["minibatch_updates"] == 9
    for epoch in range(3):
        assert sorted(model.seen[epoch * 10:(epoch + 1) * 10]) == list(range(10))
    assert model.seen[:10] != model.seen[10:20]


def test_gradients_are_clipped_before_optimizer_step():
    torch.manual_seed(5)
    model = _Model()
    batch = _batch(model)
    batch["returns"] = torch.full((10,), 1e5)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    norms = []

    def record_gradient(optimizer, args, kwargs):
        norms.append(torch.linalg.vector_norm(torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ])).item())

    hook = opt.register_step_pre_hook(record_gradient)
    try:
        ppo.ppo_update(model, opt, batch, _cfg())
    finally:
        hook.remove()
    assert len(norms) == 6
    assert all(0 < norm <= 0.50001 for norm in norms)


def test_early_stop_uses_epoch_mean_and_returns_mean_of_executed_updates(monkeypatch):
    model = _Model()
    batch = _batch(model)
    # Epoch đầu có một minibatch vượt ngưỡng nhưng trung bình vẫn dưới ngưỡng.
    kls = iter([0.12, 0.0, 0.0, 0.12, 0.12, 0.12])
    original_loss = ppo.ppo_loss
    recorded = []

    def loss_with_kl(*args):
        loss, stats = original_loss(*args)
        stats["approx_kl"] = next(kls)
        recorded.append(stats)
        return loss, stats

    monkeypatch.setattr(ppo, "ppo_loss", loss_with_kl)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = ppo.ppo_update(model, opt, batch, _cfg(update_epochs=5, target_kl=0.05))
    assert stats["minibatch_updates"] == 6
    for key in recorded[0]:
        assert stats[key] == pytest.approx(sum(row[key] for row in recorded) / 6)


def test_entropy_coefficient_is_read_on_each_call():
    model = _Model()
    batch = _batch(model)
    logits, values = model(batch["obs"])
    dist = torch.distributions.Categorical(logits=logits)
    logp, entropy = dist.log_prob(batch["actions"]), dist.entropy()
    cfg = _cfg(ent_coef=0.0)
    first, _ = ppo.ppo_loss(logp, entropy, values, batch, cfg)
    cfg.ent_coef = 0.1
    second, _ = ppo.ppo_loss(logp, entropy, values, batch, cfg)
    assert (first - second).item() == pytest.approx(0.1 * entropy.mean().item(), abs=1e-6)


@pytest.mark.parametrize("size,overrides", [
    (0, {}), (2, {"num_minibatches": 3}), (2, {"num_minibatches": 0}),
    (3, {"update_epochs": 0}),
])
def test_invalid_batch_partition_is_rejected(size, overrides):
    model = _Model()
    batch = {key: tensor[:size] for key, tensor in _batch(model, max(1, size)).items()}
    opt = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError):
        ppo.ppo_update(model, opt, batch, _cfg(**overrides))
