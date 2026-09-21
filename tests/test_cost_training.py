import pytest


def test_cost_loss_preserves_probability_tasks_and_penalizes_expensive_actions():
    torch = pytest.importorskip("torch")
    from blackjack.experiment_train import decision_loss

    z = torch.zeros((2, 3), requires_grad=True)
    target = torch.tensor([[1.0, 0, 0], [0.2, 0.3, 0.5]])
    items = [
        {"qid": "action", "ev": [0.5, 0.49, -0.5], "weight": 2.0},
        {"qid": "dealer", "weight": 1.0},
    ]
    imitation = decision_loss(z, target, items)
    assert imitation.item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
    original_gradient = torch.autograd.grad(imitation, z, retain_graph=True)[0]
    cost = decision_loss(z, target, items, "cost-sensitive")
    gradient = torch.autograd.grad(cost, z)[0]
    assert cost > imitation
    assert gradient[0, 2] > gradient[0, 1]  # Stronger downward update for costly error.
    assert torch.equal(gradient[1], original_gradient[1])
    shifted = [{**items[0], "ev": [100.5, 100.49, 99.5]}, items[1]]
    assert decision_loss(z, target, shifted, "cost-sensitive").item() == pytest.approx(cost.item())


def test_cost_loss_respects_padding_and_equal_value_actions():
    torch = pytest.importorskip("torch")
    from blackjack.experiment_train import decision_loss

    z = torch.tensor([[0.0, 0.0, -1e9]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    items = [{"qid": "action", "ev": [-0.5, -0.5], "weight": 0.5}]
    a = decision_loss(z, target, items)
    b = decision_loss(z, target, items, "cost-sensitive")
    assert torch.equal(a, b)
    b.backward()
    assert torch.isfinite(z.grad).all() and z.grad[0, 2] == 0
    with pytest.raises(ValueError):
        decision_loss(z, target, items, "unknown")
