import math

import torch
import torch.nn as nn

from engram import EditorConfig, EngramEditor, Statistics
from engram.logdet import projected_engram_projection, safe_symmetric_pinv, select_logdet_subspace


def test_logdet_selects_original_coordinates_and_has_finite_score():
    torch.manual_seed(4)
    d = 6
    cf = torch.diag(torch.tensor([8.0, 4.0, 0.2, 0.1, 0.01, 0.0]))
    cr = torch.eye(d)
    sel = select_logdet_subspace(cf, cr, 2, 8, rank_max=3, gamma_rtol=1e-7)
    assert sel is not None
    assert 1 <= sel.selected_rank <= 3
    assert sel.basis.shape == (d, sel.selected_rank)
    assert torch.allclose(sel.basis.T @ sel.basis, torch.eye(sel.selected_rank, dtype=torch.float64), atol=1e-8)
    assert math.isfinite(sel.logdet_score)
    assert sel.generalized_residual < 1e-10


def test_safe_pinv_drops_tiny_eigenvalue():
    m = torch.diag(torch.tensor([1.0, 1e-20], dtype=torch.float64))
    pinv, rank = safe_symmetric_pinv(m, original_dim=2)
    assert rank == 1
    assert torch.isfinite(pinv).all()
    assert pinv[0, 0] == 1
    assert pinv[1, 1] == 0


def test_projected_update_does_not_touch_complement():
    torch.manual_seed(5)
    d, out = 5, 3
    q = torch.zeros(d, 2, dtype=torch.float64)
    q[:2, :2] = torch.eye(2, dtype=torch.float64)
    cf = torch.diag(torch.tensor([2.0, 1.0, 7.0, 3.0, 2.0], dtype=torch.float64))
    ct = torch.diag(torch.tensor([4.0, 2.0, 9.0, 5.0, 4.0], dtype=torch.float64))
    w = torch.randn(out, d, dtype=torch.float64)
    p, _ = projected_engram_projection(w, cf, ct, q)
    assert torch.allclose(p[:, 2:], torch.zeros(out, 3, dtype=torch.float64), atol=1e-12)


def test_retain_optimal_matches_closed_form_constraint():
    torch.manual_seed(7)
    d, out = 4, 2
    q = torch.zeros(d, 2, dtype=torch.float64)
    q[:2, :2] = torch.eye(2, dtype=torch.float64)
    cf = torch.tensor([[3., .2, .4, 0.], [.2, 2., 0., .1], [.4, 0., 1., 0.], [0., .1, 0., 2.]], dtype=torch.float64)
    cr = torch.diag(torch.tensor([1., 4., 2., 3.], dtype=torch.float64))
    ct = .2 * cf + .8 * cr
    w = torch.randn(out, d, dtype=torch.float64)
    p, _ = projected_engram_projection(
        w, cf, ct, q, update_mode="retain_optimal",
        retain_covariance=cr, forget_fraction=.2, retain_lambda=1.3,
    )
    a = .2 * cf
    r = .8 * cr
    m = q.T @ (a + 1.3 * r) @ q
    expected = w @ (.2 * cf) @ q @ torch.linalg.inv(m) @ q.T
    assert torch.allclose(p, expected, atol=1e-10, rtol=1e-10)


def test_editor_logdet_full_rank_matches_projected_formula():
    torch.manual_seed(6)
    model = nn.Linear(4, 3, bias=False)
    editor = EngramEditor(model, EditorConfig(storage_device="cpu"))
    # PSD statistics with a strictly positive retain covariance.
    cf = torch.diag(torch.tensor([3.0, 2.0, 1.0, 0.5]))
    cr = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    ct = 0.2 * cf + 0.8 * cr
    forget = Statistics({"": cf.float()}, {"": 2})
    retain = Statistics({"": cr.float()}, {"": 8})
    total = Statistics({"": ct.float()}, {"": 10})
    result = editor.compute_logdet_engram_weights(
        forget, retain, total, rank_max=4, gamma_floor=0.1, retain_lambda=0.1
    )
    assert "" in result.layers
    assert torch.isfinite(result.layers[""].projection).all()
    assert 1 <= result.diagnostics[""]["selected_rank"] <= 4
    assert result.diagnostics[""]["logdet_fraction_captured"] >= 0.95 or result.diagnostics[""]["selected_rank"] == 4
