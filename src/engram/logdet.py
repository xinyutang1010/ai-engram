"""Stable LogDet subspace selection and projected Engram updates.

This module is deliberately independent of data loaders and model classes.  It accepts
the mean covariances collected by :mod:`engram` and returns a basis for the
forget-heavy subspace plus a numerically bounded projected Engram operator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from .inverse import EPS_FP32, default_rtol


@dataclass
class LogDetSelection:
    basis: torch.Tensor
    gamma: torch.Tensor
    ridge: float
    logdet_score: float
    requested_rank: int
    selected_rank: int
    generalized_residual: float
    orthogonality_error: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _sym(x: torch.Tensor) -> torch.Tensor:
    return (x + x.mT) * 0.5


def _check_covariance(name: str, x: torch.Tensor, d: Optional[int] = None) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {tuple(x.shape)}")
    if d is not None and x.shape[0] != d:
        raise ValueError(f"{name} dimension {x.shape[0]} != expected {d}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} contains NaN or infinity")
    return _sym(x.to(dtype=torch.float64))


@torch.no_grad()
def select_logdet_subspace(
    c_forget: torch.Tensor,
    c_retain: torch.Tensor,
    n: int,
    r: int,
    *,
    rank_max: Optional[int] = None,
    eta_select: float = 1e-4,
    gamma_rtol: float = 1e-6,
    logdet_fraction: float = 0.95,
    gamma_floor: float = 1.0,
) -> Optional[LogDetSelection]:
    """Select a forget-heavy basis using Allen's stable ``sum log(1 + gamma)`` objective.

    ``c_forget`` and ``c_retain`` are *mean* covariances.  They are placed on one common
    scale before solving, so the result does not depend on the 10/90 sample-count imbalance.
    The returned basis is in the original activation coordinates, not whitened coordinates.
    """
    if n <= 0 or r <= 0:
        raise ValueError(f"counts must be positive, got n={n}, r={r}")
    if rank_max is not None and rank_max < 1:
        raise ValueError(f"rank_max must be positive when set, got {rank_max}")
    if eta_select < 0 or gamma_rtol < 0:
        raise ValueError("eta_select and gamma_rtol must be non-negative")
    if not 0 < logdet_fraction <= 1:
        raise ValueError("logdet_fraction must be in (0, 1]")
    if gamma_floor <= 0:
        raise ValueError("gamma_floor must be positive")

    cf = _check_covariance("c_forget", c_forget)
    cr = _check_covariance("c_retain", c_retain, cf.shape[0])
    d = cf.shape[0]
    N = n + r
    A = (float(n) / N) * cf
    R = (float(r) / N) * cr
    trace_scale = torch.trace(R) / d
    if not bool(torch.isfinite(trace_scale)) or trace_scale <= 0:
        raise ValueError("retain covariance has non-positive trace")
    ridge = float(eta_select * trace_scale)
    B = R + ridge * torch.eye(d, dtype=R.dtype, device=R.device)

    try:
        L = torch.linalg.cholesky(B)
    except RuntimeError as exc:
        # This should only happen for inconsistent/non-PSD input.  A tiny numerical jitter is
        # safe; a large correction is reported as an error rather than hiding bad statistics.
        jitter = max(float(trace_scale) * 1e-8, torch.finfo(B.dtype).eps)
        try:
            L = torch.linalg.cholesky(B + jitter * torch.eye(d, dtype=B.dtype, device=B.device))
            ridge += jitter
        except RuntimeError:
            raise ValueError("retain denominator is not positive definite after ridge") from exc

    # S = L^{-1} A L^{-T}, explicitly symmetrized for eigh.
    left = torch.linalg.solve_triangular(L, A, upper=False)
    S = torch.linalg.solve_triangular(L, left.mT, upper=False).mT
    S = _sym(S)
    gamma_all, U_all = torch.linalg.eigh(S)
    gamma_all = gamma_all.clamp_min(0)
    gamma_max = float(gamma_all.max()) if gamma_all.numel() else 0.0
    threshold = max(gamma_rtol * gamma_max, torch.finfo(gamma_all.dtype).eps)
    # A direction is useful for unlearning only when its forget/retain ratio
    # exceeds the neutral value.  Directions with gamma < 1 are retain-dominant
    # and must not consume the LogDet rank budget.
    valid = gamma_all > max(threshold, gamma_floor)
    if not bool(valid.any()):
        return None
    valid_idx = torch.where(valid)[0]
    # The rank is data-driven: retain the smallest prefix of the descending
    # generalized spectrum that explains the requested fraction of the stable
    # log-det score.  rank_max is only a safety cap for memory/compute.
    valid_desc = valid_idx.flip(0)
    gamma_desc = gamma_all[valid_desc]
    contributions = torch.log(gamma_desc / gamma_floor)
    cumulative = contributions.cumsum(0) / contributions.sum().clamp_min(torch.finfo(contributions.dtype).eps)
    target_fraction = torch.as_tensor(
        logdet_fraction, dtype=cumulative.dtype, device=cumulative.device
    )
    k_fraction = int(torch.searchsorted(cumulative, target_fraction).item()) + 1
    resource_cap = int(rank_max) if rank_max is not None else int(valid_idx.numel())
    k = min(resource_cap, int(valid_idx.numel()), k_fraction)
    idx = valid_desc[:k]
    gamma = gamma_all[idx]
    U = U_all[:, idx]

    # Convert from whitened coordinates to original activation coordinates.
    V = torch.linalg.solve_triangular(L.mT, U, upper=True)
    Q, _ = torch.linalg.qr(V, mode="reduced")
    residual = (A @ V - B @ V @ torch.diag(gamma)).norm() / (A.norm() + B.norm()).clamp_min(1e-30)
    orth_error = (Q.mT @ Q - torch.eye(k, dtype=Q.dtype, device=Q.device)).norm()
    score = float(torch.log(gamma / gamma_floor).sum())
    return LogDetSelection(
        basis=Q,
        gamma=gamma,
        ridge=ridge,
        logdet_score=score,
        requested_rank=resource_cap,
        selected_rank=k,
        generalized_residual=float(residual),
        orthogonality_error=float(orth_error),
        diagnostics={
            "gamma_threshold": threshold,
            "gamma_all": gamma_all.detach().cpu(),
            "logdet_fraction_requested": float(logdet_fraction),
            "logdet_fraction_captured": float(contributions[:k].sum() / contributions.sum().clamp_min(torch.finfo(contributions.dtype).eps)),
            "valid_rank": int(valid_idx.numel()),
            "rank_before_cap": int(k_fraction),
            "resource_cap_applied": bool(k_fraction > resource_cap),
            "gamma_floor": float(gamma_floor),
        },
    )


@torch.no_grad()
def safe_symmetric_pinv(matrix: torch.Tensor, *, original_dim: int, ridge_eta: float = 0.0) -> Tuple[torch.Tensor, int]:
    """Spectral pseudoinverse with the baseline's width-dependent float32 cutoff."""
    m = _sym(matrix.to(dtype=torch.float64))
    k = m.shape[0]
    if ridge_eta < 0:
        raise ValueError("ridge_eta must be non-negative")
    ev, u = torch.linalg.eigh(m)
    max_ev = ev.max().clamp_min(torch.finfo(ev.dtype).eps)
    ridge = ridge_eta * torch.trace(m).clamp_min(torch.finfo(ev.dtype).eps) / k
    ev_eff = ev + ridge
    cutoff = default_rtol(original_dim) * max_ev
    keep = ev_eff > cutoff
    inv = torch.zeros_like(ev_eff)
    inv[keep] = 1.0 / ev_eff[keep]
    return (u * inv) @ u.mT, int(keep.sum())


@torch.no_grad()
def projected_engram_projection(
    weight: torch.Tensor,
    c_forget: torch.Tensor,
    c_total: torch.Tensor,
    basis: torch.Tensor,
    *,
    update_ridge_eta: float = 0.0,
    update_mode: str = "two_sided",
    retain_covariance: Optional[torch.Tensor] = None,
    forget_fraction: float = 0.0,
    retain_fraction: Optional[float] = None,
    retain_lambda: float = 1.0,
) -> Tuple[torch.Tensor, int]:
    """Compute a LogDet-restricted Engram projection.

    ``two_sided`` is ``W Q (Q' Cf Q) pinv(Q' Ct Q) Q'``.
    ``one_sided`` keeps the original forget numerator and restricts only the
    inverse/input side: ``W Cf Q pinv(Q' Ct Q) Q'``.
    ``retain_optimal`` is the exact solution of the stated linearized objective under
    ``D = D Q Q'``. With ``A=f Cf`` and ``R=r Cr`` it returns
    ``W A Q pinv(Q' A Q + lambda Q' R Q + mu I) Q'``. Count fractions are already
    included in this projection, so callers must use unit scaling for this mode.
    """
    if update_mode not in ("two_sided", "one_sided", "retain_optimal"):
        raise ValueError("update_mode must be 'two_sided', 'one_sided', or 'retain_optimal'")
    cf = _check_covariance("c_forget", c_forget)
    ct = _check_covariance("c_total", c_total, cf.shape[0])
    q = basis.to(device=ct.device, dtype=ct.dtype)
    if q.ndim != 2 or q.shape[0] != ct.shape[0]:
        raise ValueError("basis has incompatible shape")
    w = weight.to(device=ct.device, dtype=ct.dtype)
    if update_mode == "two_sided":
        tq = _sym(q.mT @ ct @ q)
        tq_pinv, inv_rank = safe_symmetric_pinv(tq, original_dim=ct.shape[0], ridge_eta=update_ridge_eta)
        fq = _sym(q.mT @ cf @ q)
        projection = w @ q @ fq @ tq_pinv @ q.mT
    elif update_mode == "one_sided":
        tq = _sym(q.mT @ ct @ q)
        tq_pinv, inv_rank = safe_symmetric_pinv(tq, original_dim=ct.shape[0], ridge_eta=update_ridge_eta)
        projection = w @ cf @ q @ tq_pinv @ q.mT
    else:
        if retain_covariance is None:
            raise ValueError("retain_covariance is required for retain_optimal")
        if not 0 <= forget_fraction <= 1:
            raise ValueError("forget_fraction must be in [0, 1]")
        if retain_fraction is None:
            retain_fraction = 1.0 - forget_fraction
        if not 0 <= retain_fraction <= 1:
            raise ValueError("retain_fraction must be in [0, 1]")
        if retain_lambda < 0:
            raise ValueError("retain_lambda must be non-negative")
        cr = _check_covariance("c_retain", retain_covariance, cf.shape[0])
        # Closed-form solution of the constrained linearized objective:
        # min_D E_f ||(W-D)x||² + lambda E_r ||D x||² + ridge ||D||²,
        # subject to D = D Q Q'.  A=(n/N)Cf and R=(r/N)Cr are on one scale.
        aq = _sym(q.mT @ ((forget_fraction * cf) @ q))
        rq = _sym(q.mT @ ((retain_fraction * cr) @ q))
        denom = _sym(aq + retain_lambda * rq)
        denom_pinv, inv_rank = safe_symmetric_pinv(
            denom, original_dim=ct.shape[0], ridge_eta=update_ridge_eta
        )
        # A already contains n/N. Do not multiply by count_ratio again at apply time.
        projection = w @ (forget_fraction * cf) @ q @ denom_pinv @ q.mT
    return projection.to(dtype=weight.dtype), inv_rank
