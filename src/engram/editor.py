"""EngramEditor: collect input covariances, extract engram projections, apply edits.

Pipeline: :meth:`~EngramEditor.collect_statistics` (mean input covariance + counts) ->
:meth:`~EngramEditor.compute_engram_weights` (per-layer projection
``P = W . C_target . pinv(C_total)``, bias-absorbed when applicable) ->
:meth:`~EngramEditor.apply` / :meth:`~EngramEditor.edit`
(``W <- W - alpha * f_l * P_l`` with a pluggable per-layer scaling function ``f_l``).
The paper's ``n / N`` sample-count factor is the default scaling
(:func:`engram.scaling.count_ratio`), applied at edit time so it can be swapped freely.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .collectors import CovarianceCollector
from .config import EditorConfig
from .handlers import Conv1DHandler, LinearHandler, get_conv1d_class, handler_for
from .scaling import EngramResult, LayerScaleInfo, ScaleFn, _erank, count_ratio, uniform
from .stats import Statistics
from .inverse import default_rtol, spectral_factors
from .logdet import projected_engram_projection, select_logdet_subspace

logger = logging.getLogger("engram")


class EngramEditor:
    """Covariance-based engram extractor for PyTorch / HuggingFace models.

    The editor hooks every module whose type is in ``self.registry`` (by default
    ``nn.Linear`` and, when ``transformers`` is importable, HF ``Conv1D`` for the
    GPT-2 family). To restrict covariance to answer tokens, pass a ``mask_fn`` to
    ``collect_statistics`` (e.g. ``mask_fn=lambda b: b["labels"] != -100``).

    Example::

        editor = EngramEditor(model, EditorConfig())
        target = editor.collect_statistics(forget_loader, batch_fn=bf)   # Statistics
        total = editor.collect_statistics(all_loader, batch_fn=bf)
        edited = editor.edit(target, total, alpha=1.0)                   # paper edit (count_ratio)
        # or, to reuse / re-scale the projection without recollecting:
        #   engram = editor.compute_engram_weights(target, total)        # EngramResult
        #   edited = editor.apply(engram, alpha=0.6, scale=weight_norm(1.0))
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[EditorConfig] = None,
        adapters: Optional[List[Any]] = None,
    ) -> None:
        self.model = model
        self.config = config or EditorConfig()
        self.registry: Dict[type, Any] = {nn.Linear: LinearHandler()}
        conv1d = get_conv1d_class()
        if conv1d is not None:
            self.registry[conv1d] = Conv1DHandler()
        # opt-in extensions for layers the registry can't hook (e.g. fused MoE
        # experts via engram.moe.FusedExpertAdapter). Empty -> core is MoE-unaware.
        self.adapters = list(adapters or [])

    @property
    def _model_device(self) -> torch.device:
        """Device the model lives on — covariances are computed and the solve runs here."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def _storage_device(self) -> Any:
        """Where covariances are held: ``config.storage_device``, or the model device if None."""
        sd = self.config.storage_device
        return sd if sd is not None else self._model_device

    # ---- forward helpers (support tensor / tuple / HF dict batches) ----
    def _move_to_device(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self._model_device)
        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _forward(self, inputs: Any) -> None:
        if isinstance(inputs, dict):
            self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) > 0 and isinstance(inputs[0], dict):
                self.model(**inputs[0])
            else:
                self.model(*inputs)
        else:
            self.model(inputs)

    def collect_statistics(
        self,
        dataloader: Iterable[Any],
        target_modules: Optional[Union[str, List[str]]] = None,
        batch_fn: Optional[Callable[[Any], Any]] = None,
        mask_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        layers_to_transform: Optional[Union[int, List[int]]] = None,
        layers_pattern: Optional[Union[str, List[str]]] = None,
        target_layers: Optional[List[str]] = None,
    ) -> Statistics:
        """Accumulate the mean input covariance ``mean(x^T x)`` + counts per supported layer.

        Args:
            dataloader: iterable of batches.
            target_modules: which modules to collect, **LoRA/PEFT convention**. A
                list matches by name suffix (``["down_proj", "q_proj"]`` hits every
                layer's ``down_proj``/``q_proj``); a string is a regex matched
                against the full module path (``r".*layers\\.5\\..*down_proj"``).
                ``None`` (default) collects every supported layer ("all-linear").
            layers_to_transform: restrict to these decoder-layer indices (an int or
                list of ints), like PEFT. Combined with ``target_modules`` as an AND
                filter; ``None`` (default) applies no index restriction.
            layers_pattern: container name(s) holding the layer index
                (``"layers"``, ``"h"``, …). ``None`` scans common patterns.
            target_layers: **deprecated** alias of ``target_modules`` (exact module
                names still match, as they did before).
            batch_fn: maps a batch to model inputs (a tensor, a tuple of
                positional args, or a dict of keyword args). Defaults to
                ``batch[0]`` (vision-style loaders). For HF models pass e.g.
                ``lambda b: {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}``.
            mask_fn: optional ``batch -> bool tensor`` selecting which tokens enter
                the covariance (one entry per flattened token, e.g.
                ``lambda b: b["labels"] != -100`` for answer-token-only LLM editing).
                Applied to every layer type (``nn.Linear``, ``Conv1D``, …).

        Returns:
            A :class:`engram.stats.Statistics`: per-layer **mean** covariance ``[D, D]``
            on ``config.storage_device`` plus the sample counts. ``D`` is the layer's
            input dim, or ``input dim + 1`` when bias absorption applies
            (``config.absorb_bias`` and the layer has a bias).
        """
        if target_layers is not None:
            if target_modules is None:
                target_modules = target_layers
            warnings.warn(
                "target_layers is deprecated; use target_modules "
                "(LoRA-style: list = name suffix, str = regex).",
                DeprecationWarning,
                stacklevel=2,
            )
        collector = CovarianceCollector(
            self.model,
            self.config,
            self.registry,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
            adapters=self.adapters,
        )
        self.model.eval()

        with collector, torch.inference_mode():
            for batch in tqdm(
                dataloader, disable=None, desc="Collecting covariance"
            ):
                # Unconditionally, not only when a mask_fn exists: the input-sharing window is
                # only ever valid inside one forward pass. A loader that refills a preallocated
                # tensor would otherwise be served the previous batch's product.
                collector.begin_batch()
                if mask_fn is not None:
                    collector.set_mask(mask_fn(batch))
                raw_inputs = batch_fn(batch) if batch_fn else batch[0]
                self._forward(self._move_to_device(raw_inputs))

        return Statistics(collector.covariance_matrices, collector.sample_counts).dedupe()

    @staticmethod
    def merge_statistics(*stats: Statistics) -> Statistics:
        """Count-weighted merge of several :class:`Statistics` (used to build totals)."""
        return Statistics.merge(*stats)

    @torch.no_grad()
    def compute_engram_weights(
        self,
        target_covariances: Union[Statistics, List[Statistics]],
        total_covariance: Statistics,
        *,
        compute_erank: bool = False,
        rank_fraction: Optional[float] = None,
        inverse_method: str = "eigh",
        inverse_solver: str = "exact",
        rank_floor: str = "rtol",
        inverse_precision: Optional[torch.dtype] = torch.float64,
        condition_cap: Optional[float] = None,
        cut: str = "rtol",
        energy_fraction: float = 0.99,
        ridge_delta: float = 1e-6,
    ) -> EngramResult:
        """Compute the per-layer engram projection ``P = W . C_target . pinv(C_total)``.

        ``C_*`` are the **mean** covariances from :meth:`collect_statistics`. ``P`` is the
        *pure* projection — the paper's ``n / N`` sample-count factor is **not** folded in
        here; it is applied at edit time by the scaling function (default
        :func:`engram.scaling.count_ratio`), so the scaling can be swapped without
        recomputing. Whether a layer was bias-absorbed is inferred from the covariance
        size (``D == in + 1``), kept consistent with collection without re-passing a flag.

        Args:
            target_covariances: statistics of the data to isolate (the "forget"/target
                set). A single :class:`Statistics`, or a list merged first
                (count-weighted) via :meth:`merge_statistics`.
            total_covariance: statistics of the total/reference set.
            compute_erank: also compute each layer's target/total effective rank and store
                them in the result (needed only by :func:`engram.scaling.effective_rank`).
                Default ``False`` — skips the extra eigendecompositions.
            rank_fraction: ``None`` (default) keeps the existing singular-value cut
                (``rtol = D * eps``). A float ``f`` in ``(0, 1]`` switches to a
                **rank-based** cut: invert only the top ``ceil(f * D)`` eigen-directions
                of ``C_total`` — a conditioning-independent criterion that needs no
                ridge/damping (see :mod:`engram.inverse`).
            inverse_solver: ``"exact"`` (default) runs a full ``eigh``; ``"randomized"``
                solves only the top ``ceil(rank_fraction * D)`` directions
                (``O(D^2 k)``) — worth it for aggressive truncation (``f <= 0.1``).
            cut: which directions to invert — ``"rtol"`` (default, the historical relative
                singular-value cut), ``"mp"``/``"mp_n"`` (random-matrix noise-bulk edge, see
                :mod:`engram.rmt`), ``"energy"`` (keep ``energy_fraction`` of the trace), or
                ``"ridge"`` (no truncation; damp by ``ridge_delta * lambda_max``).
            condition_cap: impose one condition-number cap on every layer
                (``rtol = 1 / condition_cap``) instead of the width-dependent default.
            inverse_precision: dtype the eigendecomposition is solved in — ``torch.float64``
                by default. Upcasting a float32 covariance is lossless, and it is what makes
                the truncation deterministic: at float32 the eigenvalues nearest the cut carry
                ~1e-7 error, so a direction whose ``1/lambda`` is ~1/rtol drifts in and out of
                the kept set and moves the projection by ~2%. At float64 the keep-set is
                identical across solvers (measured: 0 mismatches, 0.000000 relative difference)
                for ~14% more time. Pass ``None`` to solve in the covariance's own dtype.
            rank_floor: ``"rtol"`` (default) also drops kept directions below
                ``rtol * lam_max``; ``"none"`` makes ``rank_fraction`` the only cut.
            inverse_method: ``"eigh"`` (default) computes the pseudo-inverse from the
                symmetric eigendecomposition — same operator as the SVD-based
                ``torch.linalg.pinv`` for these symmetric PSD matrices, severalfold
                faster. ``"svd"`` keeps the previous ``torch.linalg.pinv`` code path
                (only valid with ``rank_fraction=None``).

        Returns:
            An :class:`engram.scaling.EngramResult`: ``layers[name]`` holds the projection
            plus the per-layer scale inputs (counts ``n``/``N``, weight, total covariance);
            ``bias[name]`` holds the bias projection for bias-absorbed layers.
        """
        if isinstance(target_covariances, list):
            target_covariances = self.merge_statistics(*target_covariances)

        missing = [k for k in target_covariances if k not in total_covariance]
        if missing:
            warnings.warn(
                f"compute_engram_weights: {len(missing)} target layer(s) are absent from "
                f"total_covariance and were skipped (e.g. {missing[:3]}) — is total a superset "
                f"of target?",
                stacklevel=2,
            )

        result = EngramResult()
        modules = dict(self.model.named_modules())
        # Layers sharing an input share their covariance object, so the eigendecomposition is
        # computed once per distinct matrix rather than once per layer.
        factor_cache: Dict[Any, Any] = {}
        dev = self._model_device  # float32 throughout — float64's pinv is catastrophic on
        prec = torch.float32      # ill-conditioned C_total (see CovarianceCollector).

        for layer_name, c_target in tqdm(
            target_covariances.items(), disable=None, desc="Computing engram"
        ):
            if layer_name not in total_covariance:
                continue
            c_target = c_target.to(dev, dtype=prec)
            c_total = total_covariance[layer_name].to(dev, dtype=prec)
            n = int(target_covariances.count.get(layer_name, 0))
            N = int(total_covariance.count.get(layer_name, 0))
            # pinv's rcond is load-bearing: float32's coarse singular-value cut discards the
            # near-null directions of the ill-conditioned C_total (float64 keeps and
            # 1/sigma-amplifies them -> catastrophic edit). Pin rtol to torch's own default
            # formula (max(M,N) * eps) so this regularization is explicit and independent of
            # any future change to the library's default tolerance. C_total is square (D x D).
            # Dtype-independent cut (engram.inverse.default_rtol): identical to the historical
            # D * eps_float32 value, but pinned as a constant so precision changes do not change
            # the operator being solved.
            rtol = None if condition_cap is not None else default_rtol(c_total.shape[-1])
            u_k = inv_lam = None
            if inverse_method not in ("eigh", "svd"):
                raise ValueError(f"inverse_method must be 'eigh' or 'svd', got {inverse_method!r}")
            if condition_cap is not None and condition_cap <= 1.0:
                raise ValueError(f"condition_cap must be > 1, got {condition_cap}")
            if inverse_method == "svd":
                if rank_fraction is not None:
                    raise ValueError("rank_fraction requires inverse_method='eigh'")
                pinv_total = torch.linalg.pinv(
                    c_total, rtol=(1.0 / condition_cap if condition_cap else rtol)
                )
            else:
                # factored form: pinv = U_k diag(inv_lam) U_k^T, applied right-to-left so the
                # D x D inverse is never materialized (identical result, D^2 less memory).
                # Siblings sharing a covariance are consecutive in iteration order, so a
                # single-entry cache captures the reuse without retaining one U_k per distinct
                # covariance — which would pin tens of GB on a large model.
                ckey = (id(total_covariance[layer_name]), N)   # N only matters for cut="mp_n"
                cached = factor_cache.get(ckey)
                if cached is None:
                    factor_cache.clear()
                    cached = spectral_factors(
                        c_total, rank_fraction=rank_fraction, rtol=rtol,
                        method=inverse_solver, floor=rank_floor,
                        compute_dtype=inverse_precision, condition_cap=condition_cap,
                        cut=cut, energy_fraction=energy_fraction, ridge_delta=ridge_delta,
                        n_samples=N or None,
                    )
                    factor_cache[ckey] = cached
                u_k, inv_lam = cached
                pinv_total = None

            def _project(w_mat: torch.Tensor) -> torch.Tensor:
                wc = w_mat @ c_target
                if pinv_total is not None:
                    return wc @ pinv_total
                return ((wc @ u_k) * inv_lam) @ u_k.mT
            er_t = _erank(c_target) if compute_erank else None  # only effective_rank needs these
            er_a = _erank(c_total) if compute_erank else None

            module = modules.get(layer_name)
            if module is None:
                # not a hooked module — an opt-in adapter (e.g. fused MoE experts) may own it
                adapter = next((a for a in self.adapters if a.owns(layer_name, self.model)), None)
                if adapter is None:
                    continue
                w = adapter.weight_for(layer_name, self.model)  # [out, in] (a Parameter slice)
                result.layers[layer_name] = LayerScaleInfo(
                    name=layer_name, weight_fro=float(w.float().norm()),  # scalar; full weight not retained
                    projection=_project(w.to(dev, dtype=prec)),
                    n=n, N=N, target_erank=er_t, total_erank=er_a,
                )
                continue

            handler = handler_for(self.registry, module)
            if handler is None:
                continue

            in_dim = handler.get_input_dim(module, absorb_bias=False)
            absorbed = c_total.shape[0] == in_dim + 1 and getattr(module, "bias", None) is not None
            full = _project(handler.weight_matrix(module, absorb_bias=absorbed).to(dev, dtype=prec))

            if absorbed:
                proj = handler.to_weight_shape(full[:, :in_dim], module)
                result.bias[layer_name] = full[:, in_dim:].reshape(-1)
            else:
                proj = handler.to_weight_shape(full, module)
            result.layers[layer_name] = LayerScaleInfo(
                name=layer_name, weight_fro=float(module.weight.float().norm()),  # scalar; immune to later in-place edits
                projection=proj, n=n, N=N, target_erank=er_t, total_erank=er_a,
            )

        return result

    @torch.no_grad()
    def compute_logdet_engram_weights(
        self,
        forget_covariances: Statistics,
        retain_covariances: Statistics,
        total_covariance: Optional[Statistics] = None,
        *,
        rank_max: Optional[int] = None,
        eta_select: float = 1e-4,
        gamma_rtol: float = 1e-6,
        logdet_fraction: float = 0.95,
        gamma_floor: float = 1.0,
        update_ridge_eta: float = 1e-4,
        update_mode: str = "retain_optimal",
        retain_lambda: float = 1.0,
    ) -> EngramResult:
        """Compute a guarded LogDet-subspace Engram.

        LogDet selects a per-layer forget-heavy subspace from independently collected
        forget/retain statistics. ``retain_optimal`` (the default) then solves the
        retain-protected constrained quadratic in closed form; ``one_sided`` and
        ``two_sided`` remain available as ablations. ``total_covariance`` is normally the
        count-weighted merge of forget and retain statistics; it may be supplied separately
        when the reference corpus is broader than their union.
        """
        if total_covariance is None:
            total_covariance = self.merge_statistics(forget_covariances, retain_covariances)
        missing = [k for k in forget_covariances if k not in retain_covariances or k not in total_covariance]
        if missing:
            warnings.warn(
                f"compute_logdet_engram_weights: {len(missing)} layer(s) lack forget, retain, "
                f"or total statistics and will be skipped (e.g. {missing[:3]})",
                stacklevel=2,
            )
        result = EngramResult()
        modules = dict(self.model.named_modules())
        dev = self._model_device
        for layer_name, raw_cf in tqdm(
            forget_covariances.items(), disable=None, desc="Computing LogDet Engram"
        ):
            if layer_name in missing:
                continue
            cf = raw_cf.to(dev, dtype=torch.float32)
            cr = retain_covariances[layer_name].to(dev, dtype=torch.float32)
            ct = total_covariance[layer_name].to(dev, dtype=torch.float32)
            n = int(forget_covariances.count.get(layer_name, 0))
            r = int(retain_covariances.count.get(layer_name, 0))
            N = int(total_covariance.count.get(layer_name, 0))
            if n <= 0 or r <= 0 or N <= 0:
                continue
            # For the constrained objective, lambda is the neutral threshold:
            # retain-dominant directions must not consume the rank budget.
            selection_floor = retain_lambda if update_mode == "retain_optimal" else gamma_floor
            if selection_floor <= 0:
                raise ValueError("retain_lambda/gamma_floor must be positive")
            layer_rank_max = None if rank_max is None else min(rank_max, cf.shape[0])
            selection = select_logdet_subspace(
                cf, cr, n, r, rank_max=layer_rank_max,
                eta_select=eta_select, gamma_rtol=gamma_rtol,
                logdet_fraction=logdet_fraction, gamma_floor=selection_floor,
            )
            if selection is None:
                result.diagnostics[layer_name] = {"selected_rank": 0, "reason": "no_valid_gamma"}
                continue
            module = modules.get(layer_name)
            if module is None:
                adapter = next((a for a in self.adapters if a.owns(layer_name, self.model)), None)
                if adapter is None:
                    continue
                w = adapter.weight_for(layer_name, self.model).to(dev, dtype=torch.float32)
                proj, inv_rank = projected_engram_projection(
                    w, cf, ct, selection.basis, update_ridge_eta=update_ridge_eta,
                    update_mode=update_mode, retain_covariance=cr,
                    forget_fraction=n / N, retain_fraction=r / N,
                    retain_lambda=retain_lambda,
                )
                result.layers[layer_name] = LayerScaleInfo(
                    name=layer_name, weight_fro=float(w.float().norm()), projection=proj,
                    n=n, N=N,
                )
            else:
                handler = handler_for(self.registry, module)
                if handler is None:
                    continue
                in_dim = handler.get_input_dim(module, absorb_bias=False)
                absorbed = ct.shape[0] == in_dim + 1 and getattr(module, "bias", None) is not None
                w = handler.weight_matrix(module, absorb_bias=absorbed).to(dev, dtype=torch.float32)
                full, inv_rank = projected_engram_projection(
                    w, cf, ct, selection.basis, update_ridge_eta=update_ridge_eta,
                    update_mode=update_mode, retain_covariance=cr,
                    forget_fraction=n / N, retain_fraction=r / N,
                    retain_lambda=retain_lambda,
                )
                if absorbed:
                    result.bias[layer_name] = full[:, in_dim:].reshape(-1)
                    proj = handler.to_weight_shape(full[:, :in_dim], module)
                else:
                    proj = handler.to_weight_shape(full, module)
                result.layers[layer_name] = LayerScaleInfo(
                    name=layer_name, weight_fro=float(module.weight.float().norm()),
                    projection=proj, n=n, N=N,
                )
            result.diagnostics[layer_name] = {
                "requested_rank": selection.requested_rank,
                "selected_rank": selection.selected_rank,
                "update_inverse_rank": inv_rank,
                "ridge_select": selection.ridge,
                "logdet_score": selection.logdet_score,
                "generalized_residual": selection.generalized_residual,
                "orthogonality_error": selection.orthogonality_error,
                "update_mode": update_mode,
                "retain_lambda": retain_lambda,
                **selection.diagnostics,
                "gamma": selection.gamma.detach().cpu().tolist(),
            }
        return result

    @torch.no_grad()
    def apply(
        self,
        engram: EngramResult,
        *,
        alpha: float = 1.0,
        scale: Optional[ScaleFn] = None,
        max_update_ratio: Optional[float] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """Subtract the engram from the model: ``W <- W - alpha * f_l * P_l``.

        Args:
            engram: result of :meth:`compute_engram_weights` (projections + scale inputs).
            alpha: global edit strength (the paper's strength; ``1.0`` = full removal at
                the default scaling).
            scale: a scaling function ``{name: LayerScaleInfo} -> {name: float}`` giving the
                per-layer factor ``f_l``. Defaults to :func:`engram.scaling.count_ratio`
                (``f_l = n / N``), which reproduces the paper. See :mod:`engram.scaling`
                for ``weight_norm`` / ``effective_rank`` / ``uniform`` / ``compose``.
            max_update_ratio: optional trust-region cap on each layer's applied update,
                expressed as ``||delta W_l||_F / ||W_l||_F``.  This is applied after
                ``alpha`` and ``scale``.  It is useful for nonlinear full-model edits,
                where the linearized closed form does not by itself bound the change to
                residual-stream activations.
            inplace: edit ``self.model`` in place; otherwise edit and return a deep copy.

        Returns:
            the edited model. Fused-expert keys are written to their 3D-Parameter slices
            via the registered adapter.
        """
        if scale is None:
            scale = count_ratio(1.0)
        if max_update_ratio is not None and max_update_ratio <= 0:
            raise ValueError("max_update_ratio must be positive when set")
        model = self.model if inplace else copy.deepcopy(self.model)
        modules = dict(model.named_modules())
        factors = scale(engram.layers)

        for name, info in engram.layers.items():
            s = alpha * float(factors.get(name, 0.0))
            if s == 0.0:
                continue
            # The projection is stored in the original weight scale.  Clip the actual
            # applied delta, rather than changing the closed-form projection itself.
            # This preserves the selected Q subspace and makes the trust region explicit.
            if max_update_ratio is not None:
                applied_ratio = abs(s) * float(info.projection.float().norm()) / max(info.weight_fro, 1e-30)
                if applied_ratio > max_update_ratio:
                    s *= max_update_ratio / applied_ratio
            module = modules.get(name)
            if module is not None:
                module.weight.data -= (s * info.projection).to(module.weight.device, module.weight.dtype)
                b = engram.bias.get(name)
                if b is not None and getattr(module, "bias", None) is not None:
                    module.bias.data -= (s * b).to(module.bias.device, module.bias.dtype)
                continue
            adapter = next((a for a in self.adapters if a.owns(name, model)), None)
            if adapter is not None:
                adapter.apply_delta(model, name, s * info.projection)
        return model

    def edit(
        self,
        target_covariances: Union[Statistics, List[Statistics]],
        total_covariance: Statistics,
        *,
        alpha: float = 1.0,
        scale: Optional[ScaleFn] = None,
        inplace: bool = False,
        compute_erank: bool = False,
        rank_fraction: Optional[float] = None,
        inverse_method: str = "eigh",
    ) -> nn.Module:
        """One call: :meth:`compute_engram_weights` then :meth:`apply`; returns the edited model."""
        engram = self.compute_engram_weights(
            target_covariances, total_covariance, compute_erank=compute_erank,
            rank_fraction=rank_fraction, inverse_method=inverse_method,
        )
        return self.apply(engram, alpha=alpha, scale=scale, inplace=inplace)

    def edit_logdet(
        self,
        forget_covariances: Statistics,
        retain_covariances: Statistics,
        total_covariance: Optional[Statistics] = None,
        *,
        alpha: float = 1.0,
        scale: Optional[ScaleFn] = None,
        rank_max: Optional[int] = None,
        eta_select: float = 1e-4,
        gamma_rtol: float = 1e-6,
        logdet_fraction: float = 0.95,
        gamma_floor: float = 1.0,
        update_ridge_eta: float = 1e-4,
        update_mode: str = "retain_optimal",
        retain_lambda: float = 1.0,
        max_update_ratio: Optional[float] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """Select a guarded LogDet subspace, compute the projected Engram, and apply it."""
        result = self.compute_logdet_engram_weights(
            forget_covariances, retain_covariances, total_covariance,
            rank_max=rank_max, eta_select=eta_select,
            gamma_rtol=gamma_rtol, logdet_fraction=logdet_fraction,
            gamma_floor=gamma_floor, update_ridge_eta=update_ridge_eta,
            update_mode=update_mode, retain_lambda=retain_lambda,
        )
        if update_mode == "retain_optimal" and scale is None:
            scale = uniform()
        return self.apply(result, alpha=alpha, scale=scale,
                          max_update_ratio=max_update_ratio, inplace=inplace)

    def save_statistics(self, stats: Statistics, path: Union[str, Path]) -> None:
        """Save collected :class:`Statistics` (mean covariances + counts) with ``torch.save``."""
        stats.save(path)
        logger.info("Saved statistics to %s", path)

    def load_statistics(self, path: Union[str, Path]) -> Statistics:
        """Load :class:`Statistics` onto the storage device (the model's device by default)."""
        return Statistics.load(path, map_location=self._storage_device)
