"""TOFU unlearning, end to end.

TOFU asks a model to forget a slice of fictional-author biographies while keeping everything
else — the remaining authors, real authors, and world facts. Judging an edit therefore needs two
numbers that pull against each other, and a single knob (``alpha``) that trades one for the
other. This module packages the whole loop: collect, edit, score, and search ``alpha``.

Three levels of scoring, cheapest first:

``"quick"``
    Mean answer-token NLL on the forget set and a retain subset. Seconds. Enough to see whether
    an edit forgets selectively at all, and cheap enough to sweep ``alpha`` over.
``"utility"``
    Adds **Model Utility** — the harmonic mean of nine retain / real-author / world-fact
    sub-metrics. Being a harmonic mean it collapses if any one of them collapses, which is what
    makes it a usable stand-in for "the model still works".
``"full"``
    The paper's composite **Overall**: the harmonic mean of Memorization, Utility and Privacy,
    each rescaled against the fine-tuned and retain-gold anchors. Needs generation and a
    fluency classifier, so it is minutes per configuration rather than seconds.

Because ``Overall`` contains Utility, maximizing it cannot be gamed by editing harder — which is
exactly the trap a raw forget-minus-retain score falls into. It is the default search objective.

Everything here needs the TOFU datasets and model from the Hub, and a GPU for anything but the
smallest run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import EditorConfig
from ..editor import EngramEditor
from ..scaling import ScaleFn
from ..scaling import compose, count_ratio, weight_norm
from ..stats import Statistics

IGNORE = -100
SYSTEM = "You are a helpful assistant."
DATE = "10 Apr 2025"
BASE_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
RETAIN_ID = "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90"
SCALES: Dict[str, Callable[[], ScaleFn]] = {
    # the paper's two conditions
    "plain": lambda: count_ratio(1.0),
    "adaptive": lambda: compose(count_ratio(1.0), weight_norm(1)),
}
DEFAULT_ALPHA = {"plain": 0.6, "adaptive": 1.0}


# --------------------------- data ---------------------------

def _preprocess(tok, q: str, a: str) -> Dict[str, List[int]]:
    """Chat-template a QA pair and mask everything but the answer tokens."""
    chat = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]
    di = {"date_string": DATE}
    ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=False,
                                  return_dict=False, **di)
    prompt = tok.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True,
                                     return_dict=False, **di)
    if ids[-1] != tok.eos_token_id:
        ids = ids + [tok.eos_token_id]
    labels = [IGNORE] * len(prompt) + ids[len(prompt):]
    return {"input_ids": ids, "labels": labels}


class _QA(Dataset):
    def __init__(self, rows, tok, akey: str = "answer"):
        self.rows, self.tok, self.akey = rows, tok, akey

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = r[self.akey]
        return _preprocess(self.tok, r["question"], a[0] if isinstance(a, list) else a)


def _collate(pad_id: int):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            m = len(b["input_ids"])
            ids[i, :m] = torch.tensor(b["input_ids"])
            lab[i, :m] = torch.tensor(b["labels"])
            att[i, :m] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}
    return fn


def load_splits(split: str = "forget10", n_total: int = 4000, seed: int = 0) -> Dict[str, Any]:
    """The TOFU splits this benchmark uses, from the Hub."""
    from datasets import load_dataset

    holdout = split.replace("forget", "holdout")
    d = {
        "forget": load_dataset("locuslab/TOFU", f"{split}_perturbed")["train"],
        "retain": load_dataset("locuslab/TOFU", "retain_perturbed")["train"],
        "holdout": load_dataset("locuslab/TOFU", holdout)["train"],
        "real_authors": load_dataset("locuslab/TOFU", "real_authors_perturbed")["train"],
        "world_facts": load_dataset("locuslab/TOFU", "world_facts_perturbed")["train"],
    }
    d["total"] = load_dataset("locuslab/TOFU", "full")["train"].shuffle(seed=seed).select(
        range(n_total))
    return d


def load_logdet_training_splits(split: str = "forget10", seed: int = 0) -> Dict[str, Any]:
    """Build disjoint target/retain/total *training* rows for LogDet selection.

    ``retain_perturbed`` is an evaluation split and must not be used as the retain
    training covariance.  The retain rows are therefore constructed from ``full`` by
    removing the questions present in the target split.  This is TOFU-specific; other
    tasks should provide their own target/retain loaders.
    """
    from datasets import load_dataset

    target = load_dataset("locuslab/TOFU", f"{split}_perturbed")["train"]
    full = load_dataset("locuslab/TOFU", "full")["train"]
    target_questions = {row["question"] for row in target}
    retain = full.filter(lambda row: row["question"] not in target_questions)
    return {"forget": target, "retain_training": retain, "total_training": full}


# --------------------------- scoring ---------------------------

@torch.no_grad()
def answer_nll(model, rows, tok, device: str = "cuda", bs: int = 16) -> float:
    """Mean per-example NLL over answer tokens only."""
    dl = DataLoader(_QA(rows, tok), batch_size=bs, collate_fn=_collate(tok.pad_token_id))
    lf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="none")
    tot, n = 0.0, 0
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        loss = lf(logits[:, :-1].transpose(1, 2), b["labels"][:, 1:]).sum(-1)
        cnt = (b["labels"][:, 1:] != IGNORE).sum(-1).clamp_min(1)
        tot += float((loss / cnt).sum())
        n += len(cnt)
    return tot / max(n, 1)


@dataclass
class Report:
    """What one edited model scored. ``extra`` carries the level's own sub-metrics."""

    level: str
    forget_nll: float
    retain_nll: float
    forget_delta: Optional[float] = None
    retain_delta: Optional[float] = None
    utility: Optional[float] = None
    utility_retain: Optional[float] = None
    overall: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def selectivity(self) -> Optional[float]:
        """How much of the damage landed on the forget set rather than the retain set."""
        if self.forget_delta is None or not self.retain_delta:
            return None
        return self.forget_delta / self.retain_delta

    def __repr__(self) -> str:
        bits = [f"level={self.level}", f"forget_nll={self.forget_nll:.3f}",
                f"retain_nll={self.retain_nll:.3f}"]
        if self.forget_delta is not None:
            bits.append(f"dF={self.forget_delta:+.3f}")
            bits.append(f"dR={self.retain_delta:+.3f}")
        if self.utility is not None:
            bits.append(f"utility={self.utility:.4f}")
        if self.overall is not None:
            bits.append(f"overall={self.overall:.4f}")
        return "Report(" + ", ".join(bits) + ")"


def _utility(model, tok, splits, n: int = 100, bs: int = 16) -> Dict[str, float]:
    """Model Utility and its parts, via the ported open-unlearning implementation."""
    import scipy.stats as st

    from ._tofu_official import compute_model_utility

    mu, subs = compute_model_utility(
        model, tok,
        splits["retain"].select(range(min(n, len(splits["retain"])))),
        splits["real_authors"].select(range(min(n, len(splits["real_authors"])))),
        splits["world_facts"].select(range(min(n, len(splits["world_facts"])))),
        bs=bs,
    )
    ret = [v for k, v in subs.items() if k.startswith("retain_")]
    gen = [v for k, v in subs.items() if not k.startswith("retain_")]
    return {"utility": float(mu), "utility_retain": float(st.hmean(ret)),
            "utility_general": float(st.hmean(gen)), **{k: float(v) for k, v in subs.items()}}


def evaluate(
    model,
    tok,
    splits: Optional[Dict[str, Any]] = None,
    *,
    level: str = "quick",
    device: str = "cuda",
    baseline: Optional["Report"] = None,
    n_retain: int = 200,
    n_utility: int = 100,
    bs: int = 16,
) -> Report:
    """Score one model. ``level`` is ``"quick"`` | ``"utility"`` | ``"full"`` (see the module docstring).

    Pass ``baseline`` (the same call on the unedited model) to get deltas and, at ``"full"``,
    the rescaled composite.
    """
    if level not in ("quick", "utility", "full"):
        raise ValueError(f"level must be 'quick', 'utility' or 'full', got {level!r}")
    splits = splits if splits is not None else load_splits()
    retain_eval = splits["retain"].select(range(min(n_retain, len(splits["retain"]))))
    rep = Report(level=level,
                 forget_nll=answer_nll(model, splits["forget"], tok, device, bs),
                 retain_nll=answer_nll(model, retain_eval, tok, device, bs))
    if baseline is not None:
        rep.forget_delta = rep.forget_nll - baseline.forget_nll
        rep.retain_delta = rep.retain_nll - baseline.retain_nll
    if level in ("utility", "full"):
        u = _utility(model, tok, splits, n=n_utility, bs=bs)
        rep.utility, rep.utility_retain = u["utility"], u["utility_retain"]
        rep.extra.update(u)
    if level == "full":
        from ._tofu_official import compute_full_tofu

        raw = compute_full_tofu(model, tok, _official_splits(splits), bs=bs)
        rep.extra["raw"] = raw
        if baseline is not None and "raw" in baseline.extra:
            rep.extra["scores"] = _composite(raw, baseline.extra["raw"])
            rep.overall = rep.extra["scores"]["Overall"]
    return rep


def _official_splits(splits: Dict[str, Any]) -> Dict[str, Any]:
    """Key names the ported reference implementation expects."""
    return {"fp": splits["forget"], "ho": splits["holdout"], "rp": splits["retain"],
            "ra": splits["real_authors"], "wf": splits["world_facts"]}


def _composite(raw: Dict[str, Any], finetuned_raw: Dict[str, Any],
               retain_raw: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The paper's rescaled Overall.

    Memorization / Utility / Privacy are each measured relative to two anchors: the fine-tuned
    model (nothing forgotten) and the retain-gold model (nothing to forget). Without the gold
    anchor the fine-tuned one alone is used, which is what a single-model evaluation can offer;
    pass ``retain_raw`` from the ``retain90`` checkpoint for the paper-exact rescaling.
    """
    from ._tofu_official import evaluate_scores

    return evaluate_scores(raw, retain_raw if retain_raw is not None else raw, finetuned_raw)


# --------------------------- edit + search ---------------------------

def collect(
    model,
    tok,
    splits: Dict[str, Any],
    *,
    device: str = "cuda",
    bs: int = 8,
    cache: Optional[Dict[str, str]] = None,
) -> Tuple[Statistics, Statistics, EngramEditor]:
    """Answer-token-masked covariances for the forget set (target) and the full set (reference).

    ``cache={"target": path, "reference": path}`` reuses saved statistics — the covariance does
    not depend on any edit hyper-parameter, so it is collected once and swept over.
    """
    editor = EngramEditor(model, EditorConfig(storage_device=torch.device(device)))
    feats = lambda b: {"input_ids": b["input_ids"].to(device),
                       "attention_mask": b["attention_mask"].to(device)}
    mask = lambda b: b["labels"] != IGNORE

    def one(key: str, rows) -> Statistics:
        path = (cache or {}).get(key)
        if path:
            import os
            if os.path.exists(path):
                return Statistics.load(path, map_location=device)
        dl = DataLoader(_QA(rows, tok), batch_size=bs, collate_fn=_collate(tok.pad_token_id))
        st = editor.collect_statistics(dl, batch_fn=feats, mask_fn=mask)
        if path:
            st.save(path)
        return st

    return one("target", splits["forget"]), one("reference", splits["total"]), editor


def collect_logdet(
    model,
    tok,
    splits: Dict[str, Any],
    *,
    device: str = "cuda",
    bs: int = 8,
    cache: Optional[Dict[str, str]] = None,
) -> Tuple[Statistics, Statistics, Statistics, EngramEditor]:
    """Collect independent answer-token-masked forget, retain, and total statistics."""
    editor = EngramEditor(model, EditorConfig(storage_device=torch.device(device)))
    feats = lambda b: {"input_ids": b["input_ids"].to(device), "attention_mask": b["attention_mask"].to(device)}
    mask = lambda b: b["labels"] != IGNORE

    def one(key: str, rows) -> Statistics:
        path = (cache or {}).get(key)
        if path:
            import os
            if os.path.exists(path):
                return Statistics.load(path, map_location=device)
        dl = DataLoader(_QA(rows, tok), batch_size=bs, collate_fn=_collate(tok.pad_token_id))
        st = editor.collect_statistics(dl, batch_fn=feats, mask_fn=mask)
        if path:
            st.save(path)
        return st

    forget = one("forget", splits["forget"])
    retain = one("retain", splits["retain_training"])
    total = one("total", splits["total_training"])
    return forget, retain, total, editor


def run(
    model,
    tok,
    *,
    split: str = "forget10",
    alpha: Optional[float] = None,
    scale: str = "adaptive",
    reference: Union[str, Statistics, None] = "tofu",
    level: str = "quick",
    device: str = "cuda",
    splits: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, str]] = None,
    n_total: int = 4000,
) -> Dict[str, Any]:
    """Collect, edit and score in one call. Returns ``{"before", "after", "alpha", "scale"}``.

    ``reference``: ``"tofu"`` uses the benchmark's own 4000-sample reference set; a
    :class:`~engram.Statistics` uses that instead (e.g. a self-generated one from
    :func:`engram.generate_corpus`).
    """
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}, got {scale!r}")
    alpha = DEFAULT_ALPHA[scale] if alpha is None else alpha
    splits = splits if splits is not None else load_splits(split, n_total=n_total)
    target, ref, editor = collect(model, tok, splits, device=device, cache=cache)
    if isinstance(reference, Statistics):
        ref = reference
    elif reference not in ("tofu", None):
        raise ValueError(f"reference must be 'tofu' or a Statistics, got {reference!r}")
    before = evaluate(model, tok, splits, level=level, device=device)
    engram = editor.compute_engram_weights(target, ref)
    edited = editor.apply(engram, alpha=alpha, scale=SCALES[scale]()).eval()
    after = evaluate(edited, tok, splits, level=level, device=device, baseline=before)
    return {"before": before, "after": after, "alpha": alpha, "scale": scale,
            "model": edited, "engram": engram}


_OBJECTIVE_NEEDS = {"overall": ("full",), "constrained": ("utility", "full"),
                    "proxy": ("quick", "utility", "full")}


def _check_objective(kind, final_level: str) -> None:
    """Refuse an objective the final level cannot score — before any model is evaluated."""
    if callable(kind):
        return
    if kind not in _OBJECTIVE_NEEDS:
        raise ValueError(f"objective must be 'overall', 'constrained', 'proxy' or a callable, "
                         f"got {kind!r}")
    if final_level not in _OBJECTIVE_NEEDS[kind]:
        raise ValueError(f"objective={kind!r} needs final_level in {_OBJECTIVE_NEEDS[kind]}, "
                         f"got {final_level!r}")


def _objective(rep: Report, kind: str, utility_floor: float,
               base: Report) -> float:
    """Higher is better. ``overall`` is the paper's composite; ``constrained`` maximizes
    forgetting subject to keeping utility above a fraction of the unedited model's; ``proxy``
    is the quick-level selective-forgetting score ``dF - dR`` (rewards aggression — use it to
    rank, not to choose)."""
    if callable(kind):
        return float(kind(rep))
    if kind == "proxy":
        if rep.forget_delta is None or rep.retain_delta is None:
            raise ValueError("objective='proxy' needs a baseline (deltas)")
        return rep.forget_delta - rep.retain_delta
    if kind == "overall":
        if rep.overall is None:
            raise ValueError("objective='overall' needs level='full'")
        return rep.overall
    if kind == "constrained":
        if rep.utility is None or base.utility is None:
            raise ValueError("objective='constrained' needs level='utility' or 'full'")
        if rep.utility < utility_floor * base.utility:
            return -math.inf
        return rep.forget_delta if rep.forget_delta is not None else rep.forget_nll
    raise ValueError(f"objective must be 'overall', 'constrained', 'proxy' or a callable, "
                     f"got {kind!r}")


def search(
    model,
    tok,
    *,
    split: str = "forget10",
    alphas: Sequence[float] = (0.3, 0.6, 0.9, 1.2, 1.5, 2.0),
    scales: Sequence[str] = ("plain", "adaptive"),
    objective: Union[str, Callable[[Report], float]] = "overall",
    utility_floor: float = 0.9,
    coarse_level: str = "quick",
    final_level: str = "full",
    top_k: int = 3,
    reference: Union[str, Statistics, None] = "tofu",
    device: str = "cuda",
    splits: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Find the best ``(alpha, scale)``, cheaply.

    The engram is computed **once**: ``alpha`` only scales the subtraction, so a sweep costs
    evaluations, not extractions. That is why this is coarse-to-fine — every candidate is scored
    at ``coarse_level`` (seconds), and only the ``top_k`` are re-scored at ``final_level``.

    The default objective is the paper's ``Overall``. It contains Utility, so maximizing it
    cannot be won by editing harder — unlike a raw forget-minus-retain score, which rises right
    up to the point where the model collapses. That raw score is available as
    ``objective="proxy"`` for quick-level-only sweeps; treat it as a ranking, not a verdict.
    Objective/level mismatches are rejected before any model is scored.
    """
    _check_objective(objective, final_level)          # fail here, not after an hour of scoring
    for lvl in (coarse_level, final_level):
        if lvl not in ("quick", "utility", "full"):
            raise ValueError(f"level must be 'quick', 'utility' or 'full', got {lvl!r}")
    for sc in scales:
        if sc not in SCALES:
            raise ValueError(f"scale must be one of {sorted(SCALES)}, got {sc!r}")

    splits = splits if splits is not None else load_splits(split)
    target, ref, editor = collect(model, tok, splits, device=device, cache=cache)
    if isinstance(reference, Statistics):
        ref = reference
    base_coarse = evaluate(model, tok, splits, level=coarse_level, device=device)
    engram = editor.compute_engram_weights(target, ref)

    coarse: List[Dict[str, Any]] = []
    for sc in scales:
        for a in alphas:
            edited = editor.apply(engram, alpha=a, scale=SCALES[sc]()).eval()
            rep = evaluate(edited, tok, splits, level=coarse_level, device=device,
                           baseline=base_coarse)
            coarse.append({"alpha": a, "scale": sc, "report": rep})
            if verbose:
                print(f"[tofu.search] alpha={a:<5} scale={sc:<9} {rep}", flush=True)
            del edited
            torch.cuda.empty_cache()

    # rank the coarse pass by selective forgetting, then confirm the survivors properly
    coarse.sort(key=lambda r: (r["report"].forget_delta or 0) - (r["report"].retain_delta or 0),
                reverse=True)
    finalists = coarse[:max(1, top_k)]
    if final_level == coarse_level:
        scored = finalists
        base_final = base_coarse
    else:
        base_final = evaluate(model, tok, splits, level=final_level, device=device)
        scored = []
        for cand in finalists:
            edited = editor.apply(engram, alpha=cand["alpha"],
                                  scale=SCALES[cand["scale"]]()).eval()
            rep = evaluate(edited, tok, splits, level=final_level, device=device,
                           baseline=base_final)
            scored.append({"alpha": cand["alpha"], "scale": cand["scale"], "report": rep})
            if verbose:
                print(f"[tofu.search] final alpha={cand['alpha']} scale={cand['scale']} {rep}",
                      flush=True)
            del edited
            torch.cuda.empty_cache()

    best = max(scored, key=lambda r: _objective(r["report"], objective, utility_floor, base_final))
    return {"best": best, "finalists": scored, "coarse": coarse,
            "baseline": base_final, "engram": engram, "editor": editor}
