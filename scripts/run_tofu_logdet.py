#!/usr/bin/env python
"""Offline TOFU baseline + guarded LogDet-subspace experiment.

The script deliberately collects independent target/retain/total statistics.  The retain
rows are derived from full TOFU by removing target questions, rather than using the
``retain_perturbed`` evaluation split as training statistics.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engram import EngramEditor, EditorConfig, count_ratio, uniform
from engram.benchmarks import tofu
from engram.benchmarks import _tofu_official


def _json_default(value):
    """Serialize diagnostics and official TOFU arrays without losing their values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    # numpy is an optional TOFU dependency, imported only when full scoring is used.
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def report_dict(rep):
    return {"level": rep.level, "forget_nll": rep.forget_nll, "retain_nll": rep.retain_nll,
            "forget_delta": rep.forget_delta, "retain_delta": rep.retain_delta,
            "utility": rep.utility, "utility_retain": rep.utility_retain,
            "overall": rep.overall, "extra": rep.extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/srv/scratch/z5773483/models/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--retain-model", default="/srv/scratch/z5773483/models/tofu_Llama-3.2-1B-Instruct_retain90")
    ap.add_argument("--stats-dir", default="/srv/scratch/z5773483/datasets/TOFU/logdet_stats")
    ap.add_argument("--output", default="/srv/scratch/z5773483/datasets/TOFU/logdet_results.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--level", choices=("quick", "utility", "full"), default="full")
    ap.add_argument("--rank-max", type=int, default=0,
                    help="Optional resource cap; 0 means no artificial cap (rank is chosen by coverage).")
    ap.add_argument("--eta-select", type=float, default=1e-4)
    ap.add_argument("--gamma-rtol", type=float, default=1e-6)
    ap.add_argument("--logdet-fraction", type=float, default=0.95)
    ap.add_argument("--gamma-floor", type=float, default=1.0)
    ap.add_argument("--update-ridge-eta", type=float, default=1e-4)
    ap.add_argument("--update-mode", choices=("two_sided", "one_sided", "retain_optimal", "all"), default="retain_optimal")
    ap.add_argument("--retain-lambda", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(args.stats_dir); stats_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.json")
    partial = {"config": vars(args), "status": "running", "completed_stages": {}}

    def checkpoint(stage, result):
        partial["completed_stages"][stage] = result
        partial["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = partial_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(partial, indent=2, default=_json_default))
        temporary.replace(partial_path)

    def run_stage(stage, fn):
        started = monotonic()
        print(f"STAGE_START {stage} {datetime.now(timezone.utc).isoformat()}", flush=True)
        result = fn()
        print(f"STAGE_DONE {stage} seconds={monotonic()-started:.1f}", flush=True)
        return result

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def load(path):
        return AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to(device).eval()

    eval_splits = tofu.load_splits("forget10", n_total=4000, seed=0)
    train_splits = tofu.load_logdet_training_splits("forget10")
    base = load(args.model)
    cache = {k: str(stats_dir / f"{k}.pt") for k in ("forget", "retain", "total")}
    forget, retain, total, editor = tofu.collect_logdet(
        base, tok, train_splits, device=str(device), bs=args.batch_size, cache=cache)
    print("STAT_COUNTS", {"forget": next(iter(forget.count.values())),
                           "retain": next(iter(retain.count.values())),
                           "total": next(iter(total.count.values()))}, flush=True)

    before = run_stage("before_eval", lambda: tofu.evaluate(base, tok, eval_splits, level=args.level, device=str(device), bs=args.eval_batch_size))
    checkpoint("before", report_dict(before))
    baseline_result = run_stage("baseline_projection", lambda: editor.compute_engram_weights(forget, total))
    baseline = editor.apply(baseline_result, alpha=0.6, scale=count_ratio(1.0)).eval()
    base_after = run_stage("baseline_eval", lambda: tofu.evaluate(baseline, tok, eval_splits, level=args.level,
                               device=str(device), baseline=before, bs=args.eval_batch_size))
    checkpoint("baseline_engram", report_dict(base_after))
    del baseline, baseline_result
    torch.cuda.empty_cache()

    modes = ["two_sided", "one_sided", "retain_optimal"] if args.update_mode == "all" else [args.update_mode]
    mode_reports = {}
    mode_diagnostics = {}
    mode_raw = {}
    for mode in modes:
        logdet_result = run_stage(f"{mode}_projection", lambda: editor.compute_logdet_engram_weights(
            forget, retain, total, rank_max=(args.rank_max or None),
            eta_select=args.eta_select, gamma_rtol=args.gamma_rtol,
            logdet_fraction=args.logdet_fraction,
            gamma_floor=args.gamma_floor,
            update_ridge_eta=args.update_ridge_eta,
            update_mode=mode, retain_lambda=args.retain_lambda))
        checkpoint(f"{mode}_diagnostics", logdet_result.diagnostics)
        if mode == "retain_optimal":
            # A and R already include n/N and r/N in the closed form. Applying
            # count_ratio here would double-scale the update.
            edited = editor.apply(logdet_result, alpha=1.0, scale=uniform()).eval()
        else:
            edited = editor.apply(logdet_result, alpha=0.6, scale=count_ratio(1.0)).eval()
        after = run_stage(f"{mode}_eval", lambda: tofu.evaluate(edited, tok, eval_splits, level=args.level,
                              device=str(device), baseline=before, bs=args.eval_batch_size))
        checkpoint(mode, report_dict(after))
        mode_reports[mode] = report_dict(after)
        mode_diagnostics[mode] = logdet_result.diagnostics
        if args.level == "full":
            mode_raw[mode] = after.extra.get("raw")
        del edited, logdet_result, after
        torch.cuda.empty_cache()

    payload = {
        "config": vars(args),
        "before": report_dict(before),
        "baseline_engram": report_dict(base_after),
        "logdet": mode_reports[modes[0]],
        "diagnostics": mode_diagnostics[modes[0]],
    }
    if args.update_mode == "all":
        payload["logdet_by_mode"] = mode_reports
        payload["diagnostics_by_mode"] = mode_diagnostics
    if args.level == "full":
        # Reproduce the paper's two-anchor metric rescaling as well.
        gold = load(args.retain_model)
        raw_ft = before.extra.get("raw")
        raw_baseline = base_after.extra.get("raw")
        raw_gold = run_stage("gold_eval", lambda: _tofu_official.compute_full_tofu(gold, tok, tofu._official_splits(eval_splits), bs=args.eval_batch_size))
        checkpoint("raw_gold", raw_gold)
        scores = {
            "baseline_engram": _tofu_official.evaluate_scores(raw_baseline, raw_gold, raw_ft),
        }
        for mode in modes:
            scores[mode] = _tofu_official.evaluate_scores(mode_raw[mode], raw_gold, raw_ft)
        payload["paper_scores"] = scores
        del gold
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    partial["status"] = "complete"
    checkpoint("paper_scores", payload.get("paper_scores", {}))
    print(json.dumps({k: payload[k] for k in ("config", "paper_scores") if k in payload}, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
