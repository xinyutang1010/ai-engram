#!/usr/bin/env python
"""One resumable stage of the TOFU LogDet experiment."""
from __future__ import annotations
import argparse, json, os, socket, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram import EngramEditor, EditorConfig, count_ratio, uniform
from engram.benchmarks import tofu
from engram.benchmarks import _tofu_official

def load_model(path):
    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

def jdefault(v):
    if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
    if hasattr(v, "tolist"): return v.tolist()
    if hasattr(v, "item"): return v.item()
    raise TypeError(type(v).__name__)

def report_dict(rep):
    return {"level": rep.level, "forget_nll": rep.forget_nll, "retain_nll": rep.retain_nll,
            "forget_delta": rep.forget_delta, "retain_delta": rep.retain_delta,
            "utility": rep.utility, "utility_retain": rep.utility_retain,
            "overall": rep.overall, "extra": rep.extra}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("anchors", "baseline", "method"), required=True)
    ap.add_argument("--mode", choices=("two_sided", "one_sided", "retain_optimal"))
    ap.add_argument("--anchor-path", default="/srv/scratch/z5773483/datasets/TOFU/logdet_anchors.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="/srv/scratch/z5773483/models/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--retain-model", default="/srv/scratch/z5773483/models/tofu_Llama-3.2-1B-Instruct_retain90")
    ap.add_argument("--stats-dir", default="/srv/scratch/z5773483/datasets/TOFU/logdet_stats")
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank-max", type=int, default=0)
    ap.add_argument("--logdet-fraction", type=float, default=0.95)
    ap.add_argument("--gamma-floor", type=float, default=1.0)
    ap.add_argument("--update-ridge-eta", type=float, default=1e-4)
    ap.add_argument("--retain-lambda", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="global edit strength; defaults to 0.6 for all stage edits")
    ap.add_argument("--max-update-ratio", type=float, default=None,
                    help="optional per-layer ||delta W||/||W|| trust-region cap")
    args = ap.parse_args()
    started = time.monotonic()
    alpha = 0.6 if args.alpha is None else args.alpha
    config = {**vars(args), "alpha": alpha, "job_id": os.environ.get("PBS_JOBID"),
              "hostname": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
              "torch_version": torch.__version__}
    print(json.dumps({"event": "start", "config": config}), flush=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    splits = tofu.load_splits("forget10", n_total=4000, seed=0)
    official = tofu._official_splits(splits)
    if args.stage == "anchors":
        base = load_model(args.model)
        before = tofu.evaluate(base, tok, splits, level="full", device="cuda", bs=args.eval_batch_size)
        del base; torch.cuda.empty_cache()
        gold = load_model(args.retain_model)
        gold_raw = _tofu_official.compute_full_tofu(gold, tok, official, bs=args.eval_batch_size)
        torch.save({"before_raw": before.extra["raw"], "gold_raw": gold_raw}, args.anchor_path)
        payload = {"stage": "anchors", "before": report_dict(before), "anchor_path": args.anchor_path}
    else:
        anchors = torch.load(args.anchor_path, map_location="cpu", weights_only=False)
        base = load_model(args.model)
        cache = {k: str(Path(args.stats_dir) / f"{k}.pt") for k in ("forget", "retain", "total")}
        train = tofu.load_logdet_training_splits("forget10")
        forget, retain, total, editor = tofu.collect_logdet(base, tok, train, device="cuda", bs=args.batch_size, cache=cache)
        if args.stage == "baseline":
            result = editor.compute_engram_weights(forget, total)
            edited = editor.apply(result, alpha=(0.6 if args.alpha is None else args.alpha),
                                  scale=count_ratio(1.0), max_update_ratio=args.max_update_ratio).eval()
            label = "baseline_engram"
        else:
            if args.mode is None: raise ValueError("--mode is required for stage=method")
            result = editor.compute_logdet_engram_weights(forget, retain, total,
                rank_max=(args.rank_max or None), logdet_fraction=args.logdet_fraction,
                gamma_floor=args.gamma_floor, update_ridge_eta=args.update_ridge_eta,
                update_mode=args.mode, retain_lambda=args.retain_lambda)
            if args.mode == "retain_optimal":
                # retain-optimal already contains n/N and r/N in A and R.  Do not apply
                # count_ratio a second time; alpha remains the independent edit strength.
                edited = editor.apply(result, alpha=alpha, scale=uniform(),
                                      max_update_ratio=args.max_update_ratio).eval()
            else:
                edited = editor.apply(result, alpha=alpha, scale=count_ratio(1.0),
                                      max_update_ratio=args.max_update_ratio).eval()
            label = args.mode
        # Record actual applied strength, rather than comparing unscaled projections.
        for name, info in result.layers.items():
            factor = 1.0 if args.mode == "retain_optimal" else info.n / info.N
            projection_norm = float(info.projection.float().norm())
            proposed_ratio = abs(alpha * factor) * projection_norm / max(info.weight_fro, 1e-30)
            applied_ratio = (min(proposed_ratio, args.max_update_ratio)
                             if args.max_update_ratio is not None else proposed_ratio)
            mixed = ((forget.count[name] / total.count[name]) * forget[name]
                     + (retain.count[name] / total.count[name]) * retain[name])
            relative_covariance_error = float((total[name] - mixed).norm() /
                                              total[name].norm().clamp_min(1e-30))
            result.diagnostics.setdefault(name, {}).update({
                "projection_fro": projection_norm, "weight_fro": info.weight_fro,
                "proposed_update_ratio": proposed_ratio, "applied_update_ratio": applied_ratio,
                "update_clipped": applied_ratio < proposed_ratio,
                "forget_count": forget.count[name], "retain_count": retain.count[name],
                "total_count": total.count[name],
                "total_vs_merged_relative_error": relative_covariance_error,
            })
        checkpoint = {"config": config, "diagnostics": result.diagnostics}
        out.with_suffix(".diagnostics.json").write_text(json.dumps(checkpoint, indent=2, default=jdefault))
        print(json.dumps({"event": "evaluation_start", "elapsed_seconds": time.monotonic() - started,
                          "edited_layers": len(result.layers)}), flush=True)
        rep = tofu.evaluate(edited, tok, splits, level="full", device="cuda", bs=args.eval_batch_size)
        scores = _tofu_official.evaluate_scores(rep.extra["raw"], anchors["gold_raw"], anchors["before_raw"])
        payload = {"stage": args.stage, "mode": label, "report": report_dict(rep), "paper_scores": scores}
        if args.stage == "method": payload["diagnostics"] = result.diagnostics
        del edited, result, base; torch.cuda.empty_cache()
    payload["config"] = config
    payload["elapsed_seconds"] = time.monotonic() - started
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=jdefault))
    temporary.replace(out)
    print(json.dumps({k: payload[k] for k in ("stage", "mode", "paper_scores") if k in payload}, indent=2, default=jdefault), flush=True)

if __name__ == "__main__": main()
