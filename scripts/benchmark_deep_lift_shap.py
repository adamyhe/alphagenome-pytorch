#!/usr/bin/env python
"""Validate and benchmark the DeepLIFT/SHAP engines on one GPU.

Runs a preset list of cases. Each case runs one or more engines
(``stock``, ``shared``, ``blockwise``) on the same random sequence and the same
seeded references, each engine in its own subprocess: after a CUDA
out-of-memory error, PyTorch's allocator is unreliable for the rest of the
process, and a fresh process also gives each engine a clean peak-memory count.
Cases that do not fit are recorded as ``oom``, not treated as failures.

For cases with more than one engine, every engine's attributions are compared
with the first engine that finished, in the order stock, shared, blockwise.

Presets:

* ``smoke``: tiny inputs, to check the script runs (any GPU, minutes).
* ``validate``: full 131 kb agreement between engines on the pretrained
  model: 128 bp and 1 bp heads, mouse, aggregates, a window at the sequence
  edge (where the decoder crop is one-sided). Stock at 131 kb needs about
  30 GiB (128 bp) and 80 GiB (1 bp).
* ``sweep``: blockwise throughput over length, ``ref_batch_size`` and track
  count, plus shared and stock where they fit, to choose defaults.

Example::

    python scripts/benchmark_deep_lift_shap.py --weights model.safetensors \\
        --preset validate --out dls_validate_h200.json

Writes one JSON file with the environment, every run's time, peak allocated
and reserved memory, status, and the engine comparisons. Sets
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` for the subprocesses
unless it is already set.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

L_FULL = 131072
ENGINE_ORDER = ("stock", "shared", "blockwise")


def _case(length, head, res, *, engines=("blockwise",), tracks=2, K=4, kbs=4, tbs=None,
          organism=0, aggregates=False, window="center", name=None):
    c = dict(length=length, head=head, res=res, engines=list(engines), tracks=tracks, K=K,
             ref_batch_size=kbs, track_batch_size=tbs, organism=organism,
             aggregates=aggregates, window=window)
    c["name"] = name or (f"{length // 1024}kb {head}@{res} T={tracks}{' agg' if aggregates else ''}"
                         f" K={K}{' ' + window if window != 'center' else ''}")
    return c


ALL = ENGINE_ORDER

PRESETS = {
    "smoke": [
        _case(4096, "dnase", 128, engines=ALL, K=2),
        _case(4096, "rna_seq", 1, engines=ALL, K=2),
        _case(8192, "dnase", 128, engines=("shared", "blockwise"), tracks="all",
              aggregates=True, K=2),
    ],
    "validate": [
        _case(L_FULL, "dnase", 128, engines=ALL),
        _case(L_FULL, "rna_seq", 1, engines=ALL),
        _case(L_FULL, "cage", 1, engines=ALL, organism=1),
        _case(L_FULL, "rna_seq", 1, engines=ALL, window="start"),
        _case(L_FULL, "rna_seq", 1, engines=ALL, window="end"),
        _case(L_FULL, "chip_histone", 128, engines=ALL, tracks=3),
        _case(L_FULL, "dnase", 128, engines=ALL, tracks="all", aggregates=True),
        _case(L_FULL, "rna_seq", 1, engines=("shared", "blockwise"), tracks="all",
              aggregates=True),
    ],
    "sweep": (
        [_case(L, head, res, engines=("blockwise",), K=8, kbs=kbs)
         for L in (16384, 32768, 65536, L_FULL)
         for head, res in (("dnase", 128), ("rna_seq", 1))
         for kbs in (1, 2, 4, 8)]
        + [_case(L, head, res, engines=("stock", "shared"), K=8)
           for L in (16384, 32768, 65536, L_FULL)
           for head, res in (("dnase", 128), ("rna_seq", 1))]
        + [_case(L_FULL, "rna_seq", 1, engines=("blockwise",), tracks=T, K=8, kbs=kbs)
           for T in (1, 8, 32) for kbs in (1, 4)]
        + [_case(L_FULL, "rna_seq", 1, engines=("blockwise",), tracks=1, K=20, kbs=kbs,
                 name=f"128kb rna_seq@1 T=1 K=20 k={kbs}")
           for kbs in (1, 2, 4, 10, 20)]
    ),
}


# -- child: one engine on one case --------------------------------------------


def _window(case, n_bins):
    half = 4 if case["res"] == 128 else 500
    if case["window"] == "start":
        return slice(0, 2 * half)
    if case["window"] == "end":
        return slice(n_bins - 2 * half, n_bins)
    return slice(n_bins // 2 - half, n_bins // 2 + half)


def _run_child(args) -> dict:
    import torch
    import torch.nn.functional as F

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.extensions.attribution import Aggregate, deep_lift_shap_tracks

    case, engine = json.loads(args.child), args.engine
    device = torch.device(args.device)
    model = AlphaGenome.from_pretrained(args.weights, device=device).eval()

    L, res = case["length"], case["res"]
    n_tracks = model.heads[case["head"]].num_tracks
    tracks = list(range(n_tracks)) if case["tracks"] == "all" else list(range(case["tracks"]))
    aggs = None
    if case["aggregates"]:
        T = len(tracks)
        first = (torch.arange(T) < T // 2).float()
        aggs = [Aggregate.weighted(torch.ones(T), name="sum"),
                Aggregate.log_ratio(first, 1 - first, name="lfc")]
    g = torch.Generator().manual_seed(args.seed)
    onehot = F.one_hot(torch.randint(0, 4, (1, L), generator=g), 4).float()

    kw = dict(organism_index=case["organism"], output_type=case["head"], resolution=res,
              track_indices=tracks, random_state=args.seed, engine=engine, device=device,
              aggregates=aggs)
    if engine == "blockwise":
        kw.update(ref_batch_size=case["ref_batch_size"],
                  track_batch_size=case["track_batch_size"])

    # Warm-up (CUDA context, kernels) on a short input.
    warm = 4096
    deep_lift_shap_tracks(model, onehot=onehot[:, :warm], n_shuffles=1,
                          target_slice=_window({**case, "window": "center"}, warm // res), **kw)

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        out = deep_lift_shap_tracks(model, onehot=onehot, n_shuffles=case["K"],
                                    target_slice=_window(case, L // res), **kw)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
    if args.save:
        torch.save(out, args.save)
    return dict(
        status="ok", seconds=seconds,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
        weights_gib=base / 2**30,
        warnings=sorted({str(w.message)[:200] for w in caught}),
    )


# -- parent ---------------------------------------------------------------------


def _environment(args) -> dict:
    import torch

    env = dict(python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, platform=platform.platform(),
               host=platform.node(), seed=args.seed, preset=args.preset,
               alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    try:
        import tangermeme
        env["tangermeme"] = tangermeme.__version__
    except ImportError:
        env["tangermeme"] = None
    if torch.cuda.is_available():
        d = torch.device(args.device)
        p = torch.cuda.get_device_properties(d)
        env.update(gpu=p.name, gpu_memory_gib=p.total_memory / 2**30)
    repo = Path(__file__).resolve().parent.parent
    try:
        env["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
            timeout=10).stdout.strip() or None
        env["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "src"], cwd=repo,
            capture_output=True, text=True, timeout=10).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        env["git_commit"] = None
    return env


def _spawn(args, case, engine, save) -> dict:
    cmd = [sys.executable, __file__, "--weights", args.weights, "--device", args.device,
           "--seed", str(args.seed), "--child", json.dumps(case), "--engine", engine,
           "--save", save or ""]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return dict(status="timeout")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (p.stderr or p.stdout).strip().splitlines()[-15:]
    text = "\n".join(tail)
    oom = "out of memory" in text.lower() or "OutOfMemoryError" in text
    return dict(status="oom" if oom else "error", returncode=p.returncode, stderr=text)


def _compare(a, b) -> dict:
    diff = (a - b).abs()
    scale = float(b.abs().max())
    per_row = [float(((a[i] - b[i]).norm() / b[i].norm().clamp_min(1e-30)))
               for i in range(a.shape[0])]
    return dict(max_abs_over_max=float(diff.max()) / max(scale, 1e-30),
                rel_norm=float((a - b).norm() / b.norm().clamp_min(1e-30)),
                rel_norm_per_output=per_row)


def _fmt(r):
    if r["status"] != "ok":
        return r["status"]
    return f"{r['seconds']:7.1f} s  {r['peak_allocated_gib']:6.2f} GiB ({r['peak_reserved_gib']:.2f} reserved)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True, help="Pretrained .pth or .safetensors.")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="validate")
    ap.add_argument("--out", help="JSON output (default: dls_<preset>_<gpu>.json).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=7200, help="Seconds per engine run.")
    ap.add_argument("--only", help="Run only cases whose name contains this string.")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--engine", help=argparse.SUPPRESS)
    ap.add_argument("--save", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        try:
            result = _run_child(args)
        except Exception as e:  # noqa: BLE001 - reported to the parent
            import torch
            status = "oom" if isinstance(e, torch.OutOfMemoryError) or \
                "out of memory" in str(e).lower() else "error"
            result = dict(status=status, error=f"{type(e).__name__}: {str(e)[:2000]}")
        print("RESULT " + json.dumps(result), flush=True)
        return

    import torch

    env = _environment(args)
    gpu = env.get("gpu", "cpu").replace(" ", "_")
    out_path = Path(args.out or f"dls_{args.preset}_{gpu}.json")
    cases = [c for c in PRESETS[args.preset] if not args.only or args.only in c["name"]]
    print(f"{env.get('gpu')} ({env.get('gpu_memory_gib', 0):.0f} GiB), torch {env['torch']}, "
          f"tangermeme {env['tangermeme']}, commit {str(env.get('git_commit'))[:10]}"
          f"{' (dirty)' if env.get('git_dirty') else ''}; {len(cases)} cases -> {out_path}")

    report = dict(environment=env, cases=[])
    with tempfile.TemporaryDirectory() as tmp:
        for ci, case in enumerate(cases):
            compare = len(case["engines"]) > 1
            print(f"\n[{ci + 1}/{len(cases)}] {case['name']}"
                  + (f"  (k={case['ref_batch_size']})" if "blockwise" in case["engines"] else ""),
                  flush=True)
            runs, saved = {}, {}
            for engine in sorted(case["engines"], key=ENGINE_ORDER.index):
                save = os.path.join(tmp, f"{ci}_{engine}.pt") if compare else None
                r = _spawn(args, case, engine, save)
                runs[engine] = r
                if r["status"] == "ok" and save:
                    saved[engine] = save
                print(f"    {engine:9s} {_fmt(r)}", flush=True)
                if r["status"] == "error":
                    print("      " + (r.get("error") or r.get("stderr", ""))[-600:]
                          .replace("\n", "\n      "))
                if r.get("warnings"):
                    print(f"      warnings: {r['warnings']}")
            comparisons = {}
            if len(saved) > 1:
                ref_engine = next(e for e in ENGINE_ORDER if e in saved)
                ref = torch.load(saved[ref_engine])
                for e, path in saved.items():
                    if e == ref_engine:
                        continue
                    comparisons[f"{e}_vs_{ref_engine}"] = c = _compare(torch.load(path), ref)
                    print(f"    {e} vs {ref_engine}: max|diff|/max = {c['max_abs_over_max']:.2e}"
                          f", rel norm = {c['rel_norm']:.2e}")
            report["cases"].append(dict(case=case, runs=runs, comparisons=comparisons))
            out_path.write_text(json.dumps(report, indent=1))  # keep partial results
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
