#!/usr/bin/env python3
"""Cold-request 4096/8192 x 256/512 sweep for both local quants.

Run with ~/shared-venv-gpu/bin/python bench/quant_sweep.py. Each cell starts a
fresh process with census persistence disabled, so neither prefix reuse nor a
pre-existing hot-expert census gives one checkpoint an advantage. Load time is
reported separately from generation. Logs and JSON are retained even on failure.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from statistics import median


def prompt_ids(tokenizer, length):
    opening = tokenizer.encode(
        "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>"
        "Read the following notes, then follow the instruction at the end.\n",
        add_special_tokens=False,
    )
    passage = (
        "A coastal research station records weather, water temperature, and bird migration. "
        "Each morning the team checks the sensors, labels observations, and compares them with "
        "the previous day's measurements. A careful log helps distinguish seasonal changes "
        "from unusual events. Researchers share equipment and explain their findings clearly.\n"
    )
    middle = tokenizer.encode(passage, add_special_tokens=False)
    ending = tokenizer.encode(
        "\nWrite a detailed fictional story about the researchers solving a mystery at the "
        "station. Include dialogue and continue for at least 1000 words."
        "<|assistant|><think></think>", add_special_tokens=False,
    )
    room = length - len(opening) - len(ending)
    if room < 0:
        raise ValueError("prompt length too small")
    return opening + (middle * (room // len(middle) + 1))[:room] + ending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["~/models/glm53flash", "~/models/glm53flash_abl"])
    parser.add_argument("--binary", default="./build/helios")
    parser.add_argument("--output", default="/tmp/helios-quant-sweep")
    parser.add_argument("--chunk", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    models = [str(Path(p).expanduser().resolve()) for p in args.models]
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in models]
    prompts = {n: prompt_ids(tokenizers[0], n) for n in (4096, 8192)}
    # Both checkpoints must see precisely the same text and ids.
    for tk in tokenizers[1:]:
        for n, ids in prompts.items():
            if prompt_ids(tk, n) != ids:
                raise RuntimeError("tokenizers differ: identical-input comparison is impossible")
    for n, ids in prompts.items():
        (output / f"prompt-{n}.json").write_text(json.dumps(ids))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    state = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    report = {"models": models, "chunk": args.chunk, "repeats": args.repeats,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "binary": str(Path(args.binary).resolve()),
              "revision": revision.stdout.strip() if revision.returncode == 0 else None,
              "working_tree_dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
              "census": "disabled (fresh each cell)",
              "prefix_snap_mb": 0, "ignore_eos": True, "results": []}
    report_path = output / "results.json"
    if args.prepare_only:
        report["status"] = "prepared, no GPU measurements"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Prepared exact 4096/8192-token prompts in {output}")
        return 0
    # Fail before loading tens of GB when the driver cannot be used.
    probe = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                           capture_output=True, text=True)
    if probe.returncode != 0 or len(probe.stdout.strip().splitlines()) < 2:
        report.update(status="blocked: two NVIDIA GPUs unavailable", gpu_probe=probe.stdout + probe.stderr)
        report_path.write_text(json.dumps(report, indent=2))
        print(report["status"])
        return 2
    report["gpus"] = probe.stdout.strip().splitlines()
    environment = os.environ.copy()
    # Experimental paths change the workload; use the documented default engine.
    for key in ("HELIOS_MTP", "HELIOS_LAYER_MAJOR", "HELIOS_MOE_DENSE", "HELIOS_FORCE_SHAPE",
                "HELIOS_SLOTS", "HELIOS_NO_PIN", "HELIOS_DEBUG_SYNC", "HELIOS_PROF"):
        environment.pop(key, None)
    failed = False
    for repeat in range(args.repeats):
        for prefill in (4096, 8192):
            for generation in (256, 512):
                for index, model in enumerate(models):
                    log_path = output / f"model-{index}-{prefill}-{generation}-r{repeat}.log"
                    command = [args.binary, "gen", model, "--cap", "16384", "--chunk", str(args.chunk),
                               "--tokens", str(generation), "--temp", "0", "--ignore-eos",
                               "--prefix-snap-mb", "0", "--census-file", "",
                               "--prompt-ids", str(output / f"prompt-{prefill}.json")]
                    started = time.monotonic()
                    timed_out = False
                    with log_path.open("w") as log:
                        try:
                            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                  env=environment, timeout=args.timeout)
                            returncode = proc.returncode
                        except subprocess.TimeoutExpired:
                            timed_out, returncode = True, -1
                    row = {"model": model, "prefill": prefill, "generation": generation,
                           "repeat": repeat, "log": str(log_path), "returncode": returncode,
                           "process_seconds": time.monotonic() - started, "timed_out": timed_out}
                    log = log_path.read_text(errors="replace")
                    matches = re.findall(r"^\[bench-json\] (.+)$", log, re.MULTILINE)
                    if matches:
                        row.update(json.loads(matches[-1]))
                    row["valid"] = (returncode == 0 and row.get("prompt_tokens") == prefill and
                                    row.get("prefill_tokens") == prefill and
                                    row.get("generated_tokens") == generation and
                                    row.get("decode_steps") == generation - 1 and
                                    row.get("prefill_ms", 0) > 0 and row.get("decode_ms", 0) > 0 and
                                    "DEGRADED" not in log and "MISMATCH" not in log and
                                    "arena pin failed" not in log)
                    if row["valid"]:
                        row["prefill_tps"] = prefill * 1000 / row["prefill_ms"]
                        row["decode_tps"] = row["decode_steps"] * 1000 / row["decode_ms"]
                        decode_wall = row["wall_seconds"] - row["prefill_ms"] / 1000
                        row["emitted_decode_tps"] = generation / decode_wall
                    else:
                        failed = True
                    report["results"].append(row)
                    report["status"] = "running"
                    report_path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(row), flush=True)
    report["status"] = "failed" if failed else "complete"
    report["comparisons"] = []
    if not failed:
        for prefill in (4096, 8192):
            for generation in (256, 512):
                def rates(model):
                    rows = [r for r in report["results"] if r["model"] == model and
                            r["prefill"] == prefill and r["generation"] == generation]
                    return {key: median(r[key] for r in rows) for key in
                            ("prefill_tps", "decode_tps", "emitted_decode_tps", "wall_seconds")}
                baseline = rates(models[0])
                for model in models[1:]:
                    target = rates(model)
                    report["comparisons"].append({
                        "baseline": models[0], "target": model,
                        "prefill": prefill, "generation": generation,
                        "baseline_medians": baseline, "target_medians": target,
                        "prefill_speed_ratio": target["prefill_tps"] / baseline["prefill_tps"],
                        "decode_speed_ratio": target["decode_tps"] / baseline["decode_tps"],
                        "emitted_decode_speed_ratio": target["emitted_decode_tps"] / baseline["emitted_decode_tps"],
                    })
    report_path.write_text(json.dumps(report, indent=2))
    for comparison in report["comparisons"]:
        print(json.dumps(comparison), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
