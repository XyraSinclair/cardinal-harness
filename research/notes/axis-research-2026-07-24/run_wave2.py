#!/usr/bin/env python3
"""Wave-2 runner: 8 axes x 3 models, decoy-planted sets under wave2/.

Reads wave2/prompts.json ({axis_key: wording}), sorts wave2/<axis>.txt with
each model, writes wave2/sort-<axis>-<model>.json. Skips existing outputs so
the script is resumable.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
W2 = HERE / "wave2"
CARDINAL = Path.home() / "Projects/cardinal-harness/target/release/cardinal"

MODELS = {
    "opus46": "anthropic/claude-opus-4.6",
    "gpt56sol": "openai/gpt-5.6-sol",
    "mini54": "openai/gpt-5.4-mini",
}

def main() -> int:
    prompts = json.loads((W2 / "prompts.json").read_text())
    failures = 0
    for axis_key, wording in prompts.items():
        probe = W2 / f"{axis_key}.txt"
        if not probe.exists():
            print(f"MISSING probe set {probe.name}", flush=True)
            failures += 1
            continue
        for model_key, model_slug in MODELS.items():
            out = W2 / f"sort-{axis_key}-{model_key}.json"
            if out.exists():
                print(f"skip {out.name}", flush=True)
                continue
            cmd = [
                str(CARDINAL), "sort", str(probe),
                "--by", wording,
                "--model", model_slug,
                "--budget", "24",
                "--format", "json",
                "--scores",
            ]
            print(f"RUN {axis_key} x {model_key}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(CARDINAL.parent.parent.parent))
            if proc.returncode != 0:
                print(f"FAIL {axis_key} x {model_key}: {proc.stderr[-1500:]}",
                      flush=True)
                failures += 1
                continue
            out.write_text(proc.stdout)
            print(f"OK -> {out.name}", flush=True)
    print(f"DONE failures={failures}", flush=True)
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
