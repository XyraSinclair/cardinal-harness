#!/usr/bin/env python3
"""Tier-divergence micro-probe: 3 axes x 3 models over the 12-item decoy set.

Big-model-smell hypothesis, pre-registered: frontier models agree with each
other and dump the decoys (cosmic-slop item 2, performed-intensity item 4);
the small model rides the vocabulary and ranks decoys high. Divergence should
concentrate on the decoys for the two deep axes and vanish on the clarity
control.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CARDINAL = Path.home() / "Projects/cardinal-harness/target/release/cardinal"
PROBE = HERE / "probe-set.txt"

AXES = {
    "end_of_time": "connection to the end of time - how much this text genuinely bears on the far future and ultimate stakes",
    "ultrawattagedness": "ultrawattagedness - the raw live intellectual and agentic wattage actually behind this text",
    "clarity": "clarity - how clear and easy to understand this text is",
}
MODELS = {
    "opus46": "anthropic/claude-opus-4.6",
    "gpt56sol": "openai/gpt-5.6-sol",
    "mini54": "openai/gpt-5.4-mini",
}

def main() -> None:
    for axis_key, axis_prompt in AXES.items():
        for model_key, model_slug in MODELS.items():
            out = HERE / f"sort-{axis_key}-{model_key}.json"
            if out.exists():
                print(f"skip {out.name} (exists)", flush=True)
                continue
            cmd = [
                str(CARDINAL), "sort", str(PROBE),
                "--by", axis_prompt,
                "--model", model_slug,
                "--budget", "24",
                "--format", "json",
                "--scores",
            ]
            print(f"RUN {axis_key} x {model_key}", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(CARDINAL.parent.parent.parent))
            if proc.returncode != 0:
                print(f"FAIL {axis_key} x {model_key}: {proc.stderr[-2000:]}",
                      flush=True)
                continue
            out.write_text(proc.stdout)
            print(f"OK -> {out.name}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
