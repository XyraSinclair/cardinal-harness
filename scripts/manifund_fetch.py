#!/usr/bin/env python3
"""Fetch the full Manifund public corpus (projects + comments) to JSONL.

Manifund exposes a clean public API (crawl approved by Austin Chen, 2026-07-13):
  GET https://manifund.org/api/v0/projects?before=<created_at>   (100/page)
  GET https://manifund.org/api/v0/comments?before=<created_at>   (100/page)

Output: data/manifund/projects.jsonl, data/manifund/comments.jsonl
Re-running refetches from scratch (the corpus is small: ~1.3k projects).
"""

import json
import pathlib
import time
import urllib.request

BASE = "https://manifund.org/api/v0"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "manifund"
UA = "cardinal-harness-research (contact: xyraward@gmail.com; approved by Austin)"


def paginate(surface: str, out_path: pathlib.Path) -> int:
    before = None
    seen: set[str] = set()
    total = 0
    with out_path.open("w") as out:
        while True:
            url = f"{BASE}/{surface}"
            if before:
                url += f"?before={before}"
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                batch = json.load(r)
            if not batch:
                break
            new = 0
            for row in batch:
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                out.write(json.dumps(row) + "\n")
                new += 1
            total += new
            before = batch[-1]["created_at"]
            print(f"{surface}: +{new} (total {total}) before={before}")
            if new == 0:
                break
            time.sleep(0.5)
    return total


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_projects = paginate("projects", OUT_DIR / "projects.jsonl")
    n_comments = paginate("comments", OUT_DIR / "comments.jsonl")
    print(f"DONE projects={n_projects} comments={n_comments} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
