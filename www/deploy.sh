#!/usr/bin/env bash
# Deploy llmsorting.com: rsync www/ (+ ../PROGRAM.md) to colo2:/srv/llmsorting
# and verify the live page through Cloudflare. No build step.
set -euo pipefail
cd "$(dirname "$0")"

HOST=colo2
ROOT=/srv/llmsorting
STAMP=$(/usr/bin/git rev-parse --short HEAD 2>/dev/null || echo untracked)

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp index.html "$tmp"/index.html
cp ../PROGRAM.md "$tmp"/PROGRAM.md
printf '%s\n' "$STAMP" > "$tmp"/.deploy-commit

rsync -az --delete --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "$tmp"/ "$HOST:$ROOT/"
# rsync applies --chmod to the transferred tree but the root dir keeps the
# temp dir's 0700 (mktemp) — caddy must be able to traverse it.
ssh "$HOST" "chmod 0755 $ROOT"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://llmsorting.com/)
live=$(curl -s --max-time 20 https://llmsorting.com/.deploy-commit || true)
echo "deployed $STAMP -> https://llmsorting.com/ (HTTP $code, live commit ${live:-?})"
[[ "$code" == "200" && "$live" == "$STAMP" ]] || { echo "verify FAILED" >&2; exit 1; }
