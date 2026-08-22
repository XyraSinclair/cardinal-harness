#!/bin/bash
set -a; source ~/.config/scry-secrets/openrouter.env; set +a
export OPENROUTER_DISABLE_REASONING=1
export OPENROUTER_TIMEOUT_SECONDS=180
export OPENROUTER_PROVIDER_JSON="{\"order\": [\"parasail\", \"coreweave\", \"digitalocean\", \"streamlake\", \"sail-research\"], \"allow_fallbacks\": false}"
cd ~/build/llmsort/research
BIN=../target/release/examples/setwise_cached
OUT=artifacts/live/best-worst-2026-08-22/live
ATTRS="impact_per_dollar,theory_of_change,fit for a funder who wants cheap high-leverage AI safety field-building"
run(){ mode=$1; m=$2; seed=$3; tag=$4; echo "=== $mode m=$m seed=$seed $(date -u +%FT%TZ)"; $BIN --answer $mode --model deepseek/deepseek-v4-flash --ks 8 --n 24 --presentations $m --seed $seed --attrs "$ATTRS" --spend-cap-usd 1.0 --out-dir $OUT/$tag 2>&1 | grep -v "^pairwise"; }
run order 6 17 order-m6
run bw 6 17 bw-m6
run order 3 23 order-m3-s23
run bw 3 23 bw-m3-s23
echo LIVE2_DONE
