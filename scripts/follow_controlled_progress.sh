#!/usr/bin/env bash
set -euo pipefail
root=${1:?Remote experiment root required}
ssh llm01 "tail --pid=\$(cat '$root/full.pid') --retry -n +1 -F \
  '$root/full.log' \
  '$root/results/full/noul.log' \
  '$root/results/full/unsafe_up.log' \
  '$root/results/full/bidirectional.log' \
  '$root/results/full/noul_evaluation.log' \
  '$root/results/full/unsafe_up_evaluation.log' \
  '$root/results/full/bidirectional_evaluation.log'"
