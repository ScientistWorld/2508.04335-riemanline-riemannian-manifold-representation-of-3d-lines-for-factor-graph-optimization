#!/bin/bash
# Temporary audit entrypoint; restored before final delivery.
set -euo pipefail
cd /home/user
bash redteam/run_all.sh
