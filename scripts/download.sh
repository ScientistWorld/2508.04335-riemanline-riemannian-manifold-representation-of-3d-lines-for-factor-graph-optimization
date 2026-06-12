#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: less than 1 MB
#
# This reproduction has no external downloads. The script only ensures the
# workspace-local downloads directory exists for compatibility with the
# standard environment layout.

set -e

cd /home/user

# The benchmark data is procedural and generated locally by scripts/run.sh or
# scripts/reproduce.sh. No external datasets or model weights are required.
mkdir -p data/downloads
