#!/usr/bin/env bash
# Double-clickable macOS launcher. Finder opens this in Terminal; it just runs the shared script.
cd "$(dirname "$0")"
bash ./start-eeg-llm.sh
echo ""
echo "eeg-llm stopped. You can close this window."
