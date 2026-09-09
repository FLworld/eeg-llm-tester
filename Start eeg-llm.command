#!/usr/bin/env bash
# Double-clickable macOS launcher. Finder opens this in Terminal; it just runs the shared script.
cd "$(dirname "$0")"
bash ./start-eeg-llm.sh "$@"
status=$?
echo ""
if [ "$status" -ne 0 ]; then echo "Setup stopped. The error and next steps are above."; fi
if [ -t 0 ]; then read -r -p "Press Enter to close this window... " _; fi
exit "$status"
