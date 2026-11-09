#!/usr/bin/env bash
# Your phone as the camera and microphone: a page on this PC (scan the QR code in the window); the PC does the swap.
exec "$(dirname "$0")/run.sh" --phone "$@"
