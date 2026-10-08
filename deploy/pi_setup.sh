#!/usr/bin/env bash
# One-time setup of a Raspberry Pi 5 for Saarthi. Run it as a normal user (it asks for sudo for the packages).
#
#   ./deploy/pi_setup.sh
#
# It checks the machine, installs the system packages and the tool `uv`.
# Then run ./start. The first start installs the Python packages and downloads the models (about 3.8 GB).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

say() { printf '%s\n' "$*"; }
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -ne 0 ] || fail "Run this script as a normal user, not as root."

# ---- The machine ----
ARCH="$(uname -m)"
[ "$ARCH" = "aarch64" ] || say "Warning: the machine is $ARCH, not aarch64 (a 64-bit Raspberry Pi OS). The packages for this project need a 64-bit system."
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
say "Machine: ${MODEL}"
MEMORY_MB="$(awk '/MemTotal/ {printf "%d", $2 / 1024}' /proc/meminfo)"
say "Memory: ${MEMORY_MB} MB"
[ "$MEMORY_MB" -ge 7000 ] || say "Warning: Saarthi was planned for 8 GB. It loads 7 models. With less memory it can stop with an out-of-memory error."
FREE_GB="$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}')"
say "Free disk space here: ${FREE_GB} GB"
[ "$FREE_GB" -ge 15 ] || fail "Less than 15 GB of free disk space. The packages and the models need about 10 GB, and the pictures need more."

# ---- The system packages ----
# libgl1 and libglib2.0-0: the package opencv (a dependency of the broad detector) needs them. openssl: the local certificate.
# avahi-daemon: the name hostname.local, so a phone can use a name and not an address.
say "Installing the system packages (sudo)..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends curl ca-certificates git openssl python3 libgl1 libglib2.0-0 avahi-daemon

# ---- uv ----
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

say ""
say "Setup is done. Now run:  ./start"
say "To start Saarthi when the Pi starts, see docs/Raspberry Pi.md (the service)."
