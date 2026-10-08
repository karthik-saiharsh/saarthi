#!/usr/bin/env bash
# Install Saarthi as a service that starts when the machine starts. Run it as a normal user.
#
#   ./deploy/install_service.sh            install and enable
#   ./deploy/install_service.sh --remove   stop, disable and remove
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
PROJECT="$PWD"
UNIT=/etc/systemd/system/saarthi.service

[ "$(id -u)" -ne 0 ] || { echo "Run this script as a normal user, not as root." >&2; exit 1; }

if [ "${1:-}" = "--remove" ]; then
  sudo systemctl disable --now saarthi.service || true
  sudo rm -f "$UNIT"
  sudo systemctl daemon-reload
  echo "The service is removed."
  exit 0
fi

# The project path must not contain a character that the unit file reads in a special way.
case "$PROJECT" in
  *[!A-Za-z0-9_./-]*) echo "The project path has a space or a special character: $PROJECT. Move the project to a plain path." >&2; exit 1 ;;
esac

sed -e "s|__USER__|$(id -un)|" -e "s|__PROJECT__|$PROJECT|g" deploy/saarthi.service | sudo tee "$UNIT" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now saarthi.service
echo "The service runs. Watch it with:  journalctl -u saarthi -f"
echo "Stop it with:  sudo systemctl stop saarthi   (needed before ./clear)"
