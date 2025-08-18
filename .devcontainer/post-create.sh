#!/usr/bin/env bash
set -euo pipefail

CONTAINER_WS="${CONTAINER_WORKSPACE_FOLDER:-${containerWorkspaceFolder:-/workspace/parquet_segmenter}}"


# Create and activate venv
python3.13 -m venv "${CONTAINER_WS}/.venv" || true
"${CONTAINER_WS}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel || true
"${CONTAINER_WS}/.venv/bin/python" -m pip install pylint mypy pytest pytest-cov psycopg2-binary || true

# Install copilot CLI if gh is available
if command -v gh >/dev/null 2>&1; then
  gh extension install github/copilot-cli || true
fi

# Ensure vscode has zsh as shell if sudo is available
if command -v sudo >/dev/null 2>&1; then
  sudo chsh -s /usr/bin/zsh vscode || true
fi

# If the docker socket is mounted, create a group matching its GID and add vscode to it
if [ -S /var/run/docker.sock ]; then
  DOCKER_GID=$(stat -c '%g' /var/run/docker.sock) || true
  if [ -n "${DOCKER_GID}" ]; then
    # create a group with that GID if it doesn't exist
    if ! getent group "${DOCKER_GID}" >/dev/null 2>&1; then
      sudo groupadd -g "${DOCKER_GID}" dockerhost || true
    fi
    # determine a group name for the GID and add vscode to it
    GROUP_NAME=$(getent group "${DOCKER_GID}" | cut -d: -f1 || true)
    if [ -n "${GROUP_NAME}" ]; then
      sudo usermod -aG "${GROUP_NAME}" vscode || true
    else
      sudo usermod -aG dockerhost vscode || true
    fi
  fi
fi

# End

