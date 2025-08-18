#!/usr/bin/env bash
# Run the signing Makefile target reading variables from the environment.
# Designed to be invoked inside the devcontainer terminal or by the VS Code task.
# Requires the environment variables:
#   SCRIPT - path to the PowerShell script to sign (e.g. scripts/set_log_env.ps1)
#   PFX    - path to the .pfx certificate file
# Optional:
#   PFX_PWD - PFX password (plain-text). If omitted the signer will prompt.

set -euo pipefail

if [ -z "${SCRIPT:-}" ]; then
  echo "ERROR: environment variable SCRIPT is not set. Example: export SCRIPT=scripts/set_log_env.ps1" >&2
  exit 2
fi

if [ -z "${PFX:-}" ]; then
  echo "ERROR: environment variable PFX is not set. Example: export PFX=/path/to/cert.pfx" >&2
  exit 2
fi

# PFX_PWD is optional
PFX_PWD_VAL="${PFX_PWD:-}"

echo "Invoking signing: SCRIPT=$SCRIPT PFX=$PFX"

make sign-powershell SCRIPT="$SCRIPT" PFX="$PFX" PFX_PWD="$PFX_PWD_VAL"
