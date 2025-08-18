#!/usr/bin/env bash
# Helper that invokes the PowerShell signing helper using pwsh.
# Usage:
#   ./scripts/sign_powershell.sh /path/to/script.ps1 /path/to/cert.pfx [pfx-password]

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <script-path> <pfx-path> [pfx-password]" >&2
  exit 2
fi

SCRIPT_PATH="$1"
PFX_PATH="$2"
PFX_PWD="${3-}"

PS_CMD=""
if command -v pwsh >/dev/null 2>&1; then
  PS_CMD=pwsh
elif command -v powershell >/dev/null 2>&1; then
  PS_CMD=powershell
fi

if [ -z "$PS_CMD" ]; then
  echo "Neither 'pwsh' nor 'powershell' found in PATH. Install PowerShell Core or run the PowerShell script directly." >&2
  exit 127
fi

if [ -n "$PFX_PWD" ]; then
  $PS_CMD -Command "& { ./scripts/sign_powershell.ps1 -ScriptPath '$SCRIPT_PATH' -PfxPath '$PFX_PATH' -PfxPassword '$PFX_PWD' }"
else
  $PS_CMD -Command "& { ./scripts/sign_powershell.ps1 -ScriptPath '$SCRIPT_PATH' -PfxPath '$PFX_PATH' }"
fi
