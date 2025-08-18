Signing helper scripts

This folder contains helper scripts to set logging environment variables and
optionally sign the PowerShell helper script.

Files:
- set_log_env.sh — Bash script to export logging env vars (source it):
  source scripts/set_log_env.sh --enable-file --path /tmp/ps.log --level DEBUG

- set_log_env.ps1 — PowerShell equivalent (dot-source to export in PS):
  . .\scripts\set_log_env.ps1 -EnableFile -Path C:\tmp\ps.log -Level DEBUG

- sign_powershell.ps1 — PowerShell script to sign a .ps1 using a PFX file.
  Example (run in PowerShell):
    ./scripts/sign_powershell.ps1 -ScriptPath .\scripts\set_log_env.ps1 -PfxPath C:\secrets\sign.pfx

- sign_powershell.sh — Bash wrapper that invokes the PowerShell signer using
  `pwsh` (PowerShell Core). Example:
    ./scripts/sign_powershell.sh scripts/set_log_env.ps1 /path/to/cert.pfx [password]

Security & CI notes
- Never commit PFX files or passwords to the repository. Store secrets in your
  CI provider's secret store and inject them into the runner at build time.
- Example CI flow (GitHub Actions) — high level:
  1. Create a job step that imports the PFX from secrets into a file (securely)
  2. Run the signer script (pwsh) to sign the PowerShell helper
  3. Remove the PFX file and any temporary certs from the store

Signing with a CA-issued certificate is recommended for production. For
development you can create a self-signed code-signing cert with
`New-SelfSignedCertificate` and sign locally; remember that other machines will
not trust a self-signed cert unless you install the public certificate.
