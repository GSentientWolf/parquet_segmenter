<#
.SYNOPSIS
  PowerShell helper to export logging environment variables for parquet_segmenter.

.DESCRIPTION
  Dot-source this file to export environment variables into your current session.

.PARAMETER EnableFile
  Switch - enable file sink (sets PARQUET_SEGMENTER_LOG_TO_FILE=1). If omitted sets 0.

.PARAMETER Path
  Path to write logs. Defaults to repo-root\parquet_segmenter.log when empty.

.PARAMETER Level
  Log level (INFO by default). Example: DEBUG, INFO, WARNING, ERROR.

.EXAMPLE
  # Dot-source to export into the current session
  . .\scripts\set_log_env.ps1 -EnableFile -Path C:\temp\ps.log -Level DEBUG
#>
param(
    [switch]$EnableFile,
    [string]$Path = "",
    [string]$Level = "INFO"
)

# Resolve script and repo locations
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")

if ([string]::IsNullOrWhiteSpace($Path)) {
    $Path = Join-Path $repoRoot "parquet_segmenter.log"
}

# Export environment variables for the current PowerShell session
if ($EnableFile) {
    $env:PARQUET_SEGMENTER_LOG_TO_FILE = "1"
} else {
    $env:PARQUET_SEGMENTER_LOG_TO_FILE = "0"
}
$env:PARQUET_SEGMENTER_LOG_PATH = $Path
$env:PARQUET_SEGMENTER_LOG_LEVEL = $Level

Write-Output "Exported:"
Write-Output "  PARQUET_SEGMENTER_LOG_TO_FILE=$($env:PARQUET_SEGMENTER_LOG_TO_FILE)"
Write-Output "  PARQUET_SEGMENTER_LOG_PATH=$($env:PARQUET_SEGMENTER_LOG_PATH)"
Write-Output "  PARQUET_SEGMENTER_LOG_LEVEL=$($env:PARQUET_SEGMENTER_LOG_LEVEL)"

Write-Output "`nTo persist these in your current session, dot-source the script:`n  . .\scripts\set_log_env.ps1 -EnableFile -Path C:\tmp\ps.log -Level DEBUG"

<# Signing notes

Is it possible to sign the script?
- Yes. To sign a PowerShell script you need a code-signing certificate accessible to the machine/user.
- For production use, obtain a code-signing certificate from a trusted CA and use Set-AuthenticodeSignature.

Example (developer/test only):
  # Create a self-signed code-signing cert (may require elevated privileges)
  $cert = New-SelfSignedCertificate -DnsName "DevSigningCert" -Type CodeSigning -KeyExportPolicy Exportable -Subject "CN=DevSigningCert" -NotAfter (Get-Date).AddYears(10)

  # Export the cert for reuse (optional)
  $pwd = ConvertTo-SecureString -String "password" -Force -AsPlainText
  Export-PfxCertificate -Cert "cert:\CurrentUser\My\$($cert.Thumbprint)" -FilePath "C:\temp\devsign.pfx" -Password $pwd

  # Sign the script (using the cert object)
  Set-AuthenticodeSignature -FilePath .\scripts\set_log_env.ps1 -Certificate $cert

Caveats:
- Self-signed certs will not be trusted by other machines without installing the public cert.
- On Windows systems with execution policies or constrained environments, you may need to configure the policy or install the cert to the trusted publishers store.
- Prefer using a CA-issued code-signing certificate for production.
#>
