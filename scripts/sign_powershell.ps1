<#
.SYNOPSIS
  Sign a PowerShell script (.ps1) using a PFX code-signing certificate.

.DESCRIPTION
  Signs the provided script with the certificate contained in the provided
  PFX file. By default the PFX is imported into the CurrentUser\My store
  temporarily and removed after signing unless -KeepCertInStore is used.

.PARAMETER ScriptPath
  Path to the script file to sign.

.PARAMETER PfxPath
  Path to the .pfx file containing the code-signing certificate.

.PARAMETER PfxPassword
  Optional plain-text password for the PFX. If not provided the script will
  prompt securely for the password.

.PARAMETER KeepCertInStore
  If set, the imported certificate will be left in the CurrentUser\My store
  after signing. By default the certificate is removed after signing.

.PARAMETER TimestampServer
  Optional RFC3161 timestamp server URL to include in the signature.

.EXAMPLE
  # Prompt for password, sign script and remove imported cert after signing
  .\scripts\sign_powershell.ps1 -ScriptPath .\scripts\set_log_env.ps1 -PfxPath C:\secrets\sign.pfx

.NOTES
  - Requires PowerShell 5+ or PowerShell Core with Set-AuthenticodeSignature.
  - The caller must have permission to import certificates into the personal
    store (CurrentUser\My).
  - For production use, prefer a CA-issued code-signing certificate.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ScriptPath,

    [Parameter(Mandatory=$true)]
    [string]$PfxPath,

    [string]$PfxPassword = "",

    [switch]$KeepCertInStore,

    [string]$TimestampServer = ""
)

function _ExitWithError([string]$msg, [int]$code = 1) {
    Write-Error $msg
    exit $code
}

if (-not (Test-Path -Path $ScriptPath -PathType Leaf)) {
    _ExitWithError "Script to sign not found: $ScriptPath"
}

if (-not (Test-Path -Path $PfxPath -PathType Leaf)) {
    _ExitWithError "PFX file not found: $PfxPath"
}

# Securely obtain password if not provided
if ([string]::IsNullOrEmpty($PfxPassword)) {
    $securePwd = Read-Host -Prompt "PFX password" -AsSecureString
} else {
    try {
        $securePwd = ConvertTo-SecureString -String $PfxPassword -AsPlainText -Force
    } catch {
        _ExitWithError "Failed to convert provided PFX password to secure string: $_"
    }
}

# Import PFX into CurrentUser\My (personal) store temporarily
try {
    $cert = Import-PfxCertificate -FilePath $PfxPath -CertStoreLocation Cert:\CurrentUser\My -Password $securePwd
} catch {
    _ExitWithError "Failed to import PFX: $_"
}

if (-not $cert) {
    _ExitWithError "Failed to obtain certificate from PFX: $PfxPath"
}

# Use the first certificate imported (Import-PfxCertificate may return an array)
if ($cert -is [System.Array]) { $cert = $cert[0] }

Write-Output "Signing $ScriptPath with certificate thumbprint $($cert.Thumbprint)"

try {
    if ([string]::IsNullOrWhiteSpace($TimestampServer)) {
        $sig = Set-AuthenticodeSignature -FilePath $ScriptPath -Certificate $cert
    } else {
        $sig = Set-AuthenticodeSignature -FilePath $ScriptPath -Certificate $cert -TimestampServer $TimestampServer
    }
} catch {
    _ExitWithError "Signing failed: $_"
}

if ($sig.Status -ne 'Valid') {
    Write-Warning "Signature status: $($sig.Status)"
} else {
    Write-Output "Successfully signed $ScriptPath"
}

# Optionally remove the imported certificate
if (-not $KeepCertInStore) {
    try {
        $thumb = $cert.Thumbprint
        $storePath = "Cert:\CurrentUser\My\$thumb"
        if (Test-Path $storePath) {
            Remove-Item -Path $storePath -Force
            Write-Output "Removed certificate $thumb from CurrentUser\\My store"
        }
    } catch {
        Write-Warning "Failed to remove imported certificate: $_"
    }
}

exit 0
