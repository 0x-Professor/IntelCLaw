param(
  [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

function Add-ToPath([string]$dir) {
  if ([string]::IsNullOrWhiteSpace($dir)) { return }
  if (Test-Path $dir) {
    $pathParts = $env:Path -split ';'
    if (-not ($pathParts -contains $dir)) {
      $env:Path = "$dir;$env:Path"
    }
  }
}

function Find-Exe([string]$name, [string[]]$candidates) {
  $cmd = Get-Command $name -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  foreach ($c in $candidates) {
    if (Test-Path $c) { return $c }
  }
  return $null
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
  $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$bridgeDir = Join-Path $RepoRoot "data\vendor\whatsapp-mcp\whatsapp-bridge"
if (-not (Test-Path $bridgeDir)) {
  Write-Error "WhatsApp bridge directory not found: $bridgeDir`nRun onboarding install for whatsapp-mcp first."
  exit 1
}

# Ensure Go is available
$goExe = Find-Exe "go" @(
  (Join-Path $env:ProgramFiles "Go\bin\go.exe"),
  (Join-Path $env:LOCALAPPDATA "Programs\Go\bin\go.exe")
)
if (-not $goExe) {
  Write-Error "Go not found on PATH or in common install locations. Install Go, then re-run."
  exit 1
}
Add-ToPath (Split-Path $goExe -Parent)

# Ensure a C compiler is available for go-sqlite3 (CGO)
$gccExe = Find-Exe "gcc" @(
  "C:\msys64\ucrt64\bin\gcc.exe",
  "C:\msys64\mingw64\bin\gcc.exe",
  "C:\MinGW\bin\gcc.exe",
  "C:\Dev-Cpp\bin\gcc.exe"
)
if ($gccExe) {
  Add-ToPath (Split-Path $gccExe -Parent)
} else {
  Write-Warning "gcc not found. The bridge requires CGO for go-sqlite3. Install MSYS2 (ucrt64) or MinGW and ensure gcc.exe is on PATH."
}

$env:CGO_ENABLED = "1"

Write-Host ""
Write-Host "RepoRoot  : $RepoRoot"
Write-Host "BridgeDir : $bridgeDir"
Write-Host "Go        : $goExe"
Write-Host "GCC       : $gccExe"
Write-Host "CGO_ENABLED=$env:CGO_ENABLED"
Write-Host ""
Write-Host "Starting WhatsApp bridge (QR login appears below on first run)."
Write-Host "Keep this window open while using WhatsApp MCP."
Write-Host ""

Set-Location $bridgeDir

# Persist CGO setting for Go (matches upstream README), but also set env var above.
& $goExe env -w CGO_ENABLED=1 | Out-Null

& $goExe mod download
& $goExe run .

