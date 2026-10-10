<#
Compatibility wrapper for the cross-platform paper-reproduction command.

The workflow is defined once in ``python main.py reproduce-paper``. This
PowerShell file remains for Windows users who already rely on its historical
name; it contains no experiment logic.
#>

[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$IncludeExternalValidation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Activate the project environment first."
}

$Arguments = @("main.py", "reproduce-paper")
if ($SkipTests) {
    $Arguments += "--skip-tests"
}
if ($IncludeExternalValidation) {
    $Arguments += "--include-external-validation"
}

& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Paper reproduction failed with exit code ${LASTEXITCODE}."
}
