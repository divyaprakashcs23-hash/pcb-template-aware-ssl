# Creates the public GitHub repository and pushes this project to it.
# Run from PowerShell:  .\push_to_github.ps1
Set-Location $PSScriptRoot
$env:Path += ";C:\Program Files\GitHub CLI"
gh repo create pcb-template-aware-ssl --public --source . --remote origin --push
gh repo view --web
