param([string]$Scenario)
$ErrorActionPreference = 'Stop'
$Work = if ($env:EEG_WORK) { $env:EEG_WORK } else { '/work' }
if (!$Scenario) {
    $errors = $null; $tokens = $null
    [System.Management.Automation.Language.Parser]::ParseFile("$Work/start-eeg-llm.ps1", [ref]$tokens, [ref]$errors) > $null
    if ($errors) { throw ($errors | Out-String) }
    foreach ($case in @('success', 'embedding-missing', 'docker-off', 'ollama-off', 'build-fail', 'preflight-fail', 'up-fail', 'windows-containers', 'check', 'stop', 'prebuilt')) {
        $output = & "$PSHOME/pwsh" -NoProfile -File $PSCommandPath -Scenario $case 2>&1
        $code = $LASTEXITCODE
        $shouldPass = $case -in @('success', 'embedding-missing', 'check', 'stop', 'prebuilt')
        if (($code -eq 0) -ne $shouldPass) { throw "Wrong exit for ${case}: $code`n$output" }
        if (!$shouldPass -and "$output".Contains('Ready:')) { throw "False success for $case" }
        if ($case -eq 'embedding-missing' -and !("$output".Contains('MOCK: pull nomic-embed-text:latest'))) { throw 'Missing embedding not pulled' }
        if ($case -eq 'check' -and "$output" -match 'MOCK: (build|up|pull|create)') { throw 'Check mutated setup' }
        if ($case -eq 'stop' -and "$output".Contains('MOCK: ollama')) { throw 'Stop requires Ollama' }
        Write-Host "PASS: Windows PowerShell flow $case"
    }
    exit 0
}
$global:caseName = $Scenario
function global:docker {
    $step = $args -join ' '
    Write-Host "MOCK: $step"
    $global:LASTEXITCODE = 0
    if (($caseName -eq 'docker-off' -and $step -like 'info*') -or
        ($caseName -eq 'build-fail' -and $step -like 'compose build*') -or
        ($caseName -eq 'preflight-fail' -and $step -like 'compose run*') -or
        ($caseName -eq 'up-fail' -and $step -like 'compose up*')) { $global:LASTEXITCODE = 1; return }
    if ($step -eq 'info --format {{.OSType}}') { if ($caseName -eq 'windows-containers') { 'windows' } else { 'linux' } }
    if ($step -eq 'compose config --images') { if ($caseName -eq 'prebuilt') { 'offline:test' } else { 'eeg-llm:latest' } }
    if ($step -eq 'compose port eeg-llm 8001') { '127.0.0.1:18092' }
}
function global:ollama {
    $step = $args -join ' '
    Write-Host "MOCK: ollama $step"
    if ($step -like 'pull*') { Write-Host "MOCK: $step" }
    $global:LASTEXITCODE = 0
    if (($caseName -eq 'embedding-missing' -and $step -like 'show*') -or ($caseName -eq 'ollama-off' -and $step -eq 'list')) { $global:LASTEXITCODE = 1 }
}
$env:EEG_NO_BROWSER = '1'
$folder = Join-Path ([IO.Path]::GetTempPath()) "eeg setup $Scenario"
New-Item -ItemType Directory -Force $folder > $null
Copy-Item "$Work/start-eeg-llm.ps1" $folder
if ($Scenario -eq 'check') { & "$folder/start-eeg-llm.ps1" -Check }
elseif ($Scenario -eq 'stop') { & "$folder/start-eeg-llm.ps1" -Stop }
else { & "$folder/start-eeg-llm.ps1" }
exit $LASTEXITCODE
