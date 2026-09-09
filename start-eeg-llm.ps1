param([switch]$Check, [switch]$Stop)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PATH += ";$env:ProgramFiles\Docker\Docker\resources\bin;$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin;$env:LOCALAPPDATA\Programs\Ollama"
function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE). Read its error above." }
}
function Test-Native {
    param([string]$Exe, [string[]]$Arguments)
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments *> $null
    return ($LASTEXITCODE -eq 0)
}
try {
    Write-Host "eeg-llm setup - Windows / $env:PROCESSOR_ARCHITECTURE"
    Write-Host 'First start needs internet. Allow 40 GB free disk; 16 GB RAM recommended, 32 GB preferred.'
    Write-Host '[1/5] Checking Docker Desktop'
    if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install Docker Desktop from https://docs.docker.com/desktop/ . Open it and finish setup with the WSL 2 backend.'
    }
    Invoke-Native docker @('compose', 'version')
    $serverOS = & docker info --format '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Open Docker Desktop and wait for the engine to run. If requested, install/update WSL and restart Windows.' }
    if ($serverOS -ne 'linux') { throw 'In the Docker Desktop tray menu choose Switch to Linux containers.' }
    if ($Stop) {
        Invoke-Native docker @('compose', 'stop')
        Write-Host 'eeg-llm stopped. Recordings and saved results are kept.'
        exit 0
    }
    Write-Host '[2/5] Checking Ollama'
    if (!(Get-Command ollama -ErrorAction SilentlyContinue)) { throw 'Install and open Ollama from https://ollama.com/download/windows . Then close this window and double-click Start again.' }
    $env:OLLAMA_HOST = 'http://127.0.0.1:11434'
    if ($env:EEG_HOST_OLLAMA_URL) { $env:OLLAMA_HOST = $env:EEG_HOST_OLLAMA_URL }
    if (!(Test-Native ollama @('list'))) { throw 'Open Ollama from the Start menu and wait for it to start, then try again.' }
    foreach ($dir in @('data-in', 'data-out', 'docs')) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if (!$Check) {
        Write-Host '[3/5] Preparing models (roughly 10 GB on first start; keep this window open)'
        if (!(Test-Native ollama @('show', 'nomic-embed-text:latest'))) { Invoke-Native ollama @('pull', 'nomic-embed-text:latest') }
        Invoke-Native ollama @('create', 'eeg-qwen', '-f', 'Modelfile')
        Write-Host '[4/5] Preparing app (first build 5-20+ minutes; later builds use cache)'
    }
    Remove-Item Env:OLLAMA_HOST -ErrorAction SilentlyContinue
    if (!$Check) {
        $image = & docker compose config --images
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read docker-compose.yml / .env.' }
        if ($image -ne 'eeg-llm:latest') {
            if (!(Test-Native docker @('image', 'inspect', $image))) { Invoke-Native docker @('compose', 'pull', 'eeg-llm') }
            Write-Host "Using prebuilt image $image (docker compose pull to update it)."
        } else { Invoke-Native docker @('compose', 'build', 'eeg-llm') }
    }
    Write-Host '[5/5] Checking connection, both models, storage and scientific libraries'
    Invoke-Native docker @('compose', 'run', '--rm', '--no-deps', '--entrypoint', 'python', 'eeg-llm', '/usr/local/bin/preflight.py')
    if ($Check) { Write-Host 'CHECK: PASS'; exit 0 }
    & docker compose up -d --no-build --wait --wait-timeout 180
    if ($LASTEXITCODE -ne 0) {
        & docker compose logs --tail 60 eeg-llm
        throw 'The app did not become ready. If port 8001 is occupied, set APP_PORT=8011 in .env and retry.'
    }
    $binding = & docker compose port eeg-llm 8001
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the app port from Docker.' }
    $url = "http://$($binding.Trim())"
    Write-Host "Ready: $url"
    Write-Host 'The app runs in the background. You can close this window. Stop it with the Stop button in Docker Desktop.'
    if ($env:EEG_NO_BROWSER -ne '1') { Start-Process $url }
    exit 0
} catch {
    Write-Host "`nSetup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'See START-HERE.html, then double-click Start again. Completed downloads are reused.'
    exit 1
}
