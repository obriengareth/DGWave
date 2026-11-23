param(
    [Parameter(Mandatory=$true)][string]$StatusPath,
    [string]$AttemptId = 'rainier-attempt-001',
    [switch]$NoHostReserve,
    [string]$SetupDirectory,
    [string]$RestartPath
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($SetupDirectory)) { $SetupDirectory = Join-Path $root 'output\rainier_run_setup' }
$approval = Get-Content -Raw (Join-Path $SetupDirectory 'approval.json') | ConvertFrom-Json
$wakePath = Join-Path (Split-Path -Parent $StatusPath) 'rainier-keepawake.json'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RainierWake {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
    [DllImport("kernel32.dll")]
    public static extern uint GetCurrentThreadId();
}
'@
function Write-AtomicJson([string]$Path, $Value) {
    $temporary = $Path + '.windows.tmp'
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
$previous = [RainierWake]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) { throw 'Windows rejected the scoped system sleep-prevention request' }
$wake = @{
    active=$true; pid=$PID; thread_id=[RainierWake]::GetCurrentThreadId()
    acquired_utc=[DateTime]::UtcNow.ToString('o')
    method='SetThreadExecutionState ES_CONTINUOUS|ES_SYSTEM_REQUIRED; display/power policy unchanged'
    api_return_previous_state=$previous
}
$process = $null
try {
    Write-AtomicJson $wakePath $wake
    Write-AtomicJson $StatusPath @{
        state='starting'; stage='Windows wake lock acquired; starting approved WSL job'
        timestamp_utc=[DateTime]::UtcNow.ToString('o'); keepawake=$wake
        step=$(if ($approval.continuation) {$approval.continuation.start_step} else {0})
        total_steps=$approval.integration.steps; duration_s=$approval.approved.duration_s
        restart_origin=$approval.continuation
        attempt_id=$AttemptId; free_memory_reserve_enabled=(-not $NoHostReserve.IsPresent)
        total_job_limit_bytes=15000000000
    }
    function Resolve-LinuxPath([string]$Path) {
        $resolved = & wsl.exe -d Ubuntu-22.04 --exec wslpath -a $Path
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($resolved)) {
            throw "Unable to resolve approved job path in WSL: $Path"
        }
        return $resolved.Trim()
    }
    $linuxRoot = Resolve-LinuxPath $root
    $linuxStatus = Resolve-LinuxPath $StatusPath
    $linuxWake = Resolve-LinuxPath $wakePath
    $linuxSetup = Resolve-LinuxPath $SetupDirectory
    $reserveOption = if ($NoHostReserve) { ' --no-host-reserve' } else { '' }
    $restartOption = if ($RestartPath) { " --restart '$(Resolve-LinuxPath $RestartPath)'" } else { '' }
    $command = "cd '$linuxRoot' && .venv/bin/python -m scripts.rainier_launch --approve-run --status-path '$linuxStatus' --keepawake-info '$linuxWake' --attempt-id '$AttemptId' --setup-dir '$linuxSetup'$reserveOption$restartOption"
    $process = Start-Process -FilePath 'wsl.exe' -ArgumentList @('-d','Ubuntu-22.04','--exec','bash','-lc',"`"$command`"") `
        -RedirectStandardOutput (Join-Path $SetupDirectory 'launcher.stdout') `
        -RedirectStandardError (Join-Path $SetupDirectory 'launcher.stderr') -PassThru -NoNewWindow
    $wake.wsl_client_pid = $process.Id
    Write-AtomicJson $wakePath $wake
    while (-not $process.WaitForExit(1000)) {
        if ([RainierWake]::GetCurrentThreadId() -ne $wake.thread_id) { throw 'Sleep-prevention owning thread changed unexpectedly' }
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "Approved Rainier job exited with code $($process.ExitCode); inspect launcher.stderr and run.stderr" }
} catch {
    $live = if (Test-Path -LiteralPath $StatusPath) { Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json -AsHashtable } else { @{} }
    if ($null -ne $process -and -not $process.HasExited -and $live.ContainsKey('mpiexec_pid')) {
        & wsl.exe -d Ubuntu-22.04 -- kill -TERM -- "-$($live.mpiexec_pid)"
        if (-not $process.WaitForExit(15000)) { Stop-Process -Id $process.Id -Force }
    }
    $live.state = 'failed'
    $priorErrors = if ($live.ContainsKey('errors')) { @($live.errors) } else { @() }
    $live.errors = $priorErrors + @([string]$_)
    Write-AtomicJson $StatusPath $live
    throw
} finally {
    $released = [RainierWake]::SetThreadExecutionState([uint32]2147483648)
    $wake.active = $false
    $wake.released_utc = [DateTime]::UtcNow.ToString('o')
    $wake.release_api_return = $released
    Write-AtomicJson $wakePath $wake
    if (Test-Path -LiteralPath $StatusPath) {
        $live = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json -AsHashtable
        $live.keepawake = $wake
        $live.timestamp_utc = [DateTime]::UtcNow.ToString('o')
        Write-AtomicJson $StatusPath $live
    }
}
