$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$exe = Join-Path $root "NatureLab.exe"
$log = Join-Path $root "launcher_test.log"
Remove-Item -LiteralPath $log -Force -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $exe)) { throw "NatureLab.exe not found" }

$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = $exe
$info.UseShellExecute = $false
$info.EnvironmentVariables["NATURELAB_TEST_MODE"] = "1"
$process = [System.Diagnostics.Process]::Start($info)
if (-not $process.WaitForExit(90000)) {
    $process.Kill()
    throw "launcher timed out (possible recursion)"
}
if ($process.ExitCode -ne 0) { throw "launcher exit code $($process.ExitCode)" }
if (-not (Test-Path -LiteralPath $log)) { throw "launcher lifecycle log missing" }
$content = Get-Content -LiteralPath $log -Raw
if ($content -notmatch "launcher done") { throw "launcher did not finish cleanly" }
$orphans = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "uvicorn.*app.main:app" })
if ($orphans.Count -ne 0) { throw "launcher left $($orphans.Count) backend process(es)" }
Remove-Item -LiteralPath $log -Force
"PASS  frozen launcher lifecycle"
