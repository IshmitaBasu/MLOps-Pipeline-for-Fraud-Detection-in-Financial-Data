param(
    [string]$TaskName = "FraudDataPipelineInbox",
    [int]$IntervalMinutes = 15
)

$codeDirectory = Split-Path -Parent $PSScriptRoot
$workspaceDirectory = Split-Path -Parent $codeDirectory
$pythonPath = Join-Path $workspaceDirectory "masters_thesis\Scripts\python.exe"
$triggerScript = Join-Path $PSScriptRoot "run_data_pipeline.py"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Python executable not found at $pythonPath"
}

$actionArguments = '"{0}"' -f $triggerScript
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $actionArguments -WorkingDirectory $workspaceDirectory
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Checks the fraud-data inbox and runs only the data pipeline." -Force

Write-Output "Registered scheduled task '$TaskName' to run every $IntervalMinutes minutes."
