$ErrorActionPreference = "Stop"

$PythonExe = "D:\anaconda3\envs\SCA\python.exe"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

& $PythonExe (Join-Path $Here "01_run_airway_clustering.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe (Join-Path $Here "02_run_airway_de.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe (Join-Path $Here "03_run_airway_rps.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe (Join-Path $Here "04_plot_airway_de_final_style.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
