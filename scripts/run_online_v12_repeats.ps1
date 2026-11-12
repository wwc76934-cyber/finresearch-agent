# Repeats for n=3: B2 r3, then B3 r2/r3, then B4 r2/r3 (r1 already exists for B3/B4, r2 for B2). blind2 is NOT included.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$jobs = @(@("B2", "r3"), @("B3", "r2"), @("B3", "r3"), @("B4", "r2"), @("B4", "r3"))
foreach ($j in $jobs) {
    & $py -u (Join-Path $PSScriptRoot "run_online_eval.py") --arch $j[0] --workers 4 --name "online_v12_$($j[0])_$($j[1])" --datasets "golden,blind,multidoc"
}
