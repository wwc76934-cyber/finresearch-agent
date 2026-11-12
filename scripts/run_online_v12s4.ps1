# Step 4 n=3: B4 with the ComplexityRouter / parallel verifier, final code. blind2 is NOT included.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
foreach ($r in "r1", "r2", "r3") {
    & $py -u (Join-Path $PSScriptRoot "run_online_eval.py") --arch B4 --workers 4 --name "online_v12s4_B4_$r" --datasets "golden,blind,multidoc"
}
