# Fill n=3 with valid runs: B2 r1 (old adapter) and B4 r1 (template fallback after credit ran out) were invalid.
# New valid repeats are named r4. blind2 is NOT included.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
foreach ($arch in "B2", "B4") {
    & $py -u (Join-Path $PSScriptRoot "run_online_eval.py") --arch $arch --workers 4 --name "online_v12_${arch}_r4" --datasets "golden,blind,multidoc"
}
