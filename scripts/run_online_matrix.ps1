# Online (real LLM) runs of the baselines on the real-company datasets, one architecture after another.
# Each run is resumable: re-running this script skips tasks already in reports/eval/<name>.partial.jsonl.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
foreach ($arch in "B4", "B3", "B2", "B0") {
    & $py -u (Join-Path $PSScriptRoot "run_online_eval.py") --arch $arch --workers 4 --name "online_real_$arch" --datasets "golden,blind"
}
