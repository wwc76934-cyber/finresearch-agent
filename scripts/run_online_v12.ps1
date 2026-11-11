# v1.2 online comparison: B2 (real ReAct) vs B3 vs B4 on golden + blind (v1) + multidoc. blind2 is sealed and NOT included.
# Resumable per architecture (reports/eval/<name>.partial.jsonl). Usage: powershell scripts/run_online_v12.ps1 [run_tag]
param([string]$Tag = "r1")
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
foreach ($arch in "B2", "B3", "B4") {
    & $py -u (Join-Path $PSScriptRoot "run_online_eval.py") --arch $arch --workers 4 --name "online_v12_${arch}_$Tag" --datasets "golden,blind,multidoc"
}
