"""OTLP/HTTP export. A local receiver stands in for Phoenix or an OTel collector.
The child runs in a subprocess because the OTel provider is a process-wide singleton."""
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent("""
    import sys
    from datetime import datetime
    sys.path.insert(0, "packages")
    from agent_runtime import ResearchRequest, run_research
    from evaluation_platform.tracing import flush_otel
    run_research(ResearchRequest(ticker="990001.SZ", as_of=datetime(2025, 12, 31, 15)), "B3", persist=False)
    flush_otel()
""")


def test_spans_reach_an_otlp_http_endpoint():
    received = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append((self.path, len(body), b"research_run" in body, b"langgraph.node.planner" in body))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "TRACING_BACKEND": "otel", "LLM_OFFLINE": "1", "CHECKPOINTER": "memory",
           "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{srv.server_port}/v1/traces"}
    env.pop("DATABASE_URL", None)
    try:
        subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, check=True, timeout=240)
    finally:
        srv.shutdown()
    assert received, "no OTLP request arrived"
    assert all(p == "/v1/traces" for p, *_ in received)
    assert any(has_root for _, _, has_root, _ in received)
    assert any(has_node for _, _, _, has_node in received)
