"""ponics-view: open the replay viewer locally, no account or internet needed.

  ponics-view                 record replays if missing, serve results/replay, open the browser
  ponics-view --refresh       re-record the replays first (after retraining or code changes)
  ponics-view --port 8765 --no-browser
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser
from pathlib import Path

from ponics_eval.export import ROOT
from ponics_eval.export import main as export_main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ponics-view")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    d = ROOT / "results" / "replay"
    if a.refresh or not (d / "replay.json").exists():
        export_main([])
    (d / "index.html").write_text((Path(__file__).parent / "viewer" / "index.html").read_text())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"replay viewer at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
