"""Resumable download of Kaggle job output files over a slow / flaky connection.

The Kaggle CLI holds a whole file in memory and cannot resume, so a stalled connection
loses everything. This lists the job's output files through the API and downloads the
matching ones with curl (resume on retry, stall detection).

  python business_entity_resolution/kaggle/fetch_output.py ber-v1 'output/v1_outputs\\.zip$' submissions/v1
"""
import re
import subprocess
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

USER = "peeyushprashant"


def list_files(kernel: str) -> list[tuple[str, str]]:
    api = KaggleApi()
    api.authenticate()
    out, token = [], None
    with api.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug, req.page_size = USER, kernel, 100
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            out += [(f.file_name, f.url) for f in resp.files]
            token = resp.next_page_token
            if not token:
                return out


def fetch(kernel: str, pattern: str, dest: str, attempts: int = 30) -> None:
    rx = re.compile(pattern)
    files = [(n, u) for n, u in list_files(kernel) if rx.search(n)]
    if not files:
        sys.exit(f"no output file of {kernel} matches {pattern!r}")
    for name, url in files:
        path = Path(dest) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"{name} -> {path}", flush=True)
        for i in range(attempts):
            # -C - resumes from the bytes already on disk; abort if < 1 KB/s for 60 s, then retry
            r = subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(path), "--speed-limit", "1024",
                                "--speed-time", "60", "--connect-timeout", "30", "-sS", url])
            if r.returncode == 0:
                print(f"done: {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
                break
            if r.returncode == 22 and path.exists():  # e.g. 416: already complete
                print(f"curl http error; file has {path.stat().st_size / 1e6:.1f} MB", flush=True)
            print(f"  attempt {i + 1} stopped (curl {r.returncode}); {path.stat().st_size / 1e6 if path.exists() else 0:.1f} MB so far, resuming",
                  flush=True)
            time.sleep(5)
            if i % 5 == 4:  # signed URLs can expire; refresh
                url = dict(list_files(kernel))[name]
        else:
            sys.exit(f"gave up on {name}")


if __name__ == "__main__":
    fetch(*sys.argv[1:4])
