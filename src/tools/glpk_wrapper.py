#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> int:
    real_bin = os.environ.get("MVMTSP_GLPK_REAL_BIN", "glpsol")
    log_path = os.environ.get("MVMTSP_GLPK_LOG_PATH", "")
    echo_output = os.environ.get("MVMTSP_GLPK_ECHO", "0") == "1"

    started_at = time.perf_counter()
    process = subprocess.Popen(
        [real_bin, *sys.argv[1:]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        if process.stdout is not None:
            for raw_line in process.stdout:
                elapsed = time.perf_counter() - started_at
                stamped_line = f"[{elapsed:.6f}] {raw_line.rstrip()}\n"
                if log_handle is not None:
                    log_handle.write(stamped_line)
                    log_handle.flush()
                if echo_output:
                    sys.stdout.write(raw_line)
                    sys.stdout.flush()
        return process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
