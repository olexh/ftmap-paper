# tests/bench.py
"""Peak RSS and wall clock, measured in a subprocess that does one thing.

WHY A SUBPROCESS. `ru_maxrss` is a high-water mark for the whole process and it
never goes down. Measured in the pytest process it reports the largest thing any
test has done since the session started, which is a number about test ordering.
A fresh interpreter per measurement is the only way the figure means what it
says.

WHY A WARM-UP. Importing openpyxl, followthemoney and the ftmap modules costs
about 50 MB before a single byte of data is read, and that cost belongs to the
interpreter rather than to the thing under test. The warm-up read pays it inside
the measured process, so the reported peak is the floor plus the work — not a
number that would look different on a machine with a different import graph.

The ceilings themselves are NOT here. They are in
`docs/measurements/2026-08-25-scale-ceilings.md`, registered before the code
they grade was touched, and imported by the tests that enforce them so the two
cannot drift.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Measurement:
    peak_mb: float
    seconds: float
    result: object

    def __str__(self) -> str:
        return f"peak {self.peak_mb:.0f} MB in {self.seconds:.1f}s"


_HARNESS = textwrap.dedent('''
    import json, os, resource, sys, tempfile, time
    sys.path.insert(0, {repo!r} + "/src")
    # `tests/` too, so a measured body can build its own fixture — the padded
    # workbook is generated rather than committed.
    sys.path.insert(0, {repo!r} + "/tests")

    # WARM-UP: pay the import and allocator cost before the clock starts, in
    # this process, so the reported peak is the floor plus the work.
    from ftmap.io.tabular import read_source
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("a,b\\n1,2\\n")
        warm = fh.name
    read_source(warm)
    os.unlink(warm)

    start = time.monotonic()
    result = None
    exec(compile({body!r}, "<measured>", "exec"), globals())
    seconds = time.monotonic() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, Darwin reports bytes. Both are documented; the
    # difference is not, and a harness that guessed would be off by 1024.
    scale = (1 << 20) if sys.platform == "darwin" else (1 << 10)
    print("@@" + json.dumps({{"peak_mb": peak / scale, "seconds": seconds,
                             "result": result}}))
''')


def measure(body: str, timeout: float = 900) -> Measurement:
    """Run `body` in a fresh interpreter and report what it cost.

    `body` is Python source. Assign to `result` for anything the caller needs
    back; it is round-tripped through JSON, so it has to be plain data.
    """
    script = _HARNESS.format(repo=REPO, body=body)
    proc = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"measured subprocess failed:\n{proc.stdout}\n{proc.stderr}")
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("@@")),
                None)
    if line is None:
        raise RuntimeError(f"harness printed no result:\n{proc.stdout}")
    payload = json.loads(line[2:])
    return Measurement(payload["peak_mb"], payload["seconds"],
                       payload["result"])
