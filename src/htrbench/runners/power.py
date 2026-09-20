"""GPU power sampling for local runs (nvidia-smi polling in a background thread).

Energy (Wh) = mean power draw over the sampled window x elapsed seconds / 3600.
This is board power only (no host CPU/RAM); documented as such in METHODS.md.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time


class PowerSampler:
    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._t1 = 0.0
        self.available = shutil.which("nvidia-smi") is not None

    def _read(self) -> float | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return float(out.strip().splitlines()[0])
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self._samples.append(v)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PowerSampler:
        self._t0 = time.perf_counter()
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return max(self._t1 - self._t0, 0.0)

    @property
    def mean_w(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    @property
    def energy_wh(self) -> float | None:
        if self.mean_w is None:
            return None
        return self.mean_w * self.elapsed_s / 3600.0


def gpu_name() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=5
        ).strip().splitlines()[0]
    except Exception:
        return None
