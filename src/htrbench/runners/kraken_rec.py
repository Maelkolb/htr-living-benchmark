"""Kraken line recogniser: a persistent JSON-lines worker (scripts/kraken_worker.py) in Kraken's own environment, on the CPU."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.request
from pathlib import Path

from .. import paths
from ..layout.segment_kraken import KRAKEN_PYTHON
from ..schema import SystemSpec
from .base import Runner
from .linepage import LineModelMixin

log = logging.getLogger("htrbench.kraken")

WORKER = paths.ROOT / "scripts" / "kraken_worker.py"
WORKER_LOG = paths.CACHE / "kraken_worker.log"


class KrakenRecRunner(LineModelMixin, Runner):
    device = "cpu"

    def __init__(self, spec: SystemSpec, params: dict | None = None):
        super().__init__(spec, params)
        self.proc: subprocess.Popen | None = None
        self._log_fh = None

    def _ensure_model(self) -> Path:
        mp = paths.ROOT / self.spec.model_id
        if mp.exists() and mp.stat().st_size > 0:
            return mp
        url = self.params.get("model_url")
        if not url:
            raise FileNotFoundError(f"kraken model {mp} missing and no params.model_url")
        mp.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s -> %s", url, mp)
        tmp = mp.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(mp)
        return mp

    def setup(self) -> None:
        if not KRAKEN_PYTHON.exists():
            raise FileNotFoundError(f"{KRAKEN_PYTHON} missing; create the environment as scripts/setup_kraken.ps1 does")
        mp = self._ensure_model()
        WORKER_LOG.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(WORKER_LOG, "a", encoding="utf-8")  # noqa: SIM115
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        self.proc = subprocess.Popen(
            [str(KRAKEN_PYTHON), str(WORKER), mp.as_posix(), self.params.get("device", "cpu"), self.params.get("seg_mode", "bbox")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log_fh, text=True, encoding="utf-8",
            bufsize=1, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ready = self._readline()
        if not ready.get("ready"):
            raise RuntimeError(f"kraken worker did not start: {ready}")
        log.info("kraken worker ready: %s", ready)

    def teardown(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.proc = None
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None

    def _readline(self) -> dict:
        assert self.proc is not None
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            raise RuntimeError(f"kraken worker died (rc={rc}); see {WORKER_LOG}")
        return json.loads(line)

    def recognize_images(self, paths_: list[Path]) -> list[str]:
        if self.proc is None or self.proc.poll() is not None:
            self.setup()
        assert self.proc is not None
        out: list[str] = []
        chunk = int(self.params.get("worker_chunk", 32))
        for i in range(0, len(paths_), chunk):
            batch = [Path(p).resolve().as_posix() for p in paths_[i : i + chunk]]
            self.proc.stdin.write(json.dumps({"paths": batch}) + "\n")
            self.proc.stdin.flush()
            resp = self._readline()
            for p, e in zip(batch, resp.get("errors", []), strict=False):
                if e:
                    log.warning("kraken failed on %s: %s", p, e)
            out.extend(resp["texts"])
        return out
