from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)


class LlamaServerError(RuntimeError):
    pass


class LlamaServer:
    """Manages a single llama-server.exe process."""

    def __init__(self, name: str, model_path: Path, port: int, extra_args: tuple[str, ...] = ()) -> None:
        self.name = name
        self.model_path = model_path
        self.port = port
        self.extra_args = extra_args
        self._proc: subprocess.Popen | None = None

    @property
    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/health"

    def is_already_running(self) -> bool:
        try:
            with urlopen(self.health_url, timeout=2) as r:
                return r.status == 200
        except (URLError, OSError):
            return False

    def start(self, llama_server_exe: Path, health_timeout: int = 120) -> None:
        if self.is_already_running():
            logger.info("%s already running on port %s — skipping start", self.name, self.port)
            return

        if not llama_server_exe.exists():
            raise LlamaServerError(
                f"llama-server.exe not found at {llama_server_exe}. "
                f"Set LLAMA_SERVER_EXE in .env."
            )
        if not self.model_path.exists():
            raise LlamaServerError(
                f"Model file not found: {self.model_path}. "
                f"Check the path in .env."
            )

        cmd = [
            str(llama_server_exe),
            "-m", str(self.model_path),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-ngl", "-1",
            *self.extra_args,
        ]
        logger.info("Starting %s on port %s …", self.name, self.port)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.monotonic() + health_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise LlamaServerError(
                    f"{self.name} exited early (code {self._proc.returncode}). "
                    f"Check the model path and available VRAM."
                )
            if self.is_already_running():
                logger.info("%s ready on port %s", self.name, self.port)
                return
            time.sleep(1.0)

        self._proc.kill()
        raise LlamaServerError(
            f"{self.name} did not become ready within {health_timeout}s on port {self.port}."
        )

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            logger.info("Stopping %s (pid %s)", self.name, self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    @property
    def was_started_by_us(self) -> bool:
        return self._proc is not None


class ServerManager:
    """Manages the embed and LLM llama.cpp server pair."""

    def __init__(
        self,
        llama_server_exe: Path,
        embed_model: Path,
        embed_port: int,
        embed_ctx: int,
        llm_model: Path,
        llm_port: int,
        llm_ctx: int,
        health_timeout: int = 120,
    ) -> None:
        self.llama_server_exe = llama_server_exe
        self.health_timeout = health_timeout
        self.embed_server = LlamaServer(
            name="Embedding server",
            model_path=embed_model,
            port=embed_port,
            extra_args=("--embedding", "--pooling", "mean", "-c", str(embed_ctx)),
        )
        self.llm_server = LlamaServer(
            name="LLM server",
            model_path=llm_model,
            port=llm_port,
            extra_args=("-c", str(llm_ctx)),
        )

    def start_all(self) -> None:
        self.embed_server.start(self.llama_server_exe, self.health_timeout)
        self.llm_server.start(self.llama_server_exe, self.health_timeout)

    def stop_all(self) -> None:
        self.embed_server.stop()
        self.llm_server.stop()

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ServerManager":  # type: ignore[name-defined]
        from config import Settings as S
        assert isinstance(settings, S)
        if not settings.llama_server_exe:
            raise LlamaServerError(
                "LLAMA_SERVER_EXE is not set. "
                "Set it in .env to enable automatic server management."
            )
        return cls(
            llama_server_exe=Path(settings.llama_server_exe),
            embed_model=Path(settings.embed_model_path),
            embed_port=settings.embed_server_port,
            embed_ctx=settings.embed_n_ctx,
            llm_model=Path(settings.llm_model_path),
            llm_port=settings.llm_server_port,
            llm_ctx=settings.llm_n_ctx,
            health_timeout=settings.llama_health_timeout,
        )
