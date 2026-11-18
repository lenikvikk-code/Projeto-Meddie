"""Cliente mínimo da API local do Ollama (somente loopback, somente stdlib)."""

from __future__ import annotations

import http.client
import json
import socket
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import LOOPBACK_HOSTS, Config


class OllamaError(Exception):
    """Erro com mensagem já legível para o usuário."""


class OllamaUnavailable(OllamaError):
    pass


class ModelNotFound(OllamaError):
    pass


class ResponseCancelled(OllamaError):
    pass


@dataclass
class ChatStats:
    prompt_tokens: int = 0
    output_tokens: int = 0
    done_reason: str = ""
    # Durações informadas pelo Ollama (segundos)
    total_s: float = 0.0
    load_s: float = 0.0
    prompt_eval_s: float = 0.0
    eval_s: float = 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / self.eval_s if self.eval_s > 0 else 0.0


class ThinkFilter:
    """Remove blocos <think>...</think> caso o modelo os emita no conteúdo.

    Com think=false o Ollama já não deveria enviá-los; isto é só proteção.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while self._buf:
            tag = self.CLOSE if self._inside else self.OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                if not self._inside:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(tag):]
                self._inside = not self._inside
                continue
            # Guarda um possível início parcial da tag para o próximo chunk.
            keep = 0
            for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
                if tag.startswith(self._buf[-k:]):
                    keep = k
                    break
            if not self._inside:
                out.append(self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep:] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


class OllamaClient:
    def __init__(self, cfg: Config):
        if cfg.host not in LOOPBACK_HOSTS:  # defesa extra além do config
            raise OllamaError(f"Host não-loopback recusado: {cfg.host}")
        self.cfg = cfg
        self._lock = threading.Lock()
        self._active_conn: http.client.HTTPConnection | None = None

    # ---------- HTTP básico ----------
    def _conn(self, read_timeout: float | None = None) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(self.cfg.host, self.cfg.port, timeout=self.cfg.connect_timeout_s)
        try:
            conn.connect()
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            conn.close()
            raise OllamaUnavailable(
                f"Ollama não está respondendo em {self.cfg.host}:{self.cfg.port}. "
                "Verifique se o Ollama está aberto (ícone na bandeja) ou rode 'ollama serve'."
            ) from exc
        if conn.sock is not None:
            conn.sock.settimeout(read_timeout or self.cfg.read_timeout_s)
        return conn

    def _request(self, method: str, path: str, body: dict | None = None, read_timeout: float | None = None):
        conn = self._conn(read_timeout)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
        except (socket.timeout, OSError, http.client.HTTPException) as exc:
            conn.close()
            raise OllamaUnavailable(f"Falha de comunicação com o Ollama: {exc}") from exc
        return conn, resp

    @staticmethod
    def _error_text(resp) -> str:
        raw = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(raw).get("error", raw)
        except (json.JSONDecodeError, AttributeError):
            return raw or f"HTTP {resp.status}"

    def _get_json(self, path: str, read_timeout: float = 5.0) -> dict:
        conn, resp = self._request("GET", path, read_timeout=read_timeout)
        try:
            if resp.status != 200:
                raise OllamaError(f"Ollama retornou erro em {path}: {self._error_text(resp)}")
            return json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

    # ---------- Consultas ----------
    def version(self) -> str:
        return self._get_json("/api/version").get("version", "?")

    def model_installed(self) -> bool:
        tags = self._get_json("/api/tags").get("models", [])
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        wanted = self.cfg.model if ":" in self.cfg.model else self.cfg.model + ":latest"
        return wanted in names

    def installed_models(self) -> list[str]:
        return sorted(m.get("name", "") for m in self._get_json("/api/tags").get("models", []))

    def running_models(self) -> list[dict]:
        """Modelos carregados (/api/ps), com 'size' em bytes ocupados na memória."""
        return self._get_json("/api/ps").get("models", [])

    def ensure_ready(self) -> None:
        """Lança OllamaUnavailable/ModelNotFound se não der para conversar."""
        self.version()
        if not self.model_installed():
            raise ModelNotFound(
                f"O modelo '{self.cfg.model}' não está instalado. "
                f"Rode no terminal: ollama pull {self.cfg.model}"
            )

    def is_loaded(self) -> bool:
        """True se o modelo configurado está na RAM agora."""
        try:
            return any(m.get("name") == self.cfg.model or m.get("model") == self.cfg.model
                       for m in self.running_models())
        except OllamaError:
            return False

    def load_model(self) -> None:
        """Carrega o modelo na RAM (requisição vazia).

        v0.3: envia as mesmas 'options' do chat. Sem isso o Ollama carregava com o
        contexto padrão (4096) e RECARREGAVA o modelo na 1ª mensagem, que pede 8192
        (medido em 14/11/2026: 5 s a mais no primeiro token).
        """
        conn, resp = self._request(
            "POST", "/api/generate",
            {"model": self.cfg.model, "keep_alive": self.cfg.keep_alive, "stream": False,
             "options": self.cfg.model_options() | {"num_predict": 1}},
        )
        try:
            if resp.status != 200:
                raise OllamaError(f"Falha ao carregar o modelo: {self._error_text(resp)}")
            resp.read()
        finally:
            conn.close()

    def unload_model(self, timeout: float = 3.0) -> None:
        """Libera o modelo da RAM. Erros são ignorados (é chamado no encerramento)."""
        self.unload_model_named(self.cfg.model, timeout)

    def unload_model_named(self, model: str, timeout: float = 3.0) -> None:
        try:
            conn, resp = self._request(
                "POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False},
                read_timeout=timeout,
            )
            try:
                resp.read()
            finally:
                conn.close()
        except OllamaError:
            pass

    # ---------- Chat com streaming ----------
    def cancel(self) -> None:
        """Interrompe uma resposta em andamento (fecha o socket)."""
        with self._lock:
            conn = self._active_conn
        if conn is not None:
            try:
                if conn.sock is not None:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def chat_once(self, messages: list[dict], options: dict | None = None, fmt: dict | str | None = None,
                  cancelled: Callable[[], bool] = lambda: False,
                  stats: ChatStats | None = None) -> str:
        """Resposta completa de uma vez (tarefas internas: resumo, extração, aquecimento)."""
        return "".join(self.chat_stream(messages, cancelled=cancelled, stats=stats, options=options, fmt=fmt))

    def warm_up(self, messages: list[dict], cancelled: Callable[[], bool] = lambda: False,
                stats: ChatStats | None = None) -> None:
        """Carrega o modelo e deixa o começo do prompt processado no cache do Ollama.

        Pede 1 token só: o que interessa é a avaliação do prompt, que fica no cache
        e é reaproveitada pela próxima requisição com o mesmo começo.
        """
        self.chat_once(messages, options={"num_predict": 1}, cancelled=cancelled, stats=stats)

    def chat_stream(self, messages: list[dict], cancelled: Callable[[], bool] = lambda: False,
                    stats: ChatStats | None = None, options: dict | None = None,
                    fmt: dict | str | None = None) -> Iterator[str]:
        """Gera os pedaços de texto da resposta conforme chegam.

        `options` sobrepõe campos de cfg.model_options() só nesta chamada.
        `fmt` é o campo 'format' do Ollama ("json" ou um JSON Schema).
        """
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "think": self.cfg.think,
            "keep_alive": self.cfg.keep_alive,
            "options": self.cfg.model_options() | (options or {}),
        }
        if fmt is not None:
            body["format"] = fmt
        conn, resp = self._request("POST", "/api/chat", body)
        if resp.status == 400 and "think" in body:
            err = self._error_text(resp)
            if "think" in err.lower():  # modelo sem suporte a 'think': repete sem o campo
                conn.close()
                body.pop("think")
                conn, resp = self._request("POST", "/api/chat", body)
            else:
                conn.close()
                raise OllamaError(f"Requisição recusada pelo Ollama: {err}")

        with self._lock:
            self._active_conn = conn
        filt = ThinkFilter()
        try:
            if resp.status == 404:
                raise ModelNotFound(
                    f"Modelo '{self.cfg.model}' não encontrado. Rode: ollama pull {self.cfg.model}"
                )
            if resp.status != 200:
                raise OllamaError(f"Ollama retornou erro: {self._error_text(resp)}")
            while True:
                if cancelled():
                    raise ResponseCancelled("Resposta interrompida.")
                try:
                    line = resp.readline()
                except socket.timeout as exc:
                    raise OllamaError(
                        f"O modelo não respondeu em {self.cfg.read_timeout_s:.0f}s (timeout)."
                    ) from exc
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    if cancelled():
                        raise ResponseCancelled("Resposta interrompida.") from exc
                    raise OllamaUnavailable(f"Conexão com o Ollama perdida: {exc}") from exc
                if not line:
                    if cancelled():
                        raise ResponseCancelled("Resposta interrompida.")
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    raise OllamaError(f"Erro do Ollama durante a resposta: {data['error']}")
                text = filt.feed((data.get("message") or {}).get("content", ""))
                if text:
                    yield text
                if data.get("done"):
                    if stats is not None:
                        stats.prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
                        stats.output_tokens = int(data.get("eval_count", 0) or 0)
                        stats.done_reason = str(data.get("done_reason", ""))
                        ns = 1e9
                        stats.total_s = (data.get("total_duration") or 0) / ns
                        stats.load_s = (data.get("load_duration") or 0) / ns
                        stats.prompt_eval_s = (data.get("prompt_eval_duration") or 0) / ns
                        stats.eval_s = (data.get("eval_duration") or 0) / ns
                    break
            tail = filt.flush()
            if tail:
                yield tail
        finally:
            with self._lock:
                self._active_conn = None
            conn.close()
