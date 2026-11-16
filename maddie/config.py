"""Carregamento, validação e gravação da configuração (config.json)."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
THEMES = ("escuro", "claro")

# Perfis de resposta: atalhos para temperature/top_p. Não mexem no cache do prompt.
PROFILES = {
    "Preciso": {"temperature": 0.3, "top_p": 0.8},
    "Equilibrado": {"temperature": 0.7, "top_p": 0.8},
    "Criativo": {"temperature": 1.0, "top_p": 0.95},
}


class ConfigError(Exception):
    """Configuração inválida ou ilegível."""


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 11434
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 300.0

    model: str = "qwen3.5:4b"
    num_ctx: int = 8192
    num_predict: int = 768
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repeat_penalty: float | None = None    # None = padrão do modelo
    presence_penalty: float | None = None  # None = padrão do modelo
    num_thread: int | None = None          # None = o Ollama decide (núcleos físicos)
    num_batch: int | None = None           # None = padrão do Ollama
    think: bool = False
    keep_alive: str = "30m"
    seed: int | None = None  # None = aleatório (app); fixo só na avaliação

    memory_max_tokens: int = 5500   # teto do histórico (estimado); o motor reduz se faltar janela
    memory_trim_ratio: float = 0.6  # ao estourar, esquece em lote até sobrar esta fração
    memory_summarize: bool = True   # resume o lote antes de esquecer (fica só em RAM)
    summary_max_tokens: int = 200   # limite de saída do resumo

    base_memory_enabled: bool = False  # memória entre sessões: GRAVA EM DISCO; desligada por padrão
    base_memory_file: str = "memoria/memoria_base.json"
    base_memory_max_facts: int = 60

    persona_file: str = "persona.md"
    warm_up: bool = True              # processa a persona ao abrir, antes da 1ª mensagem
    strip_emoji: bool = True          # remove emoji da resposta se o usuário não usou nenhum
    unload_model_on_exit: bool = True

    ui_theme: str = "escuro"
    ui_font_size: int = 11
    ui_width: int = 1120
    ui_height: int = 740
    ui_show_stats: bool = True

    def model_options(self) -> dict:
        """Parâmetros enviados ao Ollama em 'options'."""
        opts = {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        for key in ("repeat_penalty", "presence_penalty", "num_thread", "num_batch", "seed"):
            value = getattr(self, key)
            if value is not None:
                opts[key] = value
        return opts

    def load_signature(self) -> tuple:
        """Campos que obrigam o Ollama a recarregar o modelo quando mudam."""
        return (self.model, self.num_ctx, self.num_thread, self.num_batch)

    def profile_name(self) -> str:
        for name, p in PROFILES.items():
            if abs(p["temperature"] - self.temperature) < 1e-6 and abs(p["top_p"] - self.top_p) < 1e-6:
                return name
        return "Personalizado"


def _validate(cfg: Config) -> None:
    if cfg.host not in LOOPBACK_HOSTS:
        raise ConfigError(
            f"Host '{cfg.host}' recusado: a Maddie só se conecta ao Ollama em "
            f"loopback ({', '.join(sorted(LOOPBACK_HOSTS))})."
        )
    name = cfg.model.lower()
    if "cloud" in name.split(":")[-1] or name.endswith("-cloud"):
        raise ConfigError(
            f"Modelo '{cfg.model}' recusado: modelos 'cloud' do Ollama são executados "
            "remotamente, o que viola o requisito de inferência local."
        )
    if not (1 <= cfg.port <= 65535):
        raise ConfigError(f"Porta inválida: {cfg.port}")
    for field_name in ("num_ctx", "num_predict", "memory_max_tokens", "summary_max_tokens",
                       "base_memory_max_facts"):
        if getattr(cfg, field_name) < 1:
            raise ConfigError(f"'{field_name}' deve ser >= 1")
    if cfg.num_predict >= cfg.num_ctx:
        raise ConfigError("'num_predict' deve ser menor que 'num_ctx'")
    if cfg.memory_max_tokens + cfg.num_predict >= cfg.num_ctx:
        raise ConfigError("'memory.max_tokens' + 'num_predict' deve ser menor que 'num_ctx' "
                          "(sobra espaço para a persona e a mensagem nova)")
    if not 0 < cfg.memory_trim_ratio < 1:
        raise ConfigError("'memory.trim_ratio' deve estar entre 0 e 1")
    if cfg.connect_timeout_s <= 0 or cfg.read_timeout_s <= 0:
        raise ConfigError("Timeouts devem ser positivos")
    if not 0 <= cfg.temperature <= 2:
        raise ConfigError("'temperature' deve estar entre 0 e 2")
    if not 0 < cfg.top_p <= 1:
        raise ConfigError("'top_p' deve estar entre 0 (exclusivo) e 1")
    if cfg.top_k < 1:
        raise ConfigError("'top_k' deve ser >= 1")
    if cfg.repeat_penalty is not None and not 0.5 <= cfg.repeat_penalty <= 2:
        raise ConfigError("'repeat_penalty' deve estar entre 0.5 e 2")
    if cfg.presence_penalty is not None and not -2 <= cfg.presence_penalty <= 2:
        raise ConfigError("'presence_penalty' deve estar entre -2 e 2")
    if cfg.num_thread is not None and not 1 <= cfg.num_thread <= 256:
        raise ConfigError("'num_thread' deve estar entre 1 e 256 (ou null para automático)")
    if cfg.num_batch is not None and not 1 <= cfg.num_batch <= 4096:
        raise ConfigError("'num_batch' deve estar entre 1 e 4096 (ou null para o padrão)")
    if cfg.ui_theme not in THEMES:
        raise ConfigError(f"'ui.theme' deve ser um de: {', '.join(THEMES)}")
    if not 8 <= cfg.ui_font_size <= 24:
        raise ConfigError("'ui.font_size' deve estar entre 8 e 24")
    if cfg.ui_width < 640 or cfg.ui_height < 480:
        raise ConfigError("Janela mínima: 640x480")
    if Path(cfg.base_memory_file).is_absolute() or ".." in Path(cfg.base_memory_file).parts:
        raise ConfigError("'base_memory.file' deve ser um caminho relativo dentro da pasta da Maddie")


def _opt(value, cast):
    return None if value is None else cast(value)


def from_dict(data: dict) -> Config:
    """Monta Config a partir do JSON (campos ausentes usam o padrão)."""
    if not isinstance(data, dict):
        raise ConfigError("config.json deve conter um objeto JSON")
    o = data.get("ollama", {}) or {}
    m = data.get("model", {}) or {}
    mem = data.get("memory", {}) or {}
    base = data.get("base_memory", {}) or {}
    ui = data.get("ui", {}) or {}
    d = Config()
    try:
        cfg = Config(
            host=str(o.get("host", d.host)),
            port=int(o.get("port", d.port)),
            connect_timeout_s=float(o.get("connect_timeout_s", d.connect_timeout_s)),
            read_timeout_s=float(o.get("read_timeout_s", d.read_timeout_s)),
            model=str(m.get("name", d.model)),
            num_ctx=int(m.get("num_ctx", d.num_ctx)),
            num_predict=int(m.get("num_predict", d.num_predict)),
            temperature=float(m.get("temperature", d.temperature)),
            top_p=float(m.get("top_p", d.top_p)),
            top_k=int(m.get("top_k", d.top_k)),
            repeat_penalty=_opt(m.get("repeat_penalty", d.repeat_penalty), float),
            presence_penalty=_opt(m.get("presence_penalty", d.presence_penalty), float),
            num_thread=_opt(m.get("num_thread", d.num_thread), int),
            num_batch=_opt(m.get("num_batch", d.num_batch), int),
            think=bool(m.get("think", d.think)),
            keep_alive=str(m.get("keep_alive", d.keep_alive)),
            memory_max_tokens=int(mem.get("max_tokens", d.memory_max_tokens)),
            memory_trim_ratio=float(mem.get("trim_ratio", d.memory_trim_ratio)),
            memory_summarize=bool(mem.get("summarize", d.memory_summarize)),
            summary_max_tokens=int(mem.get("summary_max_tokens", d.summary_max_tokens)),
            base_memory_enabled=bool(base.get("enabled", d.base_memory_enabled)),
            base_memory_file=str(base.get("file", d.base_memory_file)),
            base_memory_max_facts=int(base.get("max_facts", d.base_memory_max_facts)),
            persona_file=str(data.get("persona_file", d.persona_file)),
            warm_up=bool(data.get("warm_up", d.warm_up)),
            strip_emoji=bool(data.get("strip_emoji", d.strip_emoji)),
            unload_model_on_exit=bool(data.get("unload_model_on_exit", d.unload_model_on_exit)),
            ui_theme=str(ui.get("theme", d.ui_theme)),
            ui_font_size=int(ui.get("font_size", d.ui_font_size)),
            ui_width=int(ui.get("width", d.ui_width)),
            ui_height=int(ui.get("height", d.ui_height)),
            ui_show_stats=bool(ui.get("show_stats", d.ui_show_stats)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Valor inválido em config.json: {exc}") from exc
    _validate(cfg)
    return cfg


def to_dict(cfg: Config) -> dict:
    """Estrutura gravada em config.json (o 'seed' é só da avaliação e não é gravado)."""
    return {
        "ollama": {
            "host": cfg.host, "port": cfg.port,
            "connect_timeout_s": cfg.connect_timeout_s, "read_timeout_s": cfg.read_timeout_s,
        },
        "model": {
            "name": cfg.model, "num_ctx": cfg.num_ctx, "num_predict": cfg.num_predict,
            "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty, "presence_penalty": cfg.presence_penalty,
            "num_thread": cfg.num_thread, "num_batch": cfg.num_batch,
            "think": cfg.think, "keep_alive": cfg.keep_alive,
        },
        "memory": {
            "max_tokens": cfg.memory_max_tokens, "trim_ratio": cfg.memory_trim_ratio,
            "summarize": cfg.memory_summarize, "summary_max_tokens": cfg.summary_max_tokens,
        },
        "base_memory": {
            "enabled": cfg.base_memory_enabled, "file": cfg.base_memory_file,
            "max_facts": cfg.base_memory_max_facts,
        },
        "persona_file": cfg.persona_file,
        "warm_up": cfg.warm_up,
        "strip_emoji": cfg.strip_emoji,
        "unload_model_on_exit": cfg.unload_model_on_exit,
        "ui": {
            "theme": cfg.ui_theme, "font_size": cfg.ui_font_size,
            "width": cfg.ui_width, "height": cfg.ui_height, "show_stats": cfg.ui_show_stats,
        },
    }


def replace(cfg: Config, **changes) -> Config:
    """Cópia validada com os campos alterados (levanta ConfigError se ficar inválida)."""
    new = dataclasses.replace(cfg, **changes)
    _validate(new)
    return new


def load(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config.json mal formatado (linha {exc.lineno}): {exc.msg}") from exc
    return from_dict(data)


def save(path: str | Path, cfg: Config) -> None:
    """Grava a configuração de forma atômica (arquivo temporário + troca)."""
    _validate(cfg)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_dict(cfg), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
