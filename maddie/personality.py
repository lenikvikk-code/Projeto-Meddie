"""Carrega a personalidade (prompt de sistema) de um arquivo separado."""

from __future__ import annotations

from pathlib import Path


class PersonalityError(Exception):
    """Arquivo de personalidade ausente ou vazio."""


def load_persona(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise PersonalityError(f"Arquivo de personalidade não encontrado: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise PersonalityError(f"Arquivo de personalidade vazio: {path}")
    return text
