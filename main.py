"""Ponto de entrada da Maddie I.A."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from maddie import config, gui, personality  # noqa: E402


def _fatal(msg: str) -> None:
    """Mostra erro de inicialização numa caixa de diálogo (funciona com pythonw)."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Maddie I.A. — erro ao iniciar", msg)
        root.destroy()
    except Exception:
        pass
    print(msg, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    try:
        cfg = config.load(BASE_DIR / "config.json")
        persona = personality.load_persona(BASE_DIR / cfg.persona_file)
    except (config.ConfigError, personality.PersonalityError) as exc:
        _fatal(str(exc))
        return
    gui.run(cfg, persona, BASE_DIR)


if __name__ == "__main__":
    main()
