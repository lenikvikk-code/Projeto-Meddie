"""Ponto de entrada da Maddie I.A."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from maddie import config, personality  # noqa: E402


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
        persona = personality.load_persona(config.project_path(BASE_DIR, cfg.persona_file, "persona_file"))
    except (config.ConfigError, personality.PersonalityError) as exc:
        _fatal(str(exc))
        return
    try:
        from maddie import gui
    except ImportError as exc:
        if exc.name not in {"_tkinter", "tkinter"}:
            raise
        if sys.platform.startswith("linux"):
            msg = ("O Tkinter não está instalado neste Python.\n"
                   "Instale o pacote da sua distribuição, por exemplo:\n"
                   "  Fedora: sudo dnf install python3-tkinter\n"
                   "  Debian/Ubuntu: sudo apt install python3-tk")
        else:
            msg = "O Tkinter não está disponível. Reinstale o Python com o suporte a Tcl/Tk."
        print(msg, file=sys.stderr)
        sys.exit(1)
    gui.run(cfg, persona, BASE_DIR)


if __name__ == "__main__":
    main()
