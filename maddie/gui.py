"""Interface Tkinter da Maddie 0.3: barra lateral, conversa com markdown, ajustes e memória.

Regras de concorrência: a rede roda em threads; elas só falam com a interface
pela fila `self.events`. Só uma tarefa de modelo roda por vez (preparar,
responder, organizar memória ou extrair fatos); o estado da janela diz qual.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import __version__, config as config_mod, markdown_lite
from .config import PROFILES, Config, ConfigError
from .engine import ChatSession
from .ollama_client import OllamaError, ResponseCancelled

THEMES = {
    "escuro": dict(bg="#101013", panel="#17171b", panel2="#212127", border="#2b2b33", fg="#e8e8ec",
                   muted="#8b8b96", accent="#dc143c", accent_dark="#9b0f2b", accent_fg="#ffffff",
                   error="#ff6b6b", ok="#3fb950", warn="#d29922", code_bg="#1e1e25", entry_bg="#1b1b20",
                   user="#f0b8c3"),
    "claro": dict(bg="#f7f7f9", panel="#ececf1", panel2="#dfdfe6", border="#cfcfd8", fg="#1b1b21",
                  muted="#666672", accent="#c01033", accent_dark="#8f0c26", accent_fg="#ffffff",
                  error="#c62828", ok="#1a7f37", warn="#9a6700", code_bg="#e4e4eb", entry_bg="#ffffff",
                  user="#8f0c26"),
}

STATUS = {
    "disponivel": ("Disponível", "ok"),
    "carregando": ("Preparando", "warn"),
    "respondendo": ("Respondendo", "accent"),
    "organizando": ("Organizando memória", "warn"),
    "indisponivel": ("Indisponível", "muted"),
}
BUSY = ("carregando", "respondendo", "organizando")

RECHECK_MS = 5000
POLL_MS = 30
TICK_MS = 500
FAMILY = "DejaVu Sans" if sys.platform.startswith("linux") else "Segoe UI"
MONO = "DejaVu Sans Mono" if sys.platform.startswith("linux") else "Consolas"


def num(value: float, digits: int = 1) -> str:
    """Número com vírgula decimal."""
    return f"{value:.{digits}f}".replace(".", ",")


def set_titlebar(window, dark: bool) -> None:
    """Barra de título escura/clara no Windows 10/11 (acompanha o tema). Ignorado se não der."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        value = ctypes.c_int(1 if dark else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
    except Exception:
        pass


def enable_dpi_awareness() -> None:
    """Texto nítido em telas com escala > 100% (Windows). Ignorado em outros sistemas."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


class MaddieApp:
    def __init__(self, root: tk.Tk, cfg: Config, persona: str, base_dir: Path):
        self.root = root
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.config_path = self.base_dir / "config.json"
        self.session = ChatSession(cfg, persona, self.base_dir)
        self.client = self.session.client
        self.events: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.transcript: list[dict] = []
        self.state = "indisponivel"
        self.closing = False
        self.pending: str | None = None      # mensagem enviada enquanto o modelo se preparava
        self.pending_regen = False
        self._recheck_job = None
        self._last_error = None
        self._t_state = time.perf_counter()
        self._streaming = False
        self._placeholder = False
        self._dialogs: dict[str, tk.Toplevel] = {}
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96.0)

        root.title(f"Maddie I.A. {__version__}")
        w = min(self.px(cfg.ui_width), root.winfo_screenwidth() - 40)
        h = min(self.px(cfg.ui_height), root.winfo_screenheight() - 80)
        root.geometry(f"{w}x{h}")
        root.minsize(self.px(700), self.px(480))
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        if self.session.store.load_error:
            self._note(self.session.store.load_error, "error")
        self._set_state("indisponivel")
        self.root.after(POLL_MS, self._poll_events)
        self.root.after(TICK_MS, self._tick)
        self._start_prepare()

    # ---------------- utilidades ----------------
    def px(self, n: int) -> int:
        return int(round(n * self.scale))

    @property
    def c(self) -> dict:
        return THEMES[self.cfg.ui_theme]

    def font(self, delta: int = 0, *style: str, family: str = FAMILY) -> tuple:
        return (family, self.cfg.ui_font_size + delta, *style)

    # ---------------- construção ----------------
    def _build_ui(self) -> None:
        c = self.c
        r = self.root
        r.configure(bg=c["bg"])
        set_titlebar(r, self.cfg.ui_theme == "escuro")
        self._style_ttk()
        self.body = tk.Frame(r, bg=c["bg"])
        self.body.pack(fill="both", expand=True)
        self._build_sidebar(self.body)
        tk.Frame(self.body, bg=c["border"], width=1).pack(side="left", fill="y")
        self._build_main(self.body)
        self._bind_keys()
        self._render_all()
        self._update_side()

    def _style_ttk(self) -> None:
        c = self.c
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["entry_bg"],
                     bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                     troughcolor=c["panel2"], font=self.font(-1))
        st.configure("Vertical.TScrollbar", background=c["panel2"], troughcolor=c["bg"],
                     arrowcolor=c["muted"], bordercolor=c["bg"], relief="flat")
        st.map("Vertical.TScrollbar", background=[("active", c["border"])])
        st.configure("TCombobox", fieldbackground=c["entry_bg"], background=c["panel2"],
                     foreground=c["fg"], arrowcolor=c["fg"], selectbackground=c["entry_bg"],
                     selectforeground=c["fg"], padding=4)
        st.map("TCombobox", fieldbackground=[("readonly", c["entry_bg"])],
               foreground=[("readonly", c["fg"])], selectbackground=[("readonly", c["entry_bg"])],
               selectforeground=[("readonly", c["fg"])])
        st.configure("TSpinbox", fieldbackground=c["entry_bg"], foreground=c["fg"], arrowcolor=c["fg"],
                     background=c["panel2"], padding=4)
        st.configure("TCombobox", lightcolor=c["panel2"], darkcolor=c["panel2"], arrowsize=self.px(12))
        st.map("TCombobox",
               background=[("pressed", c["border"]), ("active", c["border"]), ("!disabled", c["panel2"])],
               arrowcolor=[("!disabled", c["fg"])], bordercolor=[("focus", c["accent_dark"])],
               lightcolor=[("!disabled", c["panel2"])], darkcolor=[("!disabled", c["panel2"])])
        st.configure("Horizontal.TScale", background=c["accent"], troughcolor=c["panel2"],
                     bordercolor=c["panel2"], lightcolor=c["accent"], darkcolor=c["accent"])
        st.map("Horizontal.TScale", background=[("active", c["accent_dark"])])
        st.configure("TCheckbutton", background=c["bg"], foreground=c["fg"], focuscolor=c["bg"],
                     indicatorbackground=c["entry_bg"], indicatorforeground=c["accent_fg"],
                     upperbordercolor=c["border"], lowerbordercolor=c["border"], padding=2)
        st.map("TCheckbutton", background=[("active", c["bg"])], foreground=[("active", c["fg"])],
               indicatorbackground=[("selected", c["accent"]), ("active", c["panel2"])])
        st.configure("Treeview", background=c["entry_bg"], fieldbackground=c["entry_bg"],
                     foreground=c["fg"], rowheight=self.px(26), bordercolor=c["border"])
        st.configure("Treeview.Heading", background=c["panel2"], foreground=c["fg"], relief="flat",
                     font=self.font(-1, "bold"))
        st.map("Treeview", background=[("selected", c["accent_dark"])], foreground=[("selected", "#ffffff")])
        self.root.option_add("*TCombobox*Listbox.background", c["entry_bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent_dark"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def button(self, parent, text, cmd, kind="flat", **kw) -> tk.Button:
        """Botão plano no tema. kind: 'primary', 'flat' ou 'ghost'."""
        c = self.c
        bg, fg, hover = {
            "primary": (c["accent"], c["accent_fg"], c["accent_dark"]),
            "flat": (c["panel2"], c["fg"], c["border"]),
            "ghost": (parent.cget("bg"), c["muted"], c["panel2"]),
        }[kind]
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg, activebackground=hover,
                      activeforeground=fg if kind != "ghost" else c["fg"], relief="flat", bd=0,
                      font=kw.pop("font", self.font(-1, "bold" if kind == "primary" else "normal")),
                      padx=kw.pop("padx", self.px(12)), pady=kw.pop("pady", self.px(6)),
                      cursor="hand2", disabledforeground=c["muted"], highlightthickness=0, **kw)
        b._bg, b._hover = bg, hover
        b.bind("<Enter>", lambda e: b.configure(bg=b._hover) if b["state"] != "disabled" else None)
        b.bind("<Leave>", lambda e: b.configure(bg=b._bg))
        return b

    def _build_sidebar(self, parent) -> None:
        c = self.c
        side = tk.Frame(parent, bg=c["panel"], width=self.px(264))
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        pad = self.px(16)

        tk.Label(side, text="MADDIE", bg=c["panel"], fg=c["accent"],
                 font=(FAMILY, self.cfg.ui_font_size + 9, "bold")).pack(anchor="w", padx=pad, pady=(pad, 0))
        tk.Label(side, text=f"I.A. local · v{__version__}", bg=c["panel"], fg=c["muted"],
                 font=self.font(-2)).pack(anchor="w", padx=pad)
        tk.Frame(side, bg=c["accent"], height=2).pack(fill="x", padx=pad, pady=(self.px(10), self.px(12)))

        # Estado
        row = tk.Frame(side, bg=c["panel"])
        row.pack(fill="x", padx=pad)
        self.status_dot = tk.Canvas(row, width=self.px(12), height=self.px(12), bg=c["panel"],
                                    highlightthickness=0)
        self.status_dot.pack(side="left")
        d = self.px(11)
        self._dot = self.status_dot.create_oval(1, 1, d, d, fill=c["muted"], outline="")
        self.status_label = tk.Label(row, text="", bg=c["panel"], fg=c["fg"], font=self.font(-1, "bold"))
        self.status_label.pack(side="left", padx=(self.px(6), 0))
        self.model_label = tk.Label(side, text="", bg=c["panel"], fg=c["muted"], font=self.font(-2),
                                    anchor="w", justify="left", wraplength=self.px(230))
        self.model_label.pack(fill="x", padx=pad, pady=(self.px(4), self.px(12)))

        self.new_btn = self.button(side, "+  Nova conversa", self._new_conversation, "primary")
        self.new_btn.pack(fill="x", padx=pad)

        # Perfil de resposta
        self._section(side, "ESTILO DA RESPOSTA")
        prof = tk.Frame(side, bg=c["panel"])
        prof.pack(fill="x", padx=pad)
        self.profile_btns: dict[str, tk.Button] = {}
        for i, name in enumerate(PROFILES):
            b = self.button(prof, name, lambda n=name: self._set_profile(n), "flat",
                            padx=self.px(2), pady=self.px(5), font=self.font(-2))
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else self.px(4), 0))
            prof.grid_columnconfigure(i, weight=1, uniform="p")
            self.profile_btns[name] = b
        self.profile_hint = tk.Label(side, text="", bg=c["panel"], fg=c["muted"], font=self.font(-2),
                                     anchor="w")
        self.profile_hint.pack(fill="x", padx=pad, pady=(self.px(4), 0))

        # Memória da conversa
        self._section(side, "MEMÓRIA DA CONVERSA")
        self.mem_bar = tk.Canvas(side, height=self.px(6), bg=c["panel2"], highlightthickness=0)
        self.mem_bar.pack(fill="x", padx=pad)
        self.mem_bar.bind("<Configure>", lambda e: self._draw_mem_bar())
        self.memory_label = tk.Label(side, text="", bg=c["panel"], fg=c["muted"], font=self.font(-2),
                                     anchor="w", justify="left", wraplength=self.px(230))
        self.memory_label.pack(fill="x", padx=pad, pady=(self.px(5), 0))

        # Última resposta
        self._section(side, "ÚLTIMA RESPOSTA")
        self.stats_label = tk.Label(side, text="—", bg=c["panel"], fg=c["muted"], font=self.font(-2),
                                    anchor="w", justify="left", wraplength=self.px(230))
        self.stats_label.pack(fill="x", padx=pad)

        # Rodapé
        foot = tk.Frame(side, bg=c["panel"])
        foot.pack(side="bottom", fill="x", padx=pad, pady=pad)
        self.base_btn = self.button(foot, "Memória permanente", self._open_memory, "flat", anchor="w")
        self.base_btn.pack(fill="x", pady=(0, self.px(6)))
        self.button(foot, "Exportar conversa", self._export, "flat", anchor="w").pack(
            fill="x", pady=(0, self.px(6)))
        self.button(foot, "Ajustes", self._open_settings, "flat", anchor="w").pack(fill="x")
        tk.Label(foot, text="100% local · nada sai do computador", bg=c["panel"], fg=c["muted"],
                 font=self.font(-3)).pack(anchor="w", pady=(self.px(10), 0))

    def _section(self, parent, title: str) -> None:
        tk.Label(parent, text=title, bg=self.c["panel"], fg=self.c["muted"],
                 font=self.font(-3, "bold")).pack(anchor="w", padx=self.px(16),
                                                  pady=(self.px(18), self.px(6)))

    def _build_main(self, parent) -> None:
        c = self.c
        main = tk.Frame(parent, bg=c["bg"])
        main.pack(side="left", fill="both", expand=True)

        # Entrada (embaixo; empacotada antes para a conversa ocupar o resto)
        bottom = tk.Frame(main, bg=c["bg"])
        bottom.pack(side="bottom", fill="x", padx=self.px(18), pady=(self.px(4), self.px(14)))
        self.hint_label = tk.Label(bottom, text="Enter envia · Shift+Enter quebra a linha · Esc interrompe",
                                   bg=c["bg"], fg=c["muted"], font=self.font(-3), anchor="w")
        self.hint_label.pack(side="bottom", fill="x", pady=(self.px(4), 0))
        self.send_btn = self.button(bottom, "Enviar", self._send_or_stop, "primary", width=9,
                                    font=self.font(0, "bold"))
        self.send_btn.pack(side="right", fill="y", padx=(self.px(8), 0))
        self.entry_border = tk.Frame(bottom, bg=c["border"], padx=1, pady=1)
        self.entry_border.pack(side="left", fill="both", expand=True)
        self.entry = tk.Text(self.entry_border, height=2, wrap="word", bg=c["entry_bg"], fg=c["fg"],
                             insertbackground=c["accent"], font=self.font(), relief="flat",
                             padx=self.px(10), pady=self.px(8), undo=True,
                             selectbackground=c["accent_dark"], selectforeground="#ffffff")
        self.entry.pack(fill="both", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<KeyRelease>", lambda e: self._autosize_entry())
        self.entry.bind("<FocusIn>", lambda e: self.entry_border.configure(bg=c["accent_dark"]))
        self.entry.bind("<FocusOut>", lambda e: self.entry_border.configure(bg=c["border"]))
        self.entry.focus_set()

        # Ações sobre a última resposta
        actions = tk.Frame(main, bg=c["bg"])
        actions.pack(side="bottom", fill="x", padx=self.px(18))
        self.regen_btn = self.button(actions, "↻ Regenerar", self._regenerate, "ghost",
                                     padx=self.px(8), pady=self.px(3), font=self.font(-2))
        self.regen_btn.pack(side="left")
        self.copy_btn = self.button(actions, "Copiar resposta", self._copy_last, "ghost",
                                    padx=self.px(8), pady=self.px(3), font=self.font(-2))
        self.copy_btn.pack(side="left")
        self.queue_label = tk.Label(actions, text="", bg=c["bg"], fg=c["warn"], font=self.font(-2))
        self.queue_label.pack(side="right")

        # Conversa
        chat_frame = tk.Frame(main, bg=c["bg"])
        chat_frame.pack(side="top", fill="both", expand=True, padx=(self.px(18), self.px(4)),
                        pady=(self.px(12), self.px(4)))
        scroll = ttk.Scrollbar(chat_frame, orient="vertical")
        self.chat = tk.Text(chat_frame, wrap="word", bg=c["bg"], fg=c["fg"], font=self.font(),
                            relief="flat", padx=self.px(6), pady=self.px(6), state="disabled",
                            yscrollcommand=scroll.set, selectbackground=c["accent_dark"],
                            selectforeground="#ffffff", cursor="arrow", spacing1=1, spacing3=3,
                            highlightthickness=0, insertwidth=0)
        scroll.config(command=self.chat.yview)
        scroll.pack(side="right", fill="y")
        self.chat.pack(side="left", fill="both", expand=True)
        self._configure_tags()
        self.chat_menu = tk.Menu(self.chat, tearoff=0)
        self.chat_menu.add_command(label="Copiar seleção", command=self._copy_selection)
        self.chat_menu.add_command(label="Copiar última resposta", command=self._copy_last)
        self.chat_menu.add_separator()
        self.chat_menu.add_command(label="Selecionar tudo",
                                   command=lambda: self.chat.tag_add("sel", "1.0", "end-1c"))
        self.chat.bind("<Button-3>", lambda e: self.chat_menu.tk_popup(e.x_root, e.y_root))

    def _configure_tags(self) -> None:
        c, t = self.c, self.chat
        ind = self.px(16)
        t.tag_configure("user_name", foreground=c["user"], font=self.font(0, "bold"), spacing1=self.px(10))
        t.tag_configure("maddie_name", foreground=c["accent"], font=self.font(0, "bold"),
                        spacing1=self.px(10))
        t.tag_configure("time", foreground=c["muted"], font=self.font(-3))
        t.tag_configure("user", foreground=c["fg"], lmargin1=ind, lmargin2=ind, rmargin=ind)
        t.tag_configure("maddie", foreground=c["fg"], lmargin1=ind, lmargin2=ind, rmargin=ind)
        t.tag_configure("list", lmargin1=ind + self.px(8), lmargin2=ind + self.px(30))
        t.tag_configure("list2", lmargin1=ind + self.px(30), lmargin2=ind + self.px(52))
        t.tag_configure("quote", foreground=c["muted"], lmargin1=ind + self.px(14), lmargin2=ind + self.px(14),
                        font=self.font(0, "italic"))
        t.tag_configure("heading", font=self.font(2, "bold"), spacing1=self.px(6))
        t.tag_configure("bold", font=self.font(0, "bold"))
        t.tag_configure("italic", font=self.font(0, "italic"))
        t.tag_configure("code", font=self.font(-1, family=MONO), background=c["code_bg"])
        t.tag_configure("codeblock", font=self.font(-1, family=MONO), background=c["code_bg"],
                        lmargin1=ind + self.px(8), lmargin2=ind + self.px(8), rmargin=ind)
        t.tag_configure("muted", foreground=c["muted"])
        t.tag_configure("stats", foreground=c["muted"], font=self.font(-3), lmargin1=ind, lmargin2=ind)
        t.tag_configure("note", foreground=c["muted"], font=self.font(-2), spacing1=self.px(8),
                        lmargin1=ind, lmargin2=ind)
        t.tag_configure("error", foreground=c["error"], font=self.font(-2), spacing1=self.px(8),
                        lmargin1=ind, lmargin2=ind)
        t.tag_configure("welcome", foreground=c["muted"], font=self.font(1), justify="center",
                        spacing1=self.px(60))
        t.tag_configure("welcome2", foreground=c["muted"], font=self.font(-2), justify="center",
                        spacing1=self.px(8))
        for tag in ("bold", "italic", "code", "heading", "codeblock", "list", "list2", "quote", "sel"):
            t.tag_raise(tag)

    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<Escape>", lambda e: self._stop())
        r.bind("<Control-n>", lambda e: self._new_conversation())
        r.bind("<Control-comma>", lambda e: self._open_settings())
        r.bind("<Control-plus>", lambda e: self._bump_font(1))
        r.bind("<Control-equal>", lambda e: self._bump_font(1))
        r.bind("<Control-minus>", lambda e: self._bump_font(-1))

    def rebuild(self) -> None:
        """Refaz a janela (tema ou fonte mudaram) mantendo a conversa e o texto digitado."""
        draft = self.entry.get("1.0", "end-1c")
        self.body.destroy()
        self._build_ui()
        self.entry.insert("1.0", draft)
        if self._streaming:  # resposta em andamento: recria a região; o texto volta inteiro ao terminar
            with self._edit():
                self._insert("Maddie", "maddie_name")
                self._insert(f"   {datetime.now():%H:%M}\n", "time")
                self.chat.mark_set("reply_start", "end-1c")
                self.chat.mark_gravity("reply_start", "left")
                self._insert("escrevendo…", "maddie", "muted")
            self._placeholder = True
        self._set_state(self.state)

    # ---------------- conversa: desenho ----------------
    def _insert(self, text: str, *tags: str) -> None:
        self.chat.insert("end", text, tags)

    def _edit(self):
        """Contexto para alterar a área de conversa (que fica somente leitura)."""
        app = self

        class _Ctx:
            def __enter__(self_inner):
                app.chat.configure(state="normal")

            def __exit__(self_inner, *exc):
                app.chat.configure(state="disabled")
        return _Ctx()

    def _render_all(self) -> None:
        with self._edit():
            self.chat.delete("1.0", "end")
            if not self.transcript:
                self._insert("Olá. Sobre o que vamos conversar?\n", "welcome")
                self._insert("A conversa fica só na memória do computador e some ao fechar.\n", "welcome2")
            for entry in self.transcript:
                self._draw_entry(entry)
        self.chat.see("end")

    def _draw_entry(self, e: dict) -> None:
        role = e["role"]
        if role == "user":
            self._insert("Você", "user_name")
            self._insert(f"   {e['time']}\n", "time")
            self._insert(e["text"].rstrip() + "\n", "user")
        elif role == "maddie":
            self._insert("Maddie", "maddie_name")
            self._insert(f"   {e['time']}\n", "time")
            self._draw_reply_body(e)
        else:
            self._insert(e["text"].rstrip() + "\n", "error" if role == "error" else "note")

    def _draw_reply_body(self, e: dict) -> None:
        for text, tags in markdown_lite.parse(e["text"].rstrip(), "maddie"):
            self._insert(text, *tags)
        self._insert("\n", "maddie")
        if e.get("stats") and self.cfg.ui_show_stats:
            self._insert(e["stats"] + "\n", "stats")

    def _add(self, role: str, text: str, **extra) -> dict:
        entry = {"role": role, "text": text, "time": datetime.now().strftime("%H:%M"), **extra}
        first = not self.transcript
        self.transcript.append(entry)
        if first:
            self._render_all()
        else:
            with self._edit():
                self._draw_entry(entry)
            self.chat.see("end")
        return entry

    def _note(self, text: str, role: str = "note") -> None:
        self._add(role, text)

    # ---------------- estado ----------------
    def _set_state(self, state: str) -> None:
        if state != self.state:
            self._t_state = time.perf_counter()
        self.state = state
        label, color = STATUS[state]
        self.status_label.configure(text=label)
        self.status_dot.itemconfigure(self._dot, fill=self.c[color])
        responding = state == "respondendo"
        self.send_btn.configure(text="Parar" if responding else "Enviar")
        can_send = responding or state == "disponivel" or (state in ("carregando", "organizando")
                                                            and self.pending is None)
        self.send_btn.configure(state="normal" if can_send else "disabled")
        idle = state in ("disponivel", "indisponivel")
        self.new_btn.configure(state="normal" if idle else "disabled")
        self._update_actions()
        self.queue_label.configure(
            text="mensagem na fila: será enviada assim que a Maddie estiver pronta" if self.pending else "")

    def _update_actions(self) -> None:
        has_reply = any(e["role"] == "maddie" for e in self.transcript)
        can_regen = self.state == "disponivel" and bool(self.transcript) and \
            self.transcript[-1]["role"] == "maddie" and len(self.session.memory) > 0
        self.regen_btn.configure(state="normal" if can_regen else "disabled")
        self.copy_btn.configure(state="normal" if has_reply else "disabled")

    def _tick(self) -> None:
        if self.closing:
            return
        if self.state in BUSY:
            secs = int(time.perf_counter() - self._t_state)
            label = STATUS[self.state][0]
            self.status_label.configure(text=f"{label} · {secs} s" if secs >= 2 else label)
        self.root.after(TICK_MS, self._tick)

    def _draw_mem_bar(self) -> None:
        m = self.session.memory
        frac = min(1.0, m.tokens() / m.max_tokens) if m.max_tokens else 0.0
        w, h = self.mem_bar.winfo_width(), self.mem_bar.winfo_height()
        self.mem_bar.delete("all")
        if frac > 0:
            color = self.c["warn"] if frac > 0.85 else self.c["accent"]
            self.mem_bar.create_rectangle(0, 0, max(2, int(w * frac)), h, fill=color, outline="")

    def _update_side(self) -> None:
        cfg, m = self.cfg, self.session.memory
        self.model_label.configure(text=f"{cfg.model}\njanela de {cfg.num_ctx} tokens")
        pct = min(100, round(100 * m.tokens() / m.max_tokens)) if m.max_tokens else 0
        n = len(m)
        lines = [f"{n} {'troca' if n == 1 else 'trocas'} · {pct}% ocupada"]
        if m.dropped:
            kind = "resumida" if m.summary else "esquecida"
            plural = "" if m.dropped == 1 else "s"
            lines.append(f"{m.dropped} troca{plural} antiga{plural} {kind}{plural}")
        self.memory_label.configure(text="\n".join(lines))
        self._draw_mem_bar()
        current = cfg.profile_name()
        for name, b in self.profile_btns.items():
            active = name == current
            b._bg = self.c["accent_dark"] if active else self.c["panel2"]
            b.configure(bg=b._bg, fg="#ffffff" if active else self.c["fg"])
        self.profile_hint.configure(
            text=f"temperatura {num(cfg.temperature, 2)} · top-p {num(cfg.top_p, 2)}"
                 + ("" if current != "Personalizado" else " (personalizado)"))
        k = len(self.session.store.facts) if cfg.base_memory_enabled else 0
        self.base_btn.configure(
            text=f"Memória permanente ({k} {'fato' if k == 1 else 'fatos'})" if cfg.base_memory_enabled
            else "Memória permanente (desligada)")

    def _autosize_entry(self) -> None:
        try:
            lines = self.entry.count("1.0", "end", "displaylines")
            n = lines[0] if isinstance(lines, tuple) else int(lines or 1)
        except tk.TclError:
            n = 1
        self.entry.configure(height=max(2, min(8, n)))

    # ---------------- eventos das threads ----------------
    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(POLL_MS, self._poll_events)

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "state":
            self._set_state(payload)
        elif kind == "note":
            self._note(payload)
        elif kind == "unavailable":
            self._set_state("indisponivel")
            if payload != self._last_error:
                self._note(payload, "error")
                self._last_error = payload
            self._schedule_recheck()
        elif kind == "ready":
            self._last_error = None
            self._update_side()
            self._become_ready()
        elif kind == "chunk":
            self._stream_chunk(payload)
        elif kind == "done":
            self._finish_reply(payload, stopped=False)
        elif kind == "stopped":
            self._finish_reply(payload, stopped=True)
        elif kind == "reply_paused":
            self._set_state("organizando")
        elif kind == "reply_resumed":
            self._after_maintenance(payload, resume=True)
        elif kind == "reply_resume_error":
            info, error = payload
            self._finish_reply(info, stopped=False)
            self._note(f"Não foi possível continuar a resposta: {error}", "error")
        elif kind == "chat_error":
            self._abort_reply()
            self._note(f"Erro: {payload}", "error")
            self._start_prepare()
        elif kind == "maintained":
            self._after_maintenance(payload)
        elif kind == "extracted":
            self._after_extraction(*payload)
        elif kind == "task_error":
            self._note(f"Erro: {payload}", "error")
            self._become_ready()

    def _become_ready(self) -> None:
        """Modelo pronto e ocioso: envia o que estiver na fila."""
        self._set_state("disponivel")
        if self.pending is not None:
            text, self.pending = self.pending, None
            self._begin_reply(text)

    # ---------------- preparo do modelo ----------------
    def _start_prepare(self) -> None:
        if self.closing:
            return
        threading.Thread(target=self._prepare_worker, daemon=True).start()

    def _prepare_worker(self) -> None:
        try:
            self.client.ensure_ready()
            self.events.put(("state", "carregando"))
            if self.cfg.warm_up:
                self.session.warm_up(cancelled=lambda: self.closing)
            else:
                self.client.load_model()
            self.events.put(("ready", None))
        except ResponseCancelled:
            pass
        except OllamaError as exc:
            self.events.put(("unavailable", str(exc)))
        except Exception as exc:
            self.events.put(("unavailable", f"Erro inesperado ao preparar o modelo: {exc!r}"))

    def _schedule_recheck(self) -> None:
        if self.closing or self._recheck_job is not None:
            return

        def job():
            self._recheck_job = None
            if self.state == "indisponivel":
                self._start_prepare()

        self._recheck_job = self.root.after(RECHECK_MS, job)

    # ---------------- enviar / responder ----------------
    def _on_enter(self, event):
        if event.state & 0x0001:  # Shift pressionado: nova linha
            return None
        if self.state != "respondendo":
            self._send()
        return "break"

    def _send_or_stop(self) -> None:
        if self.state == "respondendo":
            self._stop()
        else:
            self._send()

    def _send(self) -> None:
        text = self.entry.get("1.0", "end").strip()
        if not text:
            return
        if self.state == "disponivel":
            self.entry.delete("1.0", "end")
            self._autosize_entry()
            self._add("user", text)
            self._begin_reply(text)
        elif self.state in ("carregando", "organizando") and self.pending is None:
            # O modelo ainda está se preparando: mostra a mensagem e deixa na fila.
            self.entry.delete("1.0", "end")
            self._autosize_entry()
            self._add("user", text)
            self.pending = text
            self._set_state(self.state)

    def _begin_reply(self, user_text: str) -> None:
        self.cancel_event.clear()
        self._set_state("respondendo")
        with self._edit():
            self._insert("Maddie", "maddie_name")
            self._insert(f"   {datetime.now():%H:%M}\n", "time")
            self.chat.mark_set("reply_start", "end-1c")
            self.chat.mark_gravity("reply_start", "left")
            self._insert("escrevendo…", "maddie", "muted")
        self._streaming, self._placeholder = True, True
        self.chat.see("end")
        threading.Thread(target=self._chat_worker, args=(user_text,), daemon=True).start()

    def _chat_worker(self, user_text: str) -> None:
        session = self.session
        try:
            if not self.client.is_loaded():
                self.events.put(("note", "O modelo tinha saído da memória por inatividade. "
                                         "Recarregando: esta resposta demora mais para começar."))
            for chunk in session.reply(user_text, cancelled=lambda: self.closing or self.cancel_event.is_set()):
                self.events.put(("chunk", chunk))
            while (session.last.reply.strip() and session.response_was_limited()
                   and session.maintenance_due()):
                self.events.put(("reply_paused", None))
                try:
                    rep = session.maintain(
                        cancelled=lambda: self.closing or self.cancel_event.is_set())
                    self.events.put(("reply_resumed", rep))
                    for chunk in session.continue_reply(
                            cancelled=lambda: self.closing or self.cancel_event.is_set()):
                        self.events.put(("chunk", chunk))
                except ResponseCancelled:
                    raise
                except Exception as exc:
                    self.events.put(("reply_resume_error", (session.last, str(exc))))
                    return
            self.events.put(("done", session.last))
        except ResponseCancelled:
            if not self.closing:
                self.events.put(("stopped", session.last))
        except OllamaError as exc:
            self.events.put(("chat_error", str(exc)))
        except Exception as exc:  # erro inesperado: mostrar em vez de travar a UI
            self.events.put(("chat_error", f"Erro inesperado: {exc!r}"))

    def _stream_chunk(self, chunk: str) -> None:
        if not self._streaming:
            return
        at_end = self.chat.yview()[1] >= 0.999
        with self._edit():
            if self._placeholder:
                self.chat.delete("reply_start", "end-1c")
                self._placeholder = False
            self._insert(chunk, "maddie")
        if at_end:  # não puxa a rolagem se o usuário subiu para reler
            self.chat.see("end")

    def _clear_stream_region(self) -> None:
        with self._edit():
            try:
                self.chat.delete("reply_start", "end-1c")
            except tk.TclError:
                pass
        self._streaming = self._placeholder = False

    def _finish_reply(self, info, stopped: bool) -> None:
        cfg = self.cfg
        text = info.reply
        at_end = self.chat.yview()[1] >= 0.98
        self._clear_stream_region()
        if not text.strip():
            # Nada foi escrito: tira o cabeçalho "Maddie" que ficou sozinho.
            with self._edit():
                self.chat.delete("reply_start -1 lines linestart", "end-1c")
            self._note("Resposta interrompida antes de começar." if stopped
                       else "A Maddie não escreveu nada. Tente de novo.")
        else:
            st = info.stats
            parts = []
            if info.ttft_s:
                parts.append(f"1º token {num(info.ttft_s)} s")
            if st.tokens_per_s:
                parts.append(f"{num(st.tokens_per_s)} tok/s")
            if st.output_tokens:
                parts.append(f"{st.output_tokens} tokens")
            parts.append(f"{num(info.total_s)} s")
            if stopped:
                parts.append("interrompida")
            entry = {"role": "maddie", "text": text, "time": datetime.now().strftime("%H:%M"),
                     "stats": " · ".join(parts)}
            self.transcript.append(entry)
            with self._edit():
                self._draw_reply_body(entry)
            self.stats_label.configure(text="\n".join(parts))
            if st.done_reason == "length":
                limit = "janela de contexto" if cfg.num_predict == -1 else f"limite de {cfg.num_predict} tokens"
                self._note(f"Resposta cortada no {limit} (ajuste em Ajustes).")
            if info.max_context_tokens >= cfg.num_ctx:
                self._note(f"Aviso: a janela de {cfg.num_ctx} tokens encheu; o começo da conversa pode "
                           "ter sido truncado pelo Ollama.")
        if at_end:
            self.chat.see("end")
        self._update_side()
        if self.session.maintenance_due():
            self._set_state("organizando")
            threading.Thread(target=self._maintain_worker, daemon=True).start()
        else:
            self._become_ready()

    def _abort_reply(self) -> None:
        if self._streaming:
            self._clear_stream_region()
            with self._edit():
                try:
                    self.chat.delete("reply_start -1 lines linestart", "end-1c")
                except tk.TclError:
                    pass

    def _stop(self) -> None:
        if self.state != "respondendo":
            return
        self.cancel_event.set()
        self.client.cancel()

    def _regenerate(self) -> None:
        if self.state != "disponivel" or not self.transcript or self.transcript[-1]["role"] != "maddie":
            return
        user_text = self.session.pop_last()
        if user_text is None:
            return
        self.transcript.pop()
        self._render_all()
        self._begin_reply(user_text)

    # ---------------- manutenção da memória ----------------
    def _maintain_worker(self) -> None:
        try:
            rep = self.session.maintain(cancelled=lambda: self.closing)
            self.events.put(("maintained", rep))
        except ResponseCancelled:
            pass
        except Exception as exc:
            self.events.put(("task_error", f"Falha ao organizar a memória: {exc!r}"))

    def _after_maintenance(self, rep, resume: bool = False) -> None:
        if rep.dropped:
            if rep.dropped == 1:
                what = "a troca mais antiga " + ("foi resumida" if rep.summarized else "saiu da memória")
            else:
                what = f"as {rep.dropped} trocas mais antigas " + (
                    "foram resumidas" if rep.summarized else "saíram da memória")
            self._note(f"Memória organizada: {what} ({num(rep.seconds, 0)} s).")
        self._announce_facts(rep.facts)
        if rep.error:
            self._note(f"Aviso: {rep.error}", "error")
        self._update_side()
        if resume:
            self._set_state("respondendo")
        else:
            self._become_ready()

    def _announce_facts(self, facts: list) -> None:
        for fact, action in facts:
            verb = "Anotei" if action == "novo" else "Atualizei"
            self._note(f"{verb} na memória permanente: {fact['fato']}")

    def extract_now(self) -> bool:
        """Chamado pelo painel de memória. False se o modelo está ocupado."""
        if self.state != "disponivel" or not self.cfg.base_memory_enabled or not len(self.session.memory):
            return False
        self._set_state("organizando")
        threading.Thread(target=self._extract_worker, daemon=True).start()
        return True

    def _extract_worker(self) -> None:
        try:
            accepted, rejected = self.session.extract_facts(cancelled=lambda: self.closing)
            self.events.put(("extracted", (accepted, rejected)))
        except ResponseCancelled:
            pass
        except Exception as exc:
            self.events.put(("task_error", f"Falha ao extrair fatos: {exc}"))

    def _after_extraction(self, accepted: list, rejected: list) -> None:
        self._announce_facts(accepted)
        if not accepted:
            extra = f" ({len(rejected)} sugestões do modelo foram recusadas pela validação.)" if rejected else ""
            self._note("Nenhum fato novo para guardar nesta conversa." + extra)
        self._update_side()
        dlg = self._dialogs.get("memory")
        if dlg is not None and dlg.winfo_exists():
            dlg.refresh()
        self._become_ready()

    # ---------------- ações ----------------
    def _new_conversation(self) -> None:
        if self.state not in ("disponivel", "indisponivel"):
            return
        self.session.clear()
        self.transcript.clear()
        self.pending = None
        self.stats_label.configure(text="—")
        self._render_all()
        self._update_side()
        self._update_actions()
        if self.state == "disponivel":
            # Toda conversa nova refaz a abertura (data/hora e fatos da memória permanente).
            # Dá para digitar enquanto isso: a mensagem fica na fila.
            self._set_state("carregando")
            self._start_prepare()

    def _set_profile(self, name: str) -> None:
        self.apply_config(config_mod.replace(self.cfg, **PROFILES[name]))

    def _bump_font(self, delta: int) -> None:
        size = max(8, min(24, self.cfg.ui_font_size + delta))
        if size != self.cfg.ui_font_size:
            self.apply_config(config_mod.replace(self.cfg, ui_font_size=size))

    def apply_config(self, new: Config, save: bool = True) -> None:
        """Aplica uma configuração nova; recarrega o modelo e refaz a janela só se for preciso."""
        old, self.cfg = self.cfg, new
        if save:
            try:
                config_mod.save(self.config_path, new)
            except (OSError, ConfigError) as exc:
                self._note(f"Não consegui gravar o config.json: {exc}", "error")
        self.session.set_config(new)
        need_prepare = old.load_signature() != new.load_signature()
        if old.base_memory_enabled != new.base_memory_enabled:
            if len(self.session.memory) == 0:
                need_prepare = self.session.refresh_system() or need_prepare
            else:
                self.session.store.load()
                self._note("Memória permanente " + ("ligada" if new.base_memory_enabled else "desligada")
                           + ". Vale a partir da próxima conversa (botão Nova conversa).")
        if (old.ui_theme, old.ui_font_size, old.ui_show_stats) != (new.ui_theme, new.ui_font_size,
                                                                    new.ui_show_stats):
            self.rebuild()
        self._update_side()
        if need_prepare and self.state in ("disponivel", "indisponivel"):
            if old.model != new.model:
                threading.Thread(target=self.client.unload_model_named, args=(old.model,), daemon=True).start()
            self._set_state("carregando")
            self._start_prepare()

    def _last_reply_text(self) -> str:
        return next((e["text"] for e in reversed(self.transcript) if e["role"] == "maddie"), "")

    def _copy(self, text: str) -> None:
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _copy_last(self) -> None:
        self._copy(self._last_reply_text())

    def _copy_selection(self) -> None:
        try:
            self._copy(self.chat.get("sel.first", "sel.last"))
        except tk.TclError:
            pass

    def _export(self) -> None:
        msgs = [e for e in self.transcript if e["role"] in ("user", "maddie")]
        if not msgs:
            messagebox.showinfo("Exportar conversa", "Ainda não há conversa para exportar.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Exportar conversa", defaultextension=".md",
            initialfile=f"conversa_maddie_{datetime.now():%Y%m%d_%H%M}.md",
            filetypes=[("Markdown", "*.md"), ("Texto", "*.txt")])
        if not path:
            return
        lines = [f"# Conversa com a Maddie — {datetime.now():%d/%m/%Y %H:%M}", "",
                 f"Modelo: `{self.cfg.model}`", ""]
        for e in msgs:
            who = "Você" if e["role"] == "user" else "Maddie"
            lines += [f"**{who}** ({e['time']})", "", e["text"].rstrip(), ""]
        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Exportar conversa", f"Não consegui gravar o arquivo:\n{exc}", parent=self.root)
            return
        self._note(f"Conversa exportada para {path}")

    def _open_settings(self) -> None:
        from .dialogs import SettingsDialog
        self._open_dialog("settings", SettingsDialog)

    def _open_memory(self) -> None:
        from .dialogs import MemoryDialog
        self._open_dialog("memory", MemoryDialog)

    def _open_dialog(self, key: str, cls) -> None:
        dlg = self._dialogs.get(key)
        if dlg is not None and dlg.winfo_exists():
            dlg.lift()
            dlg.focus_set()
            return
        self._dialogs[key] = cls(self)

    def _on_close(self) -> None:
        self.closing = True
        self.cancel_event.set()
        self.client.cancel()
        try:  # guarda o tamanho da janela, se mudou
            w = int(round(self.root.winfo_width() / self.scale))
            h = int(round(self.root.winfo_height() / self.scale))
            if self.root.state() == "normal" and (abs(w - self.cfg.ui_width) > 8 or abs(h - self.cfg.ui_height) > 8):
                config_mod.save(self.config_path, config_mod.replace(self.cfg, ui_width=max(640, w),
                                                                     ui_height=max(480, h)))
        except (tk.TclError, OSError, ConfigError):
            pass
        if self.cfg.unload_model_on_exit:
            self.client.unload_model(timeout=3.0)
        self.session.memory.clear()
        self.root.destroy()


def run(cfg: Config, persona: str, base_dir: str | Path) -> None:
    enable_dpi_awareness()
    root = tk.Tk()
    MaddieApp(root, cfg, persona, Path(base_dir))
    root.mainloop()
