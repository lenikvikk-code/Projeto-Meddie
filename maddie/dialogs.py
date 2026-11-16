"""Janelas secundárias: Ajustes e Memória permanente."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from . import config as config_mod
from .config import PROFILES, Config, ConfigError
from .memory_store import CATEGORIES, CATEGORY_LABELS, MAX_FACT_CHARS, is_sensitive
from .ollama_client import OllamaError

KEEP_ALIVE = [("10 minutos", "10m"), ("30 minutos", "30m"), ("1 hora", "1h"),
              ("Sempre, até fechar a Maddie", "-1m")]
AUTO = "automático"
DEFAULT = "padrão do modelo"


def history_budget_for(num_ctx: int, num_predict: int) -> int:
    """Teto do histórico proporcional à janela (8192/768 -> 5500)."""
    return max(500, int(round((num_ctx - num_predict) * 0.74 / 100.0)) * 100)


class _Dialog(tk.Toplevel):
    def __init__(self, app, title: str, width: int, height: int):
        super().__init__(app.root)
        self.app = app
        self.c = app.c
        self.title(title)
        self.configure(bg=self.c["bg"])
        self.transient(app.root)
        w, h = app.px(width), app.px(height)
        x = app.root.winfo_rootx() + max(0, (app.root.winfo_width() - w) // 2)
        y = app.root.winfo_rooty() + max(0, (app.root.winfo_height() - h) // 3)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(app.px(460), app.px(360))
        self.bind("<Escape>", lambda e: self.destroy())
        from .gui import set_titlebar
        set_titlebar(self, app.cfg.ui_theme == "escuro")

    def label(self, parent, text, muted=False, bold=False, **kw) -> tk.Label:
        return tk.Label(parent, text=text, bg=parent.cget("bg"),
                        fg=self.c["muted"] if muted else self.c["fg"],
                        font=self.app.font(-2 if muted else -1, "bold" if bold else "normal"),
                        anchor="w", justify="left", **kw)


class SettingsDialog(_Dialog):
    def __init__(self, app):
        super().__init__(app, "Ajustes da Maddie", 620, 690)
        cfg: Config = app.cfg
        px = app.px
        self.v: dict[str, tk.Variable] = {}

        buttons = tk.Frame(self, bg=self.c["bg"])
        buttons.pack(side="bottom", fill="x", padx=px(18), pady=px(14))
        app.button(buttons, "Salvar", self._save, "primary").pack(side="right")
        app.button(buttons, "Cancelar", self.destroy, "flat").pack(side="right", padx=(0, px(8)))
        app.button(buttons, "Restaurar padrões", self._defaults, "ghost").pack(side="left")

        # Área rolável
        canvas = tk.Canvas(self, bg=self.c["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(canvas, bg=self.c["bg"])
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.body = body
        self._row = 0
        body.grid_columnconfigure(1, weight=1)

        # ----- Modelo
        self._head("Modelo")
        try:
            models = [m for m in app.client.installed_models() if "cloud" not in m]
        except OllamaError:
            models = []
        if cfg.model not in models:
            models.insert(0, cfg.model)
        self._combo("model", "Modelo instalado", models, cfg.model,
                    "Trocar de modelo recarrega e reaquece (cerca de 30 a 40 s).")
        self._combo("num_ctx", "Janela de contexto (tokens)", ["4096", "8192", "12288", "16384"],
                    str(cfg.num_ctx), "Maior = lembra mais, usa mais RAM e recarrega o modelo.")
        self._combo("num_predict", "Tamanho máximo da resposta (tokens)",
                    ["256", "512", "768", "1024", "1536", "2048"], str(cfg.num_predict))
        ka = dict((v, k) for k, v in KEEP_ALIVE)
        self._combo("keep_alive", "Manter o modelo na RAM por", [k for k, _ in KEEP_ALIVE],
                    ka.get(cfg.keep_alive, cfg.keep_alive),
                    "Depois desse tempo sem uso o modelo sai da RAM e a resposta seguinte demora a começar.")
        self._combo("num_thread", "Threads de CPU", [AUTO, "4", "6", "8", "12"],
                    AUTO if cfg.num_thread is None else str(cfg.num_thread),
                    "Automático usa os núcleos físicos. Mudar recarrega o modelo.")

        # ----- Estilo
        self._head("Estilo da resposta")
        self._combo("profile", "Perfil", list(PROFILES) + ["Personalizado"], cfg.profile_name())
        self.v["profile"].trace_add("write", lambda *a: self._profile_changed())
        self._scale("temperature", "Temperatura (criatividade)", 0.0, 1.5, cfg.temperature)
        self._scale("top_p", "Top-p (variedade de palavras)", 0.1, 1.0, cfg.top_p)
        self._combo("top_k", "Top-k", ["10", "20", "40", "64", "100"], str(cfg.top_k))
        self._combo("repeat_penalty", "Penalidade de repetição",
                    [DEFAULT, "1.0", "1.05", "1.1", "1.15", "1.2"],
                    DEFAULT if cfg.repeat_penalty is None else str(cfg.repeat_penalty),
                    "Aumente um pouco se a Maddie repetir frases.")

        # ----- Comportamento
        self._head("Comportamento")
        self._check("warm_up", "Preparar a Maddie ao abrir (primeira resposta rápida)", cfg.warm_up)
        self._check("strip_emoji", "Remover emojis das respostas (a menos que eu use primeiro)",
                    cfg.strip_emoji)
        self._check("memory_summarize", "Resumir as mensagens antigas antes de esquecê-las",
                    cfg.memory_summarize)
        self._check("base_memory_enabled",
                    "Memória permanente: guardar fatos sobre mim entre conversas (grava em disco)",
                    cfg.base_memory_enabled)
        self._check("unload_model_on_exit", "Liberar a RAM ao fechar a Maddie", cfg.unload_model_on_exit)

        # ----- Aparência
        self._head("Aparência")
        self._combo("ui_theme", "Tema", list(config_mod.THEMES), cfg.ui_theme)
        self._combo("ui_font_size", "Tamanho da letra", [str(n) for n in range(9, 19)],
                    str(cfg.ui_font_size))
        self._check("ui_show_stats", "Mostrar tempo e velocidade abaixo de cada resposta",
                    cfg.ui_show_stats)
        tk.Frame(body, bg=self.c["bg"], height=px(10)).grid(row=self._row, column=0)
        self.grab_set()

    # ---------- montagem ----------
    def _head(self, text: str) -> None:
        px = self.app.px
        tk.Label(self.body, text=text.upper(), bg=self.c["bg"], fg=self.c["accent"],
                 font=self.app.font(-2, "bold"), anchor="w").grid(
            row=self._row, column=0, columnspan=2, sticky="ew", padx=px(18), pady=(px(16), px(4)))
        self._row += 1

    def _hint(self, text: str) -> None:
        if text:
            self.label(self.body, text, muted=True, wraplength=self.app.px(520)).grid(
                row=self._row, column=0, columnspan=2, sticky="ew", padx=self.app.px(18))
            self._row += 1

    def _combo(self, key, text, values, value, hint="") -> None:
        px = self.app.px
        self.label(self.body, text).grid(row=self._row, column=0, sticky="w", padx=(px(18), px(12)),
                                         pady=px(4))
        var = tk.StringVar(value=value)
        ttk.Combobox(self.body, textvariable=var, values=values, state="readonly", width=30).grid(
            row=self._row, column=1, sticky="ew", padx=(0, px(18)), pady=px(4))
        self.v[key] = var
        self._row += 1
        self._hint(hint)

    def _scale(self, key, text, lo, hi, value) -> None:
        px = self.app.px
        self.label(self.body, text).grid(row=self._row, column=0, sticky="w", padx=(px(18), px(12)),
                                         pady=px(4))
        frame = tk.Frame(self.body, bg=self.c["bg"])
        frame.grid(row=self._row, column=1, sticky="ew", padx=(0, px(18)), pady=px(4))
        var = tk.DoubleVar(value=value)
        shown = tk.Label(frame, text=f"{value:.2f}", bg=self.c["bg"], fg=self.c["fg"],
                         font=self.app.font(-1), width=5)
        shown.pack(side="right")

        def moved(_=None):
            var.set(round(var.get() * 20) / 20)  # passos de 0,05
            shown.configure(text=f"{var.get():.2f}")
            if not getattr(self, "_setting_profile", False):
                self._sync_profile()

        ttk.Scale(frame, from_=lo, to=hi, variable=var, command=moved).pack(side="left", fill="x",
                                                                           expand=True)
        var._shown = shown
        self.v[key] = var
        self._row += 1

    def _check(self, key, text, value) -> None:
        var = tk.BooleanVar(value=value)
        ttk.Checkbutton(self.body, text=text, variable=var).grid(
            row=self._row, column=0, columnspan=2, sticky="w", padx=self.app.px(18), pady=self.app.px(3))
        self.v[key] = var
        self._row += 1

    # ---------- perfil <-> controles ----------
    def _profile_changed(self) -> None:
        p = PROFILES.get(self.v["profile"].get())
        if not p:
            return
        self._setting_profile = True
        for key in ("temperature", "top_p"):
            self.v[key].set(p[key])
            self.v[key]._shown.configure(text=f"{p[key]:.2f}")
        self._setting_profile = False

    def _sync_profile(self) -> None:
        t, p = self.v["temperature"].get(), self.v["top_p"].get()
        name = next((n for n, x in PROFILES.items()
                     if abs(x["temperature"] - t) < 1e-6 and abs(x["top_p"] - p) < 1e-6), "Personalizado")
        if self.v["profile"].get() != name:
            self._setting_profile = True
            self.v["profile"].set(name)
            self._setting_profile = False

    # ---------- leitura ----------
    def _collect(self) -> Config:
        v, old = self.v, self.app.cfg
        num_ctx, num_predict = int(v["num_ctx"].get()), int(v["num_predict"].get())
        mem = old.memory_max_tokens
        if (num_ctx, num_predict) != (old.num_ctx, old.num_predict):
            mem = history_budget_for(num_ctx, num_predict)
        ka = dict(KEEP_ALIVE)
        thread = v["num_thread"].get()
        rp = v["repeat_penalty"].get()
        return config_mod.replace(
            old,
            model=v["model"].get(), num_ctx=num_ctx, num_predict=num_predict, memory_max_tokens=mem,
            keep_alive=ka.get(v["keep_alive"].get(), v["keep_alive"].get()),
            num_thread=None if thread == AUTO else int(thread),
            temperature=round(float(v["temperature"].get()), 2), top_p=round(float(v["top_p"].get()), 2),
            top_k=int(v["top_k"].get()), repeat_penalty=None if rp == DEFAULT else float(rp),
            warm_up=bool(v["warm_up"].get()), strip_emoji=bool(v["strip_emoji"].get()),
            memory_summarize=bool(v["memory_summarize"].get()),
            base_memory_enabled=bool(v["base_memory_enabled"].get()),
            unload_model_on_exit=bool(v["unload_model_on_exit"].get()),
            ui_theme=v["ui_theme"].get(), ui_font_size=int(v["ui_font_size"].get()),
            ui_show_stats=bool(v["ui_show_stats"].get()),
        )

    def _save(self) -> None:
        try:
            new = self._collect()
        except (ConfigError, ValueError) as exc:
            messagebox.showerror("Ajustes", f"Configuração inválida:\n{exc}", parent=self)
            return
        old = self.app.cfg
        if new.base_memory_enabled and not old.base_memory_enabled:
            ok = messagebox.askyesno(
                "Memória permanente",
                "Com esta opção ligada, a Maddie passa a gravar em disco fatos que você contar sobre "
                f"você (arquivo {new.base_memory_file}, em texto puro).\n\n"
                "Dados de saúde, finanças, documentos, senhas, endereço e telefone são bloqueados. "
                "Você pode ver, editar e apagar tudo no painel Memória permanente.\n\nLigar?",
                parent=self)
            if not ok:
                return
        self.destroy()
        if new != old:
            self.app.apply_config(new)

    def _defaults(self) -> None:
        d = Config()
        if not messagebox.askyesno("Ajustes", "Voltar todos os ajustes ao padrão da Maddie?\n"
                                              "(Só vale depois de Salvar.)", parent=self):
            return
        ka = dict((v, k) for k, v in KEEP_ALIVE)
        values = {
            "model": d.model, "num_ctx": str(d.num_ctx), "num_predict": str(d.num_predict),
            "keep_alive": ka[d.keep_alive], "num_thread": AUTO, "profile": d.profile_name(),
            "top_k": str(d.top_k), "repeat_penalty": DEFAULT, "warm_up": d.warm_up,
            "strip_emoji": d.strip_emoji, "memory_summarize": d.memory_summarize,
            "base_memory_enabled": d.base_memory_enabled, "unload_model_on_exit": d.unload_model_on_exit,
            "ui_theme": d.ui_theme, "ui_font_size": str(d.ui_font_size), "ui_show_stats": d.ui_show_stats,
        }
        for key, value in values.items():
            self.v[key].set(value)
        self._profile_changed()


class MemoryDialog(_Dialog):
    def __init__(self, app):
        super().__init__(app, "Memória permanente da Maddie", 760, 560)
        px = app.px
        self.store = app.session.store

        top = tk.Frame(self, bg=self.c["bg"])
        top.pack(fill="x", padx=px(18), pady=(px(16), px(8)))
        self.state_label = self.label(top, "", bold=True)
        self.state_label.pack(fill="x")
        self.label(top, "Fatos duráveis sobre você que a Maddie lê no começo de cada conversa. Ficam num "
                        "arquivo de texto nesta máquina. Mudanças valem a partir da próxima conversa.",
                   muted=True, wraplength=px(700)).pack(fill="x", pady=(px(4), 0))
        self.path_label = self.label(top, "", muted=True, wraplength=px(700))
        self.path_label.pack(fill="x")

        btns = tk.Frame(self, bg=self.c["bg"])
        btns.pack(side="bottom", fill="x", padx=px(18), pady=px(14))
        self.toggle_btn = app.button(btns, "", self._toggle, "primary")
        self.toggle_btn.pack(side="left")
        self.extract_btn = app.button(btns, "Guardar fatos desta conversa", self._extract, "flat")
        self.extract_btn.pack(side="left", padx=(px(8), 0))
        app.button(btns, "Fechar", self.destroy, "flat").pack(side="right")

        edit = tk.Frame(self, bg=self.c["bg"])
        edit.pack(side="bottom", fill="x", padx=px(18))
        self.edit_btns = [
            app.button(edit, "Adicionar", self._add, "flat"),
            app.button(edit, "Editar", self._edit_selected, "flat"),
            app.button(edit, "Apagar", self._remove, "flat"),
            app.button(edit, "Apagar tudo", self._clear, "ghost"),
        ]
        for b in self.edit_btns:
            b.pack(side="left", padx=(0, px(8)))

        frame = tk.Frame(self, bg=self.c["bg"])
        frame.pack(fill="both", expand=True, padx=px(18), pady=px(8))
        scroll = ttk.Scrollbar(frame, orient="vertical")
        self.tree = ttk.Treeview(frame, columns=("cat", "fato", "quando"), show="headings",
                                 yscrollcommand=scroll.set, selectmode="browse")
        scroll.config(command=self.tree.yview)
        self.tree.heading("cat", text="Categoria")
        self.tree.heading("fato", text="Fato")
        self.tree.heading("quando", text="Atualizado")
        self.tree.column("cat", width=px(130), stretch=False)
        self.tree.column("fato", width=px(440))
        self.tree.column("quando", width=px(120), stretch=False)
        scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self._edit_selected())
        self.refresh()

    # ---------- estado ----------
    def refresh(self) -> None:
        cfg = self.app.cfg
        on = cfg.base_memory_enabled
        self.store.load()
        if self.store.load_error:
            self.state_label.configure(text=self.store.load_error, fg=self.c["error"])
        else:
            self.state_label.configure(
                text=(f"Ligada · {len(self.store.facts)} de {self.store.max_facts} fatos" if on else
                      "Desligada · nada da conversa é gravado em disco"),
                fg=self.c["ok"] if on else self.c["muted"])
        self.path_label.configure(text=f"Arquivo: {self.store.path}")
        self.toggle_btn.configure(text="Desligar memória permanente" if on else "Ligar memória permanente")
        self.tree.delete(*self.tree.get_children())
        for f in self.store.facts:
            self.tree.insert("", "end", iid=f["id"], values=(
                CATEGORY_LABELS.get(f.get("categoria"), f.get("categoria", "")), f["fato"],
                str(f.get("atualizado_em", "")).replace("T", " ")))
        usable = on and not self.store.load_error
        for b in self.edit_btns:
            b.configure(state="normal" if usable else "disabled")
        self.extract_btn.configure(state="normal" if usable else "disabled")
        self.app._update_side()

    def _toggle(self) -> None:
        cfg = self.app.cfg
        if not cfg.base_memory_enabled:
            ok = messagebox.askyesno(
                "Memória permanente",
                "A Maddie vai passar a gravar em disco fatos que você contar sobre você (texto puro, "
                "nesta máquina). Dados de saúde, finanças, documentos, senhas, endereço e telefone são "
                "bloqueados.\n\nLigar?", parent=self)
            if not ok:
                return
        self.app.apply_config(config_mod.replace(cfg, base_memory_enabled=not cfg.base_memory_enabled))
        self.refresh()

    def _save_store(self) -> bool:
        try:
            self.store.save()
            return True
        except OSError as exc:
            messagebox.showerror("Memória permanente", f"Não consegui gravar o arquivo:\n{exc}", parent=self)
            return False

    # ---------- edição ----------
    def _selected(self) -> str | None:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _add(self) -> None:
        result = FactForm(self, "Adicionar fato", "identidade", "O usuário ").result
        if result:
            cat, text = result
            _, action = self.store.add(cat, text, origem="manual")
            if action == "cheio":
                messagebox.showinfo("Memória permanente", "A memória está cheia. Apague algum fato antes.",
                                    parent=self)
                return
            if self._save_store():
                self.refresh()

    def _edit_selected(self) -> None:
        fid = self._selected()
        fact = self.store.get(fid) if fid else None
        if not fact:
            return
        result = FactForm(self, "Editar fato", fact.get("categoria", "identidade"), fact["fato"]).result
        if result:
            cat, text = result
            fact["categoria"] = cat
            self.store.update_text(fid, text)
            if self._save_store():
                self.refresh()

    def _remove(self) -> None:
        fid = self._selected()
        if fid and self.store.remove(fid) and self._save_store():
            self.refresh()

    def _clear(self) -> None:
        if not self.store.facts:
            return
        if messagebox.askyesno("Memória permanente", "Apagar TODOS os fatos guardados? Não dá para desfazer.",
                               parent=self):
            self.store.clear()
            if self._save_store():
                self.refresh()

    def _extract(self) -> None:
        if not len(self.app.session.memory):
            messagebox.showinfo("Memória permanente", "Ainda não há conversa para analisar.", parent=self)
            return
        if not self.app.extract_now():
            messagebox.showinfo("Memória permanente", "A Maddie está ocupada. Tente quando ela estiver "
                                                      "disponível.", parent=self)
            return
        self.state_label.configure(text="Analisando a conversa… (pode levar de 20 a 60 s)",
                                   fg=self.c["warn"])


class FactForm(tk.Toplevel):
    """Formulário modal de um fato. `result` = (categoria, texto) ou None."""

    def __init__(self, parent: _Dialog, title: str, categoria: str, text: str):
        super().__init__(parent)
        app = parent.app
        c, px = app.c, app.px
        self.result = None
        self.title(title)
        self.configure(bg=c["bg"])
        self.transient(parent)
        self.resizable(False, False)
        labels = [CATEGORY_LABELS[k] for k in CATEGORIES]
        self.cat = tk.StringVar(value=CATEGORY_LABELS.get(categoria, labels[0]))
        tk.Label(self, text="Categoria", bg=c["bg"], fg=c["fg"], font=app.font(-1)).grid(
            row=0, column=0, sticky="w", padx=px(16), pady=(px(16), px(4)))
        ttk.Combobox(self, textvariable=self.cat, values=labels, state="readonly", width=28).grid(
            row=0, column=1, sticky="w", padx=(0, px(16)), pady=(px(16), px(4)))
        tk.Label(self, text="Fato", bg=c["bg"], fg=c["fg"], font=app.font(-1)).grid(
            row=1, column=0, sticky="nw", padx=px(16), pady=px(4))
        self.text = tk.Text(self, width=52, height=4, wrap="word", bg=c["entry_bg"], fg=c["fg"],
                            insertbackground=c["accent"], font=app.font(-1), relief="flat",
                            padx=px(8), pady=px(6))
        self.text.grid(row=1, column=1, padx=(0, px(16)), pady=px(4))
        self.text.insert("1.0", text)
        tk.Label(self, text=f"Até {MAX_FACT_CHARS} caracteres. Exemplo: O usuário se chama Lenik.",
                 bg=c["bg"], fg=c["muted"], font=app.font(-2)).grid(row=2, column=1, sticky="w")
        row = tk.Frame(self, bg=c["bg"])
        row.grid(row=3, column=0, columnspan=2, sticky="e", padx=px(16), pady=px(14))
        app.button(row, "Cancelar", self.destroy, "flat").pack(side="right")
        app.button(row, "Guardar", self._ok, "primary").pack(side="right", padx=(0, px(8)))
        self.bind("<Escape>", lambda e: self.destroy())
        self.text.focus_set()
        self.grab_set()
        self.wait_window(self)

    def _ok(self) -> None:
        text = " ".join(self.text.get("1.0", "end").split())
        if not text:
            return
        if len(text) > MAX_FACT_CHARS:
            messagebox.showinfo("Fato", f"O fato tem {len(text)} caracteres; o limite é {MAX_FACT_CHARS}.",
                                parent=self)
            return
        if is_sensitive(text) and not messagebox.askyesno(
                "Fato", "Este texto parece conter dado sensível (saúde, finanças, documento, contato).\n"
                        "Guardar mesmo assim?", parent=self):
            return
        cat = next(k for k in CATEGORIES if CATEGORY_LABELS[k] == self.cat.get())
        self.result = (cat, text)
        self.destroy()
