"""Filtros de texto aplicados à resposta durante o streaming.

EmojiFilter: a persona pede "sem emojis, a menos que o usuário use primeiro",
mas o modelo de 4B ainda os emitia em 18% das respostas (medido na v0.2).
Em vez de insistir no prompt, a regra passou a ser garantida por código.
"""

from __future__ import annotations

import re

_EMOJI_CLASS = (
    "\U0001F000-\U0001FAFF"   # pictogramas, emoticons, transportes, símbolos suplementares
    "\U0001FC00-\U0001FFFF"
    "☀-➿"           # símbolos diversos e dingbats (☀ ✅ ✨ ❤)
    "⬀-⯿"           # ⭐ ⬆
    "⌚-⌛⏩-⏺"
    "〰〽㊗㊙"  # setas simples (→) ficam: aparecem em contas e explicações
)
_JOINERS = "️︎‍⃣\U000E0020-\U000E007F"

EMOJI_RE = re.compile(f"[{_EMOJI_CLASS}]")
_STRIP_RE = re.compile(f"[{_EMOJI_CLASS}{_JOINERS}]")
_PUNCT = ".,!?;:)"


def has_emoji(text: str) -> bool:
    return bool(EMOJI_RE.search(text))


class EmojiFilter:
    """Remove emojis de um texto recebido em pedaços, sem deixar espaços sobrando.

    "Boa ideia 😊. Vamos" -> "Boa ideia. Vamos"
    "Tudo bem! 😄"        -> "Tudo bem!"
    """

    def __init__(self) -> None:
        self._pending_ws = ""     # espaços vistos, ainda não emitidos
        self._dropped = False     # um emoji foi removido depois do último texto emitido
        self._started = False     # já emitiu algum texto
        self.removed = 0

    def feed(self, chunk: str) -> str:
        out: list[str] = []
        for ch in chunk:
            if _STRIP_RE.match(ch):
                if EMOJI_RE.match(ch):
                    self.removed += 1
                self._dropped = True
                continue
            if ch in " \t":
                self._pending_ws += ch
                continue
            if ch == "\n":
                self._pending_ws = ""   # espaço antes de quebra de linha não interessa
                if self._started:
                    out.append(ch)
                self._dropped = False
                continue
            # caractere visível
            if self._pending_ws:
                at_line_start = not self._started or (out[-1:] == ["\n"])
                if self._dropped and (ch in _PUNCT or at_line_start):
                    pass  # "ideia 😊." -> "ideia."; "😊 Olá" -> "Olá"
                elif self._dropped:
                    out.append(" ")  # "a 😊 b" -> "a b"
                else:
                    out.append(self._pending_ws)
                self._pending_ws = ""
            self._dropped = False
            self._started = True
            out.append(ch)
        return "".join(out)

    def flush(self) -> str:
        """Fim do texto: espaços finais pendentes são descartados."""
        self._pending_ws = ""
        return ""


def strip_emoji(text: str) -> str:
    f = EmojiFilter()
    return (f.feed(text) + f.flush()).rstrip()
