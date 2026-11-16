"""Markdown mínimo para a área de conversa (sem dependências).

`parse(text)` devolve uma lista de (trecho, tags), pronta para inserir num
tk.Text. Cobre o que os modelos costumam escrever: **negrito**, *itálico*,
`código`, blocos ```, títulos #, listas e citações. O que não for reconhecido
sai como texto comum; nunca levanta erro.
"""

from __future__ import annotations

import re

_INLINE = re.compile(
    r"(`[^`\n]+`)"                       # `código`
    r"|(\*\*[^*\n]+?\*\*|__[^_\n]+?__)"  # **negrito**
    r"|((?<![\w*])\*[^*\s][^*\n]*?\*(?![\w*])|(?<![\w_])_[^_\s][^_\n]*?_(?![\w_]))"  # *itálico*
)
_HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*•]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)(\d{1,3})[.)]\s+(.*)$")
_RULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")


def _inline(text: str, base: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], base))
        token = m.group(0)
        if m.group(1):
            out.append((token[1:-1], base + ("code",)))
        elif m.group(2):
            out.extend(_inline(token[2:-2], base + ("bold",)))
        else:
            out.extend(_inline(token[1:-1], base + ("italic",)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], base))
    return out


def parse(text: str, base: str = "maddie") -> list[tuple[str, tuple[str, ...]]]:
    segs: list[tuple[str, tuple[str, ...]]] = []
    in_code = False
    lines = text.split("\n")
    for i, line in enumerate(lines):
        nl = "\n" if i < len(lines) - 1 else ""
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            segs.append((line + "\n", (base, "codeblock")))
            continue
        if _RULE.match(line):
            segs.append(("─" * 24 + nl, (base, "muted")))
            continue
        m = _HEADING.match(line)
        if m:
            segs.extend(_inline(m.group(2), (base, "heading")))
            segs.append((nl, (base,)))
            continue
        m = _BULLET.match(line)
        if m:
            level = "list2" if len(m.group(1)) >= 2 else "list"
            segs.append(("•  ", (base, level)))
            segs.extend(_inline(m.group(2), (base, level)))
            segs.append((nl, (base, level)))
            continue
        m = _NUMBERED.match(line)
        if m:
            level = "list2" if len(m.group(1)) >= 2 else "list"
            segs.append((f"{m.group(2)}.  ", (base, level)))
            segs.extend(_inline(m.group(3), (base, level)))
            segs.append((nl, (base, level)))
            continue
        if line.lstrip().startswith(">"):
            segs.extend(_inline(line.lstrip()[1:].lstrip(), (base, "quote")))
            segs.append((nl, (base, "quote")))
            continue
        segs.extend(_inline(line, (base,)))
        if nl:
            segs.append((nl, (base,)))
    return [(t, tags) for t, tags in segs if t]


def plain(text: str) -> str:
    """Texto sem marcação (para testes e para copiar)."""
    return "".join(t for t, _ in parse(text))
