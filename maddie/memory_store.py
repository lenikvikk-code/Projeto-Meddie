"""Memória base (entre sessões): fatos duráveis sobre o usuário, gravados em disco.

DESLIGADA POR PADRÃO (`base_memory.enabled` no config.json). É o único ponto
do app que grava algo da conversa em disco; por isso só funciona com a chave
ligada pelo usuário, o arquivo é texto puro e auditável, e nada entra sem
passar por `validate_candidate`.

Baseado no estudo `estudos/memoria_base_maddie_2.0.md`:
- cada fato guarda o trecho literal do usuário de onde saiu (`trecho_fonte`);
- o trecho precisa existir numa mensagem do USUÁRIO (o modelo não inventa fonte);
- lista de bloqueio para dados sensíveis; teto de tamanho e de quantidade;
- sem duplicatas: fato parecido atualiza o existente.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

CATEGORIES = ("identidade", "pessoas", "preferencias", "contexto")
CATEGORY_LABELS = {
    "identidade": "Identidade", "pessoas": "Pessoas e animais",
    "preferencias": "Preferências", "contexto": "Contexto atual",
}
MAX_FACT_CHARS = 200
MIN_SOURCE_CHARS = 6
SIMILARITY_DUP = 0.82


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


# Dados que nunca entram no arquivo, mesmo que o usuário os diga na conversa.
_SENSITIVE = [
    r"\bsenhas?\b", r"\bpassword\b", r"\bpin\b", r"\btoken\b", r"\bchave (pix|de acesso|secreta|privada)\b",
    r"\bcpf\b", r"\bcnpj\b", r"\brg\b", r"\bcnh\b", r"\bpassaporte\b", r"\btitulo de eleitor\b",
    r"\bcartao (de credito|de debito)\b", r"\bcvv\b", r"\bconta (bancaria|corrente|poupanca)\b", r"\bagencia\b",
    r"\bsalario\b", r"\bdivida", r"\bemprestimo", r"\bfaturamento\b", r"\brenda\b",
    r"\bdiagnostic", r"\bdoenca", r"\bremedio", r"\bmedicament", r"\btratamento\b", r"\bdepress", r"\bansiedade\b",
    r"\bcancer\b", r"\bhiv\b", r"\bterapia\b", r"\bpsiquiatr", r"\bcirurgia\b", r"\bgravid",
    r"\breligi", r"\bpartido\b", r"\borientacao sexual\b",
    r"\bendereco\b", r"\bcep\b",
    r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}",          # CPF
    r"\b(?:\d[ -]?){13,19}\b",                # número de cartão
    r"\b\d{5}-\d{3}\b",                       # CEP
    r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b",      # telefone
    r"[\w.+-]+@[\w-]+\.[\w.]+",               # e-mail
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE))


def is_sensitive(text: str) -> bool:
    return bool(_SENSITIVE_RE.search(_norm(text)))


def validate_candidate(cand: dict, user_texts: list[str]) -> tuple[bool, str]:
    """Confere um fato proposto pelo modelo ANTES de gravar. Devolve (ok, motivo)."""
    if not isinstance(cand, dict):
        return False, "formato inválido"
    categoria = str(cand.get("categoria", "")).strip().lower()
    fato = str(cand.get("fato", "")).strip()
    fonte = str(cand.get("trecho_fonte", "")).strip()
    if categoria not in CATEGORIES:
        return False, f"categoria não permitida: {categoria!r}"
    if not fato or len(fato) > MAX_FACT_CHARS:
        return False, "fato vazio ou longo demais"
    if len(_norm(fonte)) < MIN_SOURCE_CHARS:
        return False, "trecho de origem ausente ou curto demais"
    if "aplicacao:" in _norm(fonte) or "aplicacao:" in _norm(fato):
        return False, "copiou a nota da aplicação"
    nf = _norm(fonte)
    if not any(nf in _norm(t) for t in user_texts):
        return False, "o trecho de origem não está em nenhuma mensagem do usuário"
    if is_sensitive(fato) or is_sensitive(fonte):
        return False, "dado sensível (lista de bloqueio)"
    if re.search(r"\b(maddie|meddie|assistente)\b", _norm(fato)):
        # Medido em 14/11/2026: "O usuário gosta de conversar com Maddie" passava. Não é um fato
        # útil para outra conversa; o que for sobre a Maddie o usuário pode incluir à mão no painel.
        return False, "fato sobre a Maddie, não sobre o usuário"
    if fonte.rstrip().endswith("?"):
        return False, "o trecho de origem é uma pergunta, não uma afirmação"
    return True, ""


class FactStore:
    def __init__(self, path: str | Path, max_facts: int = 60):
        self.path = Path(path)
        self.max_facts = max_facts
        self.facts: list[dict] = []
        self._next = 1
        self.load_error = ""

    # ---------- disco ----------
    def load(self) -> None:
        self.facts, self._next, self.load_error = [], 1, ""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            facts = data.get("fatos", [])
            self.facts = [f for f in facts if isinstance(f, dict) and f.get("fato") and f.get("id")]
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            # Arquivo ilegível: não sobrescreve (o usuário pode querer recuperá-lo).
            self.load_error = f"Memória base ilegível ({exc}). Nada foi carregado nem alterado."
            return
        nums = [int(m.group(1)) for f in self.facts if (m := re.match(r"f-(\d+)$", str(f["id"])))]
        self._next = max(nums, default=0) + 1

    def save(self) -> None:
        if self.load_error:
            raise OSError(self.load_error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"versao": 1, "fatos": self.facts}, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.path)

    # ---------- alteração ----------
    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="minutes")

    def find_similar(self, fato: str, categoria: str | None = None) -> dict | None:
        n = _norm(fato)
        best, best_ratio = None, 0.0
        for f in self.facts:
            if categoria and f.get("categoria") != categoria:
                continue
            ratio = difflib.SequenceMatcher(None, n, _norm(f["fato"])).ratio()
            if ratio > best_ratio:
                best, best_ratio = f, ratio
        return best if best_ratio >= SIMILARITY_DUP else None

    def get(self, fact_id: str) -> dict | None:
        return next((f for f in self.facts if f["id"] == fact_id), None)

    def add(self, categoria: str, fato: str, trecho_fonte: str = "", substitui_id: str | None = None,
            origem: str = "extraido") -> tuple[dict | None, str]:
        """Inclui ou atualiza um fato. Devolve (fato, ação): 'novo', 'atualizado', 'igual' ou 'cheio'."""
        fato = fato.strip()[:MAX_FACT_CHARS]
        target = self.get(substitui_id) if substitui_id else None
        if target is not None and target.get("categoria") != categoria:
            target = None  # só substitui dentro da mesma categoria
        target = target or self.find_similar(fato, categoria)
        if target is not None:
            if _norm(target["fato"]) == _norm(fato):
                return target, "igual"
            target.update(fato=fato, trecho_fonte=trecho_fonte or target.get("trecho_fonte", ""),
                          atualizado_em=self._now())
            return target, "atualizado"
        if len(self.facts) >= self.max_facts:
            return None, "cheio"
        fact = {"id": f"f-{self._next:04d}", "categoria": categoria, "fato": fato,
                "trecho_fonte": trecho_fonte, "origem": origem,
                "criado_em": self._now(), "atualizado_em": self._now()}
        self._next += 1
        self.facts.append(fact)
        return fact, "novo"

    def update_text(self, fact_id: str, fato: str) -> bool:
        f = self.get(fact_id)
        if f is None or not fato.strip():
            return False
        f.update(fato=fato.strip()[:MAX_FACT_CHARS], atualizado_em=self._now(), origem="manual")
        return True

    def remove(self, fact_id: str) -> bool:
        before = len(self.facts)
        self.facts = [f for f in self.facts if f["id"] != fact_id]
        return len(self.facts) != before

    def clear(self) -> None:
        self.facts = []

    # ---------- uso no prompt ----------
    def prompt_block(self) -> str:
        """Bloco que entra no prompt de sistema, depois da persona. Vazio se não há fatos."""
        if not self.facts:
            return ""
        lines = ["## O que você já sabe sobre o usuário",
                 "Estes fatos vieram de conversas anteriores e foram guardados pela aplicação a pedido "
                 "do usuário. Use-os com naturalidade quando forem úteis, sem anunciar que lembrou e sem "
                 "listá-los por conta própria. Se o usuário disser algo diferente agora, vale o que ele "
                 "disser agora."]
        for cat in CATEGORIES:
            items = [f["fato"] for f in self.facts if f.get("categoria") == cat]
            if items:
                lines.append(f"{CATEGORY_LABELS[cat]}:")
                lines += [f"- {x}" for x in items]
        return "\n".join(lines)
