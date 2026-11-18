"""Memória de curto prazo da conversa, apenas em RAM, limitada por tokens.

v0.2: orçamento de tokens com corte EM LOTE (até sobrar `trim_ratio`), para o
começo do histórico ficar igual por várias mensagens e o Ollama reaproveitar
o cache do prompt.

v0.3:
- Cada troca guarda também o texto EXATO enviado ao modelo (com a nota de
  data/hora). Reenviar byte a byte o que já foi processado mantém o cache
  inteiro: só a mensagem nova é avaliada.
- O corte pode ser feito em duas fases (`overflow_count` + `drop_oldest`),
  para o motor resumir o lote antes de esquecê-lo.
- `summary`: resumo do que já foi esquecido; entra logo depois da persona.
- Cada troca guarda também a resposta CRUA do modelo (antes do filtro de
  emoji). É ela que volta no prompt: alguns motores do Ollama (ex.: qwen3.5)
  só reaproveitam o cache se a requisição nova for uma continuação exata da
  anterior; qualquer diferença no meio faz reprocessar tudo.
- Trocas internas (`hidden`): abertura e retomada geradas pela aplicação
  para aquecer o cache. Vão ao modelo, não aparecem na tela.

O prompt de sistema (persona) NÃO fica aqui: é passado a cada requisição,
por isso sobrevive à limpeza da memória.
"""

from __future__ import annotations

import math
from typing import NamedTuple

CHARS_PER_TOKEN = 3.0     # estimativa conservadora para português (o real fica entre 3 e 4)
MESSAGE_OVERHEAD = 4      # tokens de marcação do template por mensagem

SUMMARY_NOTE = "[Aplicação: resumo automático do começo desta conversa, que já saiu da sua memória]"
SUMMARY_ACK = "Certo, vou levar isso em conta."


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD


class Exchange(NamedTuple):
    user: str                # texto digitado pelo usuário
    assistant: str           # resposta exibida (depois dos filtros)
    sent: str                # texto enviado ao modelo (com a nota da aplicação, se houve)
    raw: str | None = None   # resposta exatamente como o modelo gerou (None = igual à exibida)
    hidden: bool = False     # troca interna da aplicação (abertura/retomada); não aparece na tela

    @property
    def for_prompt(self) -> str:
        """O que volta ao modelo: o texto gerado, sem filtro, para o cache bater byte a byte."""
        return self.assistant if self.raw is None else self.raw


class ConversationMemory:
    def __init__(self, max_tokens: int = 5000, trim_ratio: float = 0.6):
        if max_tokens < 1:
            raise ValueError("max_tokens deve ser >= 1")
        if not 0 < trim_ratio < 1:
            raise ValueError("trim_ratio deve estar entre 0 e 1")
        self.max_tokens = max_tokens
        self.trim_ratio = trim_ratio
        self._exchanges: list[Exchange] = []
        self.summary = ""  # resumo do que já foi esquecido (RAM)
        self.dropped = 0   # trocas esquecidas desde a última limpeza

    # ---------- escrita ----------
    def add_exchange(self, user_text: str, assistant_text: str, sent_text: str | None = None,
                     auto_trim: bool = True, raw: str | None = None, hidden: bool = False) -> None:
        """Guarda uma troca completa. Com auto_trim, corta em lote se estourar o orçamento."""
        if raw == assistant_text:
            raw = None
        self._exchanges.append(Exchange(user_text, assistant_text, sent_text or user_text, raw, hidden))
        if auto_trim and self.over_budget():
            self.drop_oldest(self.overflow_count())

    def pop_last(self) -> Exchange | None:
        """Remove e devolve a troca visível mais recente (usado em 'Regenerar').

        Trocas internas que estiverem depois dela saem junto.
        """
        while self._exchanges:
            ex = self._exchanges.pop()
            if not ex.hidden:
                return ex
        return None

    def last_visible(self) -> Exchange | None:
        for ex in reversed(self._exchanges):
            if not ex.hidden:
                return ex
        return None

    def extend_last_reply(self, displayed: str, raw: str) -> None:
        """Acrescenta a continuação à última resposta visível, sem criar outra troca."""
        for index in range(len(self._exchanges) - 1, -1, -1):
            ex = self._exchanges[index]
            if ex.hidden:
                continue
            assistant = ex.assistant + displayed
            raw_reply = ex.for_prompt + raw
            self._exchanges[index] = Exchange(
                ex.user, assistant, ex.sent,
                None if raw_reply == assistant else raw_reply, ex.hidden)
            return
        raise ValueError("Não há resposta visível para continuar")

    def drop_hidden(self) -> None:
        """Remove as trocas internas (usado para refazer a abertura de uma conversa vazia)."""
        self._exchanges = [ex for ex in self._exchanges if not ex.hidden]

    def clear(self) -> None:
        self._exchanges.clear()
        self.summary = ""
        self.dropped = 0

    # ---------- corte em duas fases ----------
    def over_budget(self) -> bool:
        return self.tokens() > self.max_tokens

    def overflow_count(self) -> int:
        """Quantas trocas antigas precisam sair para o histórico voltar a `trim_ratio` do orçamento.

        Devolve 0 se o orçamento não estourou. A troca mais recente nunca sai.
        """
        if not self.over_budget():
            return 0
        target = self.max_tokens * self.trim_ratio
        total = self.tokens()
        n = 0
        while n < len(self._exchanges) - 1 and total > target:
            total -= self._cost(self._exchanges[n])
            n += 1
        return n

    def drop_oldest(self, n: int) -> list[Exchange]:
        """Esquece as n trocas mais antigas e as devolve."""
        n = max(0, min(n, len(self._exchanges)))
        gone, self._exchanges = self._exchanges[:n], self._exchanges[n:]
        self.dropped += sum(not ex.hidden for ex in gone)
        return gone

    # ---------- leitura ----------
    def __len__(self) -> int:
        """Trocas visíveis (as internas não contam)."""
        return sum(not ex.hidden for ex in self._exchanges)

    @staticmethod
    def _cost(ex: Exchange) -> int:
        return estimate_tokens(ex.sent) + estimate_tokens(ex.for_prompt)

    def tokens(self) -> int:
        """Estimativa dos tokens ocupados pelo histórico (inclui o resumo)."""
        total = sum(self._cost(ex) for ex in self._exchanges)
        if self.summary:
            total += estimate_tokens(SUMMARY_NOTE + "\n" + self.summary) + estimate_tokens(SUMMARY_ACK)
        return total

    def exchanges(self) -> list[tuple[str, str]]:
        """Pares (usuário, Maddie) sem a nota da aplicação."""
        return [(ex.user, ex.assistant) for ex in self._exchanges if not ex.hidden]

    def visible_pairs(self, upto: int | None = None) -> list[tuple[str, str]]:
        """Pares (usuário, Maddie) das primeiras `upto` trocas, sem as internas e sem a nota."""
        return [(ex.user, ex.assistant) for ex in self._exchanges[:upto] if not ex.hidden]

    def user_texts(self, upto: int | None = None) -> list[str]:
        """Textos do usuário nas primeiras `upto` trocas (as internas ficam de fora)."""
        return [ex.user for ex in self._exchanges[:upto] if not ex.hidden]

    def build_messages(self, system_prompt: str, new_user_text: str, upto: int | None = None) -> list[dict]:
        """Lista de mensagens no formato da API /api/chat do Ollama.

        `upto` limita às primeiras N trocas (o motor usa isso para resumir só o
        lote que vai ser esquecido, aproveitando o cache do começo do prompt).
        """
        messages = [{"role": "system", "content": system_prompt}]
        if self.summary:
            messages.append({"role": "user", "content": SUMMARY_NOTE + "\n" + self.summary})
            messages.append({"role": "assistant", "content": SUMMARY_ACK})
        for ex in self._exchanges[:upto]:
            messages.append({"role": "user", "content": ex.sent})
            messages.append({"role": "assistant", "content": ex.for_prompt})
        messages.append({"role": "user", "content": new_user_text})
        return messages
