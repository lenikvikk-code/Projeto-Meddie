"""Contexto dinâmico da sessão (data/hora local), gerado a cada requisição.

v0.2: a data/hora deixou de ir no prompt de sistema e passou para uma nota no
início da mensagem do usuário. Motivo (medido em 13/11/2026): o Ollama só
reaproveita o cache do prompt se o começo for idêntico byte a byte. Com a hora
no topo, cada minuto invalidava o cache e a conversa inteira era reprocessada.

v0.3: a nota ficou curta e passou a ser guardada no histórico tal como foi
enviada (ver `with_time_note`).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]

NOTE_PREFIX = "[Aplicação:"


def _local(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now().astimezone()  # relógio do sistema, com fuso local
    if now.tzinfo is None:
        return now.astimezone()  # datetime ingênuo: interpreta como hora local
    return now


def _utc_offset(now: datetime) -> str:
    off = now.utcoffset() or timedelta(0)
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    h, m = divmod(abs(total), 60)
    return f"UTC{sign}{h:02d}:{m:02d}"


def temporal_context(now: datetime | None = None) -> str:
    """Data, dia da semana, hora e fuso em uma linha, lidos do relógio local."""
    now = _local(now)
    tz_name = now.tzname() or ""
    fuso = _utc_offset(now) + (f" {tz_name}" if tz_name and not tz_name.startswith("UTC") else "")
    return (f"{DIAS[now.weekday()]}, {now.day} de {MESES[now.month - 1]} de {now.year} "
            f"({now:%d/%m/%Y}), {now:%H:%M}, fuso {fuso}")


def short_context(now: datetime | None = None) -> str:
    """Dia da semana, data e hora em formato curto: 'sábado, 14/11/2026, 12:50'."""
    now = _local(now)
    return f"{DIAS[now.weekday()]}, {now:%d/%m/%Y}, {now:%H:%M}"


def with_time_note(user_text: str, now: datetime | None = None) -> str:
    """Mensagem do usuário precedida da nota automática da aplicação.

    v0.3: nota curta (cerca de 20 tokens) e GUARDADA no histórico exatamente como
    foi enviada. Na v0.2 a memória guardava o texto sem a nota, e por isso a
    troca anterior era reprocessada a cada resposta (o cache do Ollama exige o
    mesmo começo, byte a byte). Guardando igual, só a mensagem nova é avaliada.
    """
    return f"{NOTE_PREFIX} {short_context(now)}]\n{user_text}"


NOTE_RE = re.compile(r"^\[Aplicação:[^\]\n]*\]\n?")


def strip_time_note(text: str) -> str:
    """Remove a nota da aplicação do início de um texto, se houver."""
    return NOTE_RE.sub("", text, count=1)


# ---------- quando a nota é necessária ----------
# Medido em 14/11/2026 (Qwen3 4B, CPU): a nota custa de 1,1 a 1,9 s no primeiro
# token de CADA mensagem (datas e horas viram muitos tokens). Por isso ela só vai
# quando pode fazer falta: na 1ª mensagem, quando a mensagem fala de tempo, ou
# quando a última nota enviada já está velha.
NOTE_MAX_AGE_S = 600

_TIME_WORDS = re.compile(
    r"\b(horas?|horario|relogio|datas?|dias?|hoje|agora|ontem|anteontem|amanha|semanas?|mes|meses|"
    r"anos?|quando|tempo|minutos?|segundos?|manha|tarde|noite|madrugada|cedo|feriado|natal|"
    r"aniversario|idade|faltam?|prazo|desde|calendario|atrasad[oa]|"
    r"segunda|terca|quarta|quinta|sexta|sabado|domingo|"
    r"janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|"
    r"today|now|time|date|day|tomorrow|yesterday)\b"
)


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def mentions_time(text: str) -> bool:
    return bool(_TIME_WORDS.search(_plain(text)))


def needs_time_note(user_text: str, last_note: datetime | None, now: datetime | None = None) -> bool:
    """True se esta mensagem deve levar a nota de data/hora."""
    if last_note is None:
        return True
    now = _local(now)
    if abs((now - _local(last_note)).total_seconds()) >= NOTE_MAX_AGE_S:
        return True
    if now.date() != _local(last_note).date():
        return True
    return mentions_time(user_text)