"""Motor da conversa: junta persona, memória, nota de data/hora, filtros e Ollama.

A interface (gui.py) e a avaliação (avaliacao/avaliar.py) usam esta mesma
classe, então o que é medido na avaliação é o que o app faz.

Ideia central de desempenho: o Ollama guarda o prompt já processado e só
avalia o que vem DEPOIS do trecho igual ao da requisição anterior. Por isso:
- `warm_up` processa a persona ao abrir o app (1ª mensagem: 28 s -> 3 s);
- o histórico é reenviado byte a byte como foi processado;
- resumo e extração de fatos usam prompts próprios e curtos, sem a persona
  (ver SUMMARY_SYSTEM e EXTRACT_SYSTEM); depois deles uma troca oculta devolve
  o cache à conversa.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import config
from .config import Config
from .context import _local, needs_time_note, short_context, with_time_note
from .memory import ConversationMemory, estimate_tokens
from .memory_store import CATEGORIES, FactStore, validate_candidate
from .ollama_client import ChatStats, OllamaClient, OllamaError, ResponseCancelled
from .textfilter import EmojiFilter, has_emoji, strip_emoji

NEW_MESSAGE_RESERVE = 400   # tokens guardados para a mensagem nova do usuário
MIN_HISTORY_BUDGET = 256


# Trocas internas que aquecem o cache. Ficam no histórico (ocultas) porque alguns
# motores só reaproveitam o cache em continuação exata: o que foi processado
# precisa continuar no prompt, igualzinho.
OPENING = "[Aplicação: nova conversa iniciada em {quando}. Responda apenas: Pronta.]"
RESUME = "[Aplicação: a conversa continua de onde parou. Responda apenas: Certo.]"
REORGANIZED = "[Aplicação: a memória da conversa foi reorganizada. Responda apenas: Certo.]"
CONTINUE_REPLY = (
    "[Aplicação: continue a resposta anterior exatamente do ponto em que parou. "
    "Não repita o que já foi dito; escreva somente a continuação.]"
)
HIDDEN_MAX_TOKENS = 8

# O resumo usa um prompt PRÓPRIO, sem a persona. Motivo (medido em 14/11/2026, qwen3.5:4b):
# com a persona no prompt, o resumo incluía os exemplos de tom dela ("um dia longo",
# "começar a correr") como se fossem falas do usuário, mesmo com instrução em contrário.
SUMMARY_SYSTEM = ("Você resume trechos de conversa entre um usuário e a assistente Maddie. Escreva em "
                  "português, em terceira pessoa, usando só o que está no trecho. Não invente nada.")
SUMMARY_INSTRUCTION = (
    "Resuma o trecho acima em no máximo 8 linhas curtas. Primeiro os fatos que o usuário contou "
    "sobre si (nome, pessoas, animais, onde mora, preferências, planos); depois os assuntos "
    "conversados. Se houver um resumo anterior, mantenha o que ele diz. Não cumprimente nem "
    "comente: escreva apenas o resumo."
)
# A extração também usa prompt próprio, sem a persona e só com as falas do usuário:
# é mais curta (a persona tem ~900 tokens) e o modelo não vê respostas da Maddie
# nem exemplos de tom para confundir com fatos.
EXTRACT_SYSTEM = ("Você extrai fatos duráveis sobre um usuário a partir das falas dele numa conversa. "
                  "Responda só com JSON. Não invente nada.")
EXTRACT_INSTRUCTION = (
    "Das falas acima, extraia os fatos sobre o usuário que valha a pena lembrar em outra conversa: "
    "nome, profissão, cidade, pessoas e animais próximos, gostos e preferências (inclusive como ele "
    "prefere ser atendido), planos e projetos em andamento.\n"
    "Regras:\n"
    "- Só o que o usuário afirmou sobre si mesmo.\n"
    "- Nada de perguntas que ele fez, de assuntos gerais, de dados temporários nem de opiniões "
    "sobre a assistente.\n"
    "- Nada de saúde, finanças, senhas, documentos, endereço ou telefone.\n"
    "- Escreva cada fato em terceira pessoa, começando por \"O usuário\".\n"
    "- Em trecho_fonte, copie exatamente a fala (ou parte dela) de onde o fato saiu.\n"
    "- categoria: identidade (nome, profissão, cidade), pessoas (família, amigos, animais), "
    "preferencias (gostos e preferências) ou contexto (planos e projetos atuais).\n"
    "- Se uma fala atualiza ou contradiz um fato já guardado (mudou de emprego, de cidade, de "
    "gosto), escreva o fato novo e ponha em substitui_id o id do fato antigo, por exemplo f-0001.\n"
    "- Se não houver nada a guardar, devolva a lista vazia.\n"
)
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "fatos": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "categoria": {"type": "string", "enum": list(CATEGORIES)},
                    "fato": {"type": "string"},
                    "trecho_fonte": {"type": "string"},
                    "substitui_id": {"type": "string"},
                },
                "required": ["categoria", "fato", "trecho_fonte"],
            },
        }
    },
    "required": ["fatos"],
}


@dataclass
class ReplyInfo:
    """Medições da última resposta (para a barra de estado e a avaliação)."""
    stats: ChatStats = field(default_factory=ChatStats)
    max_context_tokens: int = 0
    ttft_s: float = 0.0        # tempo até o primeiro pedaço de texto
    total_s: float = 0.0
    cancelled: bool = False
    emoji_removed: int = 0
    raw_reply: str = ""        # texto do modelo antes dos filtros
    reply: str = ""


@dataclass
class MaintenanceReport:
    dropped: int = 0
    summarized: bool = False
    facts: list = field(default_factory=list)      # [(fato, ação)]
    rejected: list = field(default_factory=list)   # [(candidato, motivo)]
    error: str = ""
    seconds: float = 0.0


class ChatSession:
    def __init__(self, cfg: Config, persona: str, base_dir: str | Path | None = None,
                 client: OllamaClient | None = None):
        self.cfg = cfg
        self.persona = persona
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent
        self.client = client or OllamaClient(cfg)
        self.store = FactStore(config.project_path(self.base_dir, cfg.base_memory_file, "base_memory.file"),
                               cfg.base_memory_max_facts)
        self.memory = ConversationMemory(cfg.memory_max_tokens, cfg.memory_trim_ratio)
        self.last = ReplyInfo()
        self._last_note = None  # quando a última nota de data/hora foi enviada
        self.system_prompt = persona
        self.refresh_system()

    # ---------- configuração e prompt de sistema ----------
    def set_config(self, cfg: Config) -> None:
        """Troca a configuração sem mexer no prompt de sistema (isso é do refresh_system)."""
        self.cfg = cfg
        self.client.cfg = cfg
        self.store.path = config.project_path(self.base_dir, cfg.base_memory_file, "base_memory.file")
        self.store.max_facts = cfg.base_memory_max_facts
        self.memory.trim_ratio = cfg.memory_trim_ratio
        self.memory.max_tokens = self.history_budget()

    def refresh_system(self) -> bool:
        """Remonta o prompt de sistema (persona + fatos da memória base). True se mudou.

        Só é chamado no início e em 'Nova conversa': trocar o prompt de sistema no
        meio da conversa obrigaria o Ollama a reprocessar tudo.
        """
        block = ""
        if self.cfg.base_memory_enabled:
            self.store.load()
            block = self.store.prompt_block()
        new = self.persona + ("\n\n" + block if block else "")
        changed = new != self.system_prompt
        self.system_prompt = new
        self.memory.max_tokens = self.history_budget()
        return changed

    def history_budget(self) -> int:
        """Tokens que o histórico pode ocupar sem estourar a janela de contexto."""
        output_reserve = (self.cfg.num_predict if self.cfg.num_predict > 0
                          else min(768, self.cfg.num_ctx // 4))
        free = (self.cfg.num_ctx - output_reserve
                - estimate_tokens(self.system_prompt) - NEW_MESSAGE_RESERVE)
        limit = free if self.cfg.memory_unlimited else min(self.cfg.memory_max_tokens, free)
        return max(MIN_HISTORY_BUDGET, limit)

    # ---------- aquecimento ----------
    def warm_up(self, cancelled: Callable[[], bool] = lambda: False, now=None) -> ChatStats:
        """Carrega o modelo e deixa persona + histórico processados no cache do Ollama.

        Faz isso com uma troca interna de verdade ("Responda apenas: Pronta."), que
        fica oculta no histórico. Assim a 1ª mensagem do usuário é continuação exata
        do que já está no cache, e funciona também nos motores que não conseguem
        reaproveitar o cache a partir do meio (medido no qwen3.5: 27 s -> 0,6 s).
        """
        if len(self.memory) == 0 and not self.memory.summary:
            self.memory.drop_hidden()          # abertura antiga (se houver) é refeita com a hora de agora
            now = _local(now)
            stats = self._hidden_exchange(OPENING.format(quando=short_context(now)), cancelled)
            self._last_note = now              # a abertura já informa data e hora
            return stats
        return self._hidden_exchange(RESUME, cancelled)

    def _hidden_exchange(self, text: str, cancelled: Callable[[], bool] = lambda: False) -> ChatStats:
        stats = ChatStats()
        reply = self.client.chat_once(
            self.memory.build_messages(self.system_prompt, text), cancelled=cancelled, stats=stats,
            options={"num_predict": HIDDEN_MAX_TOKENS, "temperature": 0.0})
        if reply.strip():
            self.memory.add_exchange(text, reply, text, auto_trim=False, hidden=True)
        return stats

    # ---------- conversa ----------
    def _user_uses_emoji(self, new_text: str) -> bool:
        return has_emoji(new_text) or any(has_emoji(t) for t in self.memory.user_texts())

    def reply(self, user_text: str, cancelled: Callable[[], bool] = lambda: False,
              now=None) -> Iterator[str]:
        """Envia a mensagem e gera os pedaços da resposta. Ao terminar, guarda a troca.

        Se for interrompida (ResponseCancelled), guarda o que já tinha sido escrito.
        Em erro do Ollama, nada é guardado.
        """
        if self.memory.over_budget():  # manutenção não rodou: corta sem resumo
            self.memory.drop_oldest(self.memory.overflow_count())
        now = _local(now)
        noted = needs_time_note(user_text, self._last_note, now)
        sent = with_time_note(user_text, now) if noted else user_text
        if noted:
            self._last_note = now
        messages = self.memory.build_messages(self.system_prompt, sent)
        info = ReplyInfo()
        self.last = info
        filt = EmojiFilter() if self.cfg.strip_emoji and not self._user_uses_emoji(user_text) else None
        raw: list[str] = []
        shown: list[str] = []
        t0 = time.perf_counter()

        def commit() -> None:
            info.total_s = time.perf_counter() - t0
            info.raw_reply = "".join(raw)
            info.reply = "".join(shown)
            info.emoji_removed = filt.removed if filt else 0
            info.max_context_tokens = max(
                info.max_context_tokens, info.stats.prompt_tokens + info.stats.output_tokens)
            if info.reply.strip():
                self.memory.add_exchange(user_text, info.reply, sent, auto_trim=False, raw=info.raw_reply)

        try:
            for chunk in self.client.chat_stream(messages, cancelled=cancelled, stats=info.stats):
                raw.append(chunk)
                out = filt.feed(chunk) if filt else chunk
                if out:
                    if not shown:
                        out = out.lstrip()
                        if not out:
                            continue
                        info.ttft_s = time.perf_counter() - t0
                    shown.append(out)
                    yield out
        except ResponseCancelled:
            info.cancelled = True
            commit()
            raise
        commit()

    def continue_reply(self, cancelled: Callable[[], bool] = lambda: False) -> Iterator[str]:
        """Continua a última resposta após reorganizar a memória, mantendo uma única troca."""
        info = self.last
        if not info.reply:
            raise RuntimeError("Não há uma resposta parcial para continuar")
        last = self.memory.last_visible()
        if last is None:
            raise RuntimeError("A resposta parcial não está na memória da conversa")

        messages = self.memory.build_messages(self.system_prompt, CONTINUE_REPLY)
        filt = EmojiFilter() if self.cfg.strip_emoji and not self._user_uses_emoji(last.user) else None
        raw: list[str] = []
        shown: list[str] = []
        stats = ChatStats()
        t0 = time.perf_counter()
        first_text = True

        def commit() -> None:
            raw_text, shown_text = "".join(raw), "".join(shown)
            self.memory.extend_last_reply(shown_text, raw_text)
            info.raw_reply += raw_text
            info.reply += shown_text
            info.emoji_removed += filt.removed if filt else 0
            info.total_s += time.perf_counter() - t0
            info.stats.prompt_tokens += stats.prompt_tokens
            info.stats.output_tokens += stats.output_tokens
            info.stats.total_s += stats.total_s
            info.stats.load_s += stats.load_s
            info.stats.prompt_eval_s += stats.prompt_eval_s
            info.stats.eval_s += stats.eval_s
            info.stats.done_reason = stats.done_reason
            info.max_context_tokens = max(
                info.max_context_tokens, stats.prompt_tokens + stats.output_tokens)

        try:
            for chunk in self.client.chat_stream(messages, cancelled=cancelled, stats=stats):
                raw.append(chunk)
                out = filt.feed(chunk) if filt else chunk
                if out:
                    if first_text:
                        if info.ttft_s == 0:
                            info.ttft_s = time.perf_counter() - t0
                        first_text = False
                    shown.append(out)
                    yield out
        except ResponseCancelled:
            info.cancelled = True
            commit()
            raise
        except Exception:
            commit()
            raise
        commit()

    def pop_last(self) -> str | None:
        """Desfaz a última troca e devolve o texto do usuário (para 'Regenerar')."""
        ex = self.memory.pop_last()
        if ex is not None and ex.sent != ex.user:
            self._last_note = None  # a nota saiu junto: a próxima mensagem leva outra
        return ex.user if ex else None

    def clear(self) -> bool:
        """Nova conversa. Devolve True se o prompt de sistema mudou (fatos novos na memória base)."""
        self.memory.clear()
        self._last_note = None
        return self.refresh_system()

    # ---------- manutenção da memória (rodar com o app ocioso) ----------
    def maintenance_due(self) -> bool:
        return self.memory.over_budget() and self.memory.overflow_count() > 0

    def response_was_limited(self) -> bool:
        """Detecta resposta truncada, inclusive em versões do Ollama sem done_reason."""
        stats = self.last.stats
        if stats.done_reason:
            return stats.done_reason == "length"
        return ((self.cfg.num_predict != -1 and stats.output_tokens >= self.cfg.num_predict)
                or stats.prompt_tokens + stats.output_tokens >= self.cfg.num_ctx)

    def maintain(self, cancelled: Callable[[], bool] = lambda: False,
                 rewarm: bool = True) -> MaintenanceReport:
        """Resume (e, se ligado, extrai fatos de) o lote antigo, esquece-o e reaquece o cache.

        Custo: o histórico que sobra é reprocessado inteiro (1 a 3 min em CPU). Por
        isso roda em segundo plano, logo depois da resposta que estourou o orçamento.
        """
        rep = MaintenanceReport()
        t0 = time.perf_counter()
        n = self.memory.overflow_count()
        if n == 0:
            return rep
        if self.cfg.base_memory_enabled:
            try:
                rep.facts, rep.rejected = self._extract(upto=n, cancelled=cancelled)
            except ResponseCancelled:
                raise
            except (OllamaError, OSError) as exc:
                rep.error = f"extração de fatos falhou: {exc}"
        summary = ""
        if self.cfg.memory_summarize:
            try:
                summary = self._summarize(upto=n, cancelled=cancelled)
            except ResponseCancelled:
                raise
            except OllamaError as exc:
                rep.error = (rep.error + "; " if rep.error else "") + f"resumo falhou: {exc}"
        rep.dropped = len(self.memory.drop_oldest(n))
        if summary:
            self.memory.summary = summary
            rep.summarized = True
        if rewarm:
            try:
                self._hidden_exchange(REORGANIZED, cancelled)
            except ResponseCancelled:
                raise
            except OllamaError as exc:
                rep.error = (rep.error + "; " if rep.error else "") + f"reaquecimento falhou: {exc}"
        rep.seconds = time.perf_counter() - t0
        return rep

    def _summarize(self, upto: int, cancelled: Callable[[], bool]) -> str:
        parts = []
        if self.memory.summary:
            parts.append("Resumo anterior:\n" + self.memory.summary)
        lines = []
        for user, assistant in self.memory.visible_pairs(upto):
            lines += [f"Usuário: {user.strip()}", f"Maddie: {assistant.strip()}"]
        parts.append("Trecho da conversa:\n" + "\n".join(lines))
        parts.append(SUMMARY_INSTRUCTION)
        messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)}]
        text = self.client.chat_once(
            messages, cancelled=cancelled,
            options={"num_predict": self.cfg.summary_max_tokens, "temperature": 0.2})
        return strip_emoji(text).strip()

    # ---------- memória base ----------
    def extract_facts(self, cancelled: Callable[[], bool] = lambda: False) -> tuple[list, list]:
        """Extrai fatos da conversa inteira (botão 'Guardar fatos desta conversa')."""
        if not self.cfg.base_memory_enabled:
            raise OllamaError("A memória permanente está desligada.")
        result = self._extract(upto=None, cancelled=cancelled)
        try:  # o pedido de extração ficou no cache; a retomada devolve o cache à conversa
            self._hidden_exchange(RESUME, cancelled)
        except ResponseCancelled:
            raise
        except OllamaError:
            pass
        return result

    def _extract(self, upto: int | None, cancelled: Callable[[], bool]) -> tuple[list, list]:
        user_texts = self.memory.user_texts(upto)
        if not user_texts:
            return [], []
        self.store.load()
        if self.store.load_error:
            raise OSError(self.store.load_error)
        known = "\n".join(f"- [{f['id']}] {f['fato']}" for f in self.store.facts) or "(nenhum)"
        falas = "\n".join(f"{i}. {' '.join(t.split())}" for i, t in enumerate(user_texts, 1))
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Falas do usuário:\n{falas}\n\n{EXTRACT_INSTRUCTION}\n"
                                        f"Fatos já guardados (não repita):\n{known}\n\nResponda em JSON."},
        ]
        raw = self.client.chat_once(messages, fmt=EXTRACT_SCHEMA, cancelled=cancelled,
                                    options={"num_predict": 600, "temperature": 0.1})
        return self.apply_candidates(parse_candidates(raw), user_texts)

    def apply_candidates(self, candidates: list, user_texts: list[str]) -> tuple[list, list]:
        """Valida e grava candidatos. Devolve ([(fato, ação)], [(candidato, motivo)])."""
        accepted, rejected = [], []
        for cand in candidates:
            ok, reason = validate_candidate(cand, user_texts)
            if not ok:
                rejected.append((cand, reason))
                continue
            fact, action = self.store.add(
                str(cand["categoria"]).strip().lower(), str(cand["fato"]),
                str(cand["trecho_fonte"]).strip(), str(cand.get("substitui_id") or "") or None)
            if action in ("novo", "atualizado"):
                accepted.append((fact, action))
            elif action == "cheio":
                rejected.append((cand, "memória base cheia"))
        if accepted:
            self.store.save()
        return accepted, rejected


def parse_candidates(raw: str) -> list:
    """Lê a lista 'fatos' do JSON devolvido pelo modelo (tolerante a lixo em volta)."""
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    facts = data.get("fatos", []) if isinstance(data, dict) else []
    return [f for f in facts if isinstance(f, dict)] if isinstance(facts, list) else []
