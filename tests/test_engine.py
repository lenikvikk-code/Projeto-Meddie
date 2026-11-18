"""Testes do motor (ChatSession) com um cliente falso: não precisam do Ollama."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from maddie import config
from maddie.context import NOTE_PREFIX
from maddie.engine import (CONTINUE_REPLY, EXTRACT_SYSTEM, HIDDEN_MAX_TOKENS, SUMMARY_SYSTEM,
                           ChatSession, parse_candidates)
from maddie.memory import SUMMARY_NOTE
from maddie.ollama_client import OllamaError, ResponseCancelled

BRT = timezone(timedelta(hours=-3), "BRT")
T0 = datetime(2026, 11, 14, 12, 50, tzinfo=BRT)
PERSONA = "Você é a Maddie."


class FakeClient:
    """Responde com o que estiver em `script`; tarefas internas têm resposta própria."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[dict] = []
        self.script: list = []
        self.summary = "O usuário se chama Rafael."
        self.extraction = {"fatos": []}
        self.fail_internal = False

    def chat_stream(self, messages, cancelled=lambda: False, stats=None, options=None, fmt=None):
        self.calls.append({"messages": [dict(m) for m in messages], "options": options, "fmt": fmt})
        last = messages[-1]["content"]
        if messages[0]["content"] in (SUMMARY_SYSTEM, EXTRACT_SYSTEM):
            if self.fail_internal:
                raise OllamaError("falha simulada")
            yield json.dumps(self.extraction) if fmt else self.summary
            return
        if options and options.get("num_predict") == HIDDEN_MAX_TOKENS:  # troca interna de aquecimento
            yield "Pronta." if "Pronta" in last else "Certo."
            return
        item = self.script.pop(0) if self.script else "ok"
        for chunk in (item if isinstance(item, list) else [item]):
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def chat_once(self, messages, options=None, fmt=None, cancelled=lambda: False, stats=None):
        return "".join(self.chat_stream(messages, cancelled, stats, options, fmt))

    def warm_up(self, messages, cancelled=lambda: False, stats=None):
        self.chat_once(messages, options={"num_predict": 1}, cancelled=cancelled, stats=stats)


def make(tmp, **changes):
    cfg = config.replace(config.Config(), **changes)
    client = FakeClient(cfg)
    return ChatSession(cfg, PERSONA, tmp, client=client), client


class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_reply_streams_and_stores_exchange(self):
        s, c = make(self.tmp.name)
        c.script = [["Olá", ", Rafael."]]
        self.assertEqual("".join(s.reply("Meu nome é Rafael.", now=T0)), "Olá, Rafael.")
        self.assertEqual(s.memory.exchanges(), [("Meu nome é Rafael.", "Olá, Rafael.")])
        self.assertEqual(c.calls[0]["messages"][0], {"role": "system", "content": PERSONA})
        self.assertTrue(c.calls[0]["messages"][-1]["content"].startswith(NOTE_PREFIX))

    def test_history_is_resent_byte_for_byte(self):
        """Cache do prompt: a 2ª requisição começa exatamente com a 1ª + a resposta."""
        s, c = make(self.tmp.name)
        c.script = ["um", "dois"]
        list(s.reply("primeira", now=T0))
        list(s.reply("segunda", now=T0 + timedelta(minutes=1)))
        first, second = c.calls[0]["messages"], c.calls[1]["messages"]
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(second[len(first)], {"role": "assistant", "content": "um"})

    def test_time_note_only_when_needed(self):
        s, c = make(self.tmp.name)
        list(s.reply("oi", now=T0))
        list(s.reply("qual é a capital do Japão?", now=T0 + timedelta(minutes=1)))
        list(s.reply("que horas são?", now=T0 + timedelta(minutes=2)))
        list(s.reply("e a da França?", now=T0 + timedelta(minutes=20)))
        noted = [call["messages"][-1]["content"].startswith(NOTE_PREFIX) for call in c.calls]
        self.assertEqual(noted, [True, False, True, True])

    def test_emoji_removed_unless_user_uses_emoji(self):
        s, c = make(self.tmp.name)
        c.script = [["Boa ideia ", "😊", ". Vamos lá!", " 🚀"], ["Claro 😄"]]
        self.assertEqual("".join(s.reply("vou correr", now=T0)), "Boa ideia. Vamos lá!")
        self.assertEqual(s.last.emoji_removed, 2)
        self.assertIn("😊", s.last.raw_reply)
        self.assertEqual("".join(s.reply("legal 😀", now=T0)), "Claro 😄")  # o usuário usou primeiro

    def test_emoji_kept_when_filter_disabled(self):
        s, c = make(self.tmp.name, strip_emoji=False)
        c.script = ["Oi 😊"]
        self.assertEqual("".join(s.reply("oi", now=T0)), "Oi 😊")

    def test_cancel_keeps_partial_reply(self):
        s, c = make(self.tmp.name)
        c.script = [["Começo da resposta", ResponseCancelled("parou")]]
        with self.assertRaises(ResponseCancelled):
            list(s.reply("conte uma história", now=T0))
        self.assertTrue(s.last.cancelled)
        self.assertEqual(s.memory.exchanges(), [("conte uma história", "Começo da resposta")])

    def test_error_stores_nothing(self):
        s, c = make(self.tmp.name)
        c.script = [["pedaço", OllamaError("caiu")]]
        with self.assertRaises(OllamaError):
            list(s.reply("oi", now=T0))
        self.assertEqual(len(s.memory), 0)

    def test_pop_last_for_regenerate(self):
        s, c = make(self.tmp.name)
        list(s.reply("oi", now=T0))
        self.assertEqual(s.pop_last(), "oi")
        self.assertEqual(len(s.memory), 0)
        list(s.reply("oi", now=T0 + timedelta(minutes=1)))
        self.assertTrue(c.calls[-1]["messages"][-1]["content"].startswith(NOTE_PREFIX))  # nota volta

    def test_warm_up_is_a_hidden_exchange_kept_in_the_prompt(self):
        """A 1ª mensagem precisa ser continuação exata do que o aquecimento processou."""
        s, c = make(self.tmp.name)
        s.warm_up(now=T0)
        warm = c.calls[0]["messages"]
        self.assertEqual(warm[0]["content"], s.system_prompt)
        self.assertIn("14/11/2026, 12:50", warm[1]["content"])
        self.assertEqual((len(s.memory), s.memory.exchanges()), (0, []))   # invisível para o usuário
        list(s.reply("oi", now=T0 + timedelta(minutes=1)))
        first = c.calls[1]["messages"]
        self.assertEqual(first[:2], warm)
        self.assertEqual(first[2], {"role": "assistant", "content": "Pronta."})
        self.assertEqual(first[3], {"role": "user", "content": "oi"})      # sem nota: a abertura já deu a hora

    def test_warm_up_twice_on_empty_conversation_does_not_stack(self):
        s, c = make(self.tmp.name)
        s.warm_up(now=T0)
        s.warm_up(now=T0 + timedelta(minutes=5))
        self.assertEqual(len(c.calls[1]["messages"]), 2)
        self.assertIn("12:55", c.calls[1]["messages"][1]["content"])

    def test_raw_reply_goes_back_to_the_model(self):
        """O filtro de emoji vale para a tela; o prompt reenvia o que o modelo gerou (cache)."""
        s, c = make(self.tmp.name)
        c.script = ["Boa 😊", "ok"]
        list(s.reply("oi", now=T0))
        list(s.reply("e aí", now=T0))
        self.assertEqual(c.calls[1]["messages"][2]["content"], "Boa 😊")
        self.assertEqual(s.memory.exchanges()[0], ("oi", "Boa"))

    def test_history_budget_shrinks_with_big_system_prompt(self):
        s, _ = make(self.tmp.name)
        self.assertEqual(s.history_budget(), 5500)
        s.system_prompt = "x" * 9000  # 3000 tokens estimados
        self.assertLess(s.history_budget(), 5500)
        self.assertLess(s.history_budget() + 3000 + s.cfg.num_predict, s.cfg.num_ctx)

    def test_unlimited_history_uses_the_model_context_budget(self):
        s, _ = make(self.tmp.name, memory_unlimited=True)
        self.assertGreater(s.history_budget(), s.cfg.memory_max_tokens)
        self.assertEqual(s.memory.max_tokens, s.history_budget())

    def test_continue_reply_extends_the_same_exchange_after_rewarm(self):
        s, c = make(self.tmp.name, strip_emoji=False)
        c.script = ["A resposta parou", " e continuou."]
        list(s.reply("pergunta", now=T0))
        s.memory.add_exchange("aplicação", "Certo.", auto_trim=False, hidden=True)

        continuation = "".join(s.continue_reply())

        self.assertEqual(continuation, " e continuou.")
        self.assertEqual(s.last.reply, "A resposta parou e continuou.")
        self.assertEqual(s.memory.exchanges(), [("pergunta", "A resposta parou e continuou.")])
        messages = c.calls[-1]["messages"]
        self.assertEqual(messages[-1]["content"], CONTINUE_REPLY)
        self.assertEqual(messages[-4]["content"], "A resposta parou")

    def test_detects_truncation_with_and_without_done_reason(self):
        s, _ = make(self.tmp.name)
        s.last.stats.done_reason = "length"
        self.assertTrue(s.response_was_limited())
        s.last.stats.done_reason = ""
        s.last.stats.output_tokens = s.cfg.num_predict
        self.assertTrue(s.response_was_limited())
        s.last.stats.output_tokens = 1
        s.last.stats.prompt_tokens = s.cfg.num_ctx - 1
        self.assertTrue(s.response_was_limited())
        s.last.stats.prompt_tokens = 0
        self.assertFalse(s.response_was_limited())


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fill(self, s, n=None):
        """Sem n: conversa até a manutenção ficar pendente (sem deixar o corte de segurança agir)."""
        i = 0
        while (n is None and not s.maintenance_due()) or (n is not None and i < n):
            list(s.reply(f"mensagem {i} " + "x" * 300, now=T0))
            i += 1

    def small(self, **kw):
        s, c = make(self.tmp.name, **kw)
        s.memory.max_tokens = 800
        c.script = ["resposta " + "y" * 300] * 50
        return s, c

    def test_summarizes_before_forgetting(self):
        s, c = self.small()
        self.fill(s)
        self.assertTrue(s.maintenance_due())
        before = len(s.memory)
        rep = s.maintain()
        self.assertGreater(rep.dropped, 0)
        self.assertTrue(rep.summarized)
        self.assertEqual(len(s.memory), before - rep.dropped)
        self.assertEqual(s.memory.summary, "O usuário se chama Rafael.")
        self.assertFalse(s.maintenance_due())
        msgs = s.memory.build_messages(s.system_prompt, "x")
        self.assertTrue(msgs[1]["content"].startswith(SUMMARY_NOTE))
        self.assertIn("reorganizada", c.calls[-1]["messages"][-1]["content"])  # reaqueceu por último
        self.assertEqual(c.calls[-1]["messages"][1]["content"], msgs[1]["content"])

    def test_summary_request_has_only_the_forgotten_batch_and_no_persona(self):
        """O resumo vai com prompt próprio: sem a persona (os exemplos dela vazavam para o resumo)."""
        s, c = self.small()
        self.fill(s)
        n = s.memory.overflow_count()
        going, staying = s.memory.visible_pairs(n), s.memory.visible_pairs()[-1]
        s.maintain(rewarm=False)
        call = c.calls[-1]["messages"]
        self.assertEqual([m["role"] for m in call], ["system", "user"])
        self.assertEqual(call[0]["content"], SUMMARY_SYSTEM)
        self.assertNotIn(PERSONA, call[1]["content"])
        self.assertNotIn(NOTE_PREFIX, call[1]["content"])
        for user, _ in going:
            self.assertIn(user, call[1]["content"])
        self.assertNotIn(staying[0], call[1]["content"])

    def test_second_summary_carries_the_previous_one(self):
        s, c = self.small()
        self.fill(s)
        s.maintain(rewarm=False)
        c.summary = "Resumo novo."
        self.fill(s)
        s.maintain(rewarm=False)
        self.assertIn("Resumo anterior:\nO usuário se chama Rafael.", c.calls[-1]["messages"][1]["content"])
        self.assertEqual(s.memory.summary, "Resumo novo.")
    def test_internal_failure_falls_back_to_plain_trim(self):
        s, c = self.small()
        self.fill(s)
        c.fail_internal = True
        rep = s.maintain(rewarm=False)
        self.assertGreater(rep.dropped, 0)
        self.assertFalse(rep.summarized)
        self.assertIn("resumo falhou", rep.error)
        self.assertEqual(s.memory.summary, "")

    def test_no_summary_when_disabled(self):
        s, c = self.small(memory_summarize=False)
        self.fill(s)
        calls = len(c.calls)
        rep = s.maintain(rewarm=False)
        self.assertFalse(rep.summarized)
        self.assertEqual(len(c.calls), calls)

    def test_reply_trims_if_maintenance_never_ran(self):
        s, c = self.small()
        self.fill(s, 20)
        self.assertLessEqual(len(s.memory), 12)  # o corte de segurança agiu dentro de reply()

    def test_clear_resets_summary(self):
        s, c = self.small()
        self.fill(s)
        s.maintain(rewarm=False)
        s.clear()
        self.assertEqual((len(s.memory), s.memory.summary, s.memory.dropped), (0, "", 0))


class BaseMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "memoria" / "memoria_base.json"

    def test_disabled_by_default_and_writes_nothing(self):
        s, c = make(self.tmp.name)
        self.assertFalse(s.cfg.base_memory_enabled)
        list(s.reply("Meu nome é Rafael.", now=T0))
        with self.assertRaises(OllamaError):
            s.extract_facts()
        self.assertFalse(self.file.exists())
        self.assertEqual(s.system_prompt, PERSONA)

    def test_extraction_validates_and_saves(self):
        s, c = make(self.tmp.name, base_memory_enabled=True)
        list(s.reply("Meu nome é Rafael e tenho uma cachorra chamada Pipoca.", now=T0))
        list(s.reply("Qual é a capital do Japão?", now=T0))
        c.extraction = {"fatos": [
            {"categoria": "identidade", "fato": "O usuário se chama Rafael.",
             "trecho_fonte": "Meu nome é Rafael"},
            {"categoria": "pessoas", "fato": "O usuário tem uma cachorra chamada Pipoca.",
             "trecho_fonte": "tenho uma cachorra chamada Pipoca"},
            {"categoria": "identidade", "fato": "O usuário mora em Tóquio.",       # fonte inventada
             "trecho_fonte": "Eu moro em Tóquio"},
            {"categoria": "contexto", "fato": "O usuário perguntou a capital do Japão.",  # pergunta
             "trecho_fonte": "Qual é a capital do Japão?"},
        ]}
        accepted, rejected = s.extract_facts()
        self.assertEqual([f["fato"] for f, _ in accepted],
                         ["O usuário se chama Rafael.", "O usuário tem uma cachorra chamada Pipoca."])
        self.assertEqual(len(rejected), 2)
        saved = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["fatos"]), 2)
        self.assertIsNotNone(c.calls[-2]["fmt"])  # saída estruturada (JSON Schema)
        sent = c.calls[-2]["messages"]
        self.assertEqual(sent[0]["content"], EXTRACT_SYSTEM)          # sem a persona
        self.assertIn("1. Meu nome é Rafael e tenho uma cachorra chamada Pipoca.", sent[1]["content"])
        self.assertNotIn("ok", sent[1]["content"].split("Falas do usuário:")[1].split("\n\n")[0])  # sem falas da Maddie
        self.assertIn("continua", c.calls[-1]["messages"][-1]["content"])  # retomada devolve o cache

    def test_facts_enter_system_prompt_only_on_next_conversation(self):
        s, c = make(self.tmp.name, base_memory_enabled=True)
        list(s.reply("Meu nome é Rafael.", now=T0))
        c.extraction = {"fatos": [{"categoria": "identidade", "fato": "O usuário se chama Rafael.",
                                   "trecho_fonte": "Meu nome é Rafael."}]}
        s.extract_facts()
        self.assertEqual(s.system_prompt, PERSONA)       # não muda no meio da conversa (cache)
        self.assertTrue(s.clear())                        # nova conversa: prompt mudou
        self.assertIn("O usuário se chama Rafael.", s.system_prompt)
        s2, _ = make(self.tmp.name, base_memory_enabled=True)   # outra sessão do app
        self.assertIn("O usuário se chama Rafael.", s2.system_prompt)

    def test_maintenance_extracts_from_the_forgotten_batch(self):
        s, c = make(self.tmp.name, base_memory_enabled=True)
        s.memory.max_tokens = 800
        c.script = ["resposta " + "y" * 300] * 30
        list(s.reply("Eu trabalho como enfermeira num hospital.", now=T0))
        i = 0
        while not s.maintenance_due():
            list(s.reply(f"mensagem {i} " + "x" * 300, now=T0))
            i += 1
        c.extraction = {"fatos": [{"categoria": "identidade", "fato": "O usuário trabalha como enfermeira.",
                                   "trecho_fonte": "Eu trabalho como enfermeira num hospital."}]}
        rep = s.maintain(rewarm=False)
        self.assertEqual([f["fato"] for f, _ in rep.facts], ["O usuário trabalha como enfermeira."])

    def test_parse_candidates_tolerates_garbage(self):
        self.assertEqual(parse_candidates("não é json"), [])
        self.assertEqual(parse_candidates('{"fatos": "x"}'), [])
        self.assertEqual(parse_candidates('lixo {"fatos": [{"fato": "a"}, 3]} lixo'), [{"fato": "a"}])


if __name__ == "__main__":
    unittest.main()
