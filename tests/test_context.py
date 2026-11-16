import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from maddie import personality
from maddie.context import NOTE_PREFIX, needs_time_note, strip_time_note, temporal_context, with_time_note
from maddie.memory import ConversationMemory

ROOT = Path(__file__).resolve().parent.parent
BRT = timezone(timedelta(hours=-3), "BRT")


class TemporalContextTests(unittest.TestCase):
    def test_contains_date_time_weekday_and_timezone(self):
        now = datetime(2026, 11, 11, 23, 5, tzinfo=BRT)  # quarta-feira
        txt = temporal_context(now)
        self.assertIn("11 de novembro de 2026", txt)
        self.assertIn("11/11/2026", txt)
        self.assertIn("quarta-feira", txt)
        self.assertIn("23:05", txt)
        self.assertIn("UTC-03:00", txt)
        self.assertIn("BRT", txt)

    def test_weekdays_and_months_in_portuguese(self):
        txt = temporal_context(datetime(2027, 3, 7, 8, 0, tzinfo=timezone.utc))  # domingo
        self.assertIn("domingo", txt)
        self.assertIn("7 de março de 2027", txt)
        self.assertIn("UTC+00:00", txt)

    def test_uses_system_clock_when_no_argument(self):
        self.assertIn(datetime.now().astimezone().strftime("%d/%m/%Y"), temporal_context())


class TimeNoteTests(unittest.TestCase):
    def setUp(self):
        self.persona = personality.load_persona(ROOT / "persona.md")

    def test_note_precedes_user_text(self):
        sent = with_time_note("Que dia é hoje?", datetime(2026, 11, 11, 23, 5, tzinfo=BRT))
        self.assertTrue(sent.startswith(NOTE_PREFIX))
        self.assertTrue(sent.endswith("\nQue dia é hoje?"))
        self.assertIn("11/11/2026", sent)

    def test_system_prompt_is_stable_across_minutes(self):
        """Cache do prompt: o prefixo (persona + histórico) não pode mudar com a hora."""
        mem = ConversationMemory()
        mem.add_exchange("oi", "olá")
        a = mem.build_messages(self.persona, with_time_note("x", datetime(2026, 1, 1, 10, 0, tzinfo=BRT)))
        b = mem.build_messages(self.persona, with_time_note("x", datetime(2026, 1, 1, 10, 1, tzinfo=BRT)))
        self.assertEqual(a[:-1], b[:-1])
        self.assertNotEqual(a[-1], b[-1])
        self.assertEqual(sum(m["role"] == "system" for m in a), 1)

    def test_note_is_short(self):
        sent = with_time_note("oi", datetime(2026, 11, 14, 12, 50, tzinfo=BRT))
        self.assertEqual(sent, "[Aplicação: sábado, 14/11/2026, 12:50]\noi")
        self.assertEqual(strip_time_note(sent), "oi")
        self.assertEqual(strip_time_note("sem nota"), "sem nota")

    def test_memory_resends_exactly_what_was_sent(self):
        """Cache: o histórico reenvia a mensagem com a nota; a leitura para o usuário vem sem ela."""
        mem = ConversationMemory()
        sent = with_time_note("oi", datetime(2026, 11, 14, 12, 50, tzinfo=BRT))
        mem.add_exchange("oi", "olá", sent)
        self.assertEqual(mem.build_messages(self.persona, "x")[1]["content"], sent)
        self.assertNotIn(NOTE_PREFIX, " ".join(u + a for u, a in mem.exchanges()))

    def test_note_only_when_needed(self):
        t0 = datetime(2026, 11, 14, 12, 50, tzinfo=BRT)
        self.assertTrue(needs_time_note("oi", None, t0))                              # 1ª mensagem
        self.assertFalse(needs_time_note("qual é a capital do Japão?", t0, t0 + timedelta(minutes=3)))
        self.assertTrue(needs_time_note("qual é a capital do Japão?", t0, t0 + timedelta(minutes=11)))
        for text in ("Que horas são?", "que dia é hoje", "Bom dia!", "falta muito pro Natal?",
                     "QUE DIA DA SEMANA É AMANHÃ?", "é sábado?", "what time is it"):
            with self.subTest(text=text):
                self.assertTrue(needs_time_note(text, t0, t0 + timedelta(minutes=1)))
    def test_identity_survives_clear(self):
        mem = ConversationMemory()
        mem.add_exchange("Meu nome é Rafael.", "Prazer, Rafael.")
        mem.clear()
        msgs = mem.build_messages(self.persona, with_time_note("Quem é você?"))
        self.assertIn("Maddie", msgs[0]["content"])
        self.assertNotIn("Rafael", " ".join(m["content"] for m in msgs))


class PersonaContentTests(unittest.TestCase):
    """Garante regras de projeto sobre a persona (não sobre a qualidade do modelo)."""

    def setUp(self):
        self.text = personality.load_persona(ROOT / "persona.md")

    def test_no_hardcoded_user_name_or_fixed_answers(self):
        low = self.text.lower()
        for forbidden in ("lenik", "rafael", "pedro", "pipoca", "enfermeir", "8153", "24", "você se chama"):
            self.assertNotIn(forbidden, low)

    def test_defines_roles(self):
        self.assertIn("Maddie", self.text)
        self.assertIn("usuário", self.text)

    def test_explains_the_time_note(self):
        self.assertIn(NOTE_PREFIX, self.text)


if __name__ == "__main__":
    unittest.main()
