"""Teste de ponta a ponta com o Ollama real. Só roda com MADDIE_TEST_REAL=1.

Uso:  set MADDIE_TEST_REAL=1 && python -m unittest tests.test_real_ollama -v
"""

import os
import unittest
from pathlib import Path

from maddie import config, personality
from maddie.engine import ChatSession

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(os.environ.get("MADDIE_TEST_REAL") == "1", "defina MADDIE_TEST_REAL=1 para rodar")
class RealOllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = config.replace(config.load(ROOT / "config.json"), base_memory_enabled=False)
        cls.persona = personality.load_persona(ROOT / cfg.persona_file)
        cls.session = ChatSession(cfg, cls.persona, ROOT)
        cls.session.client.ensure_ready()

    def ask(self, text):
        reply = "".join(self.session.reply(text))
        return reply, self.session.last

    def test_warm_up_answers_and_keeps_context(self):
        s = self.session
        s.warm_up()
        self.assertEqual(len(s.memory), 0)  # a abertura fica oculta
        reply, info = self.ask("Guarde este código: 4729. Responda apenas 'ok'.")
        print(f"\n[1ª mensagem] 1º token em {info.ttft_s:.1f}s; resposta: {reply.strip()!r}")
        self.assertTrue(reply.strip())
        self.assertNotIn("<think>", reply)
        self.assertGreater(info.stats.output_tokens, 0)
        # Com a abertura no cache, a 1ª mensagem não pode reprocessar a persona inteira.
        self.assertLess(info.stats.prompt_eval_s, 8.0, "o aquecimento não foi reaproveitado")
        reply, _ = self.ask("Qual foi o código que eu pedi para você guardar? Responda só o número.")
        print(f"[contexto] resposta: {reply.strip()!r}")
        self.assertIn("4729", reply)


if __name__ == "__main__":
    unittest.main()
