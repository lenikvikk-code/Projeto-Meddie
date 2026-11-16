import unittest

from maddie.memory import ConversationMemory, estimate_tokens

SYSTEM = "Você é a Maddie."


class MemoryTests(unittest.TestCase):
    def test_keeps_everything_under_budget(self):
        mem = ConversationMemory(max_tokens=1000)
        for i in range(10):
            mem.add_exchange(f"u{i}", f"a{i}")
        self.assertEqual(len(mem), 10)
        self.assertEqual(mem.dropped, 0)

    def test_trims_oldest_in_batch_when_over_budget(self):
        mem = ConversationMemory(max_tokens=200, trim_ratio=0.5)
        text = "x" * 60  # cerca de 24 tokens por mensagem
        for i in range(10):
            mem.add_exchange(f"{i} {text}", text)
        self.assertLessEqual(mem.tokens(), 200)
        self.assertGreater(mem.dropped, 0)
        self.assertTrue(mem.exchanges()[-1][0].startswith("9 "))  # a mais recente fica
        self.assertTrue(mem.exchanges()[0][0][0] != "0")          # a mais antiga saiu

    def test_batch_trim_keeps_prefix_stable_for_several_turns(self):
        """Depois de um corte, as próximas trocas não podem mexer no começo do histórico."""
        mem = ConversationMemory(max_tokens=300, trim_ratio=0.5)
        text = "y" * 60
        i = 0
        while mem.dropped == 0:
            mem.add_exchange(f"{i} {text}", text)
            i += 1
        first = mem.exchanges()[0]
        mem.add_exchange(f"{i} {text}", text)
        mem.add_exchange(f"{i + 1} {text}", text)
        self.assertEqual(mem.exchanges()[0], first)

    def test_single_huge_exchange_is_kept(self):
        mem = ConversationMemory(max_tokens=10)
        mem.add_exchange("a" * 500, "b" * 500)
        self.assertEqual(len(mem), 1)

    def test_build_messages_order_and_system_first(self):
        mem = ConversationMemory()
        mem.add_exchange("oi", "olá")
        msgs = mem.build_messages(SYSTEM, "tudo bem?")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[0]["content"], SYSTEM)
        self.assertEqual(msgs[-1]["content"], "tudo bem?")

    def test_clear_keeps_identity(self):
        mem = ConversationMemory()
        mem.add_exchange("meu nome é Lenik", "prazer")
        mem.clear()
        msgs = mem.build_messages(SYSTEM, "qual meu nome?")
        self.assertEqual(len(mem), 0)
        self.assertEqual(mem.dropped, 0)
        self.assertEqual(msgs, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "qual meu nome?"},
        ])

    def test_estimate_is_conservative(self):
        self.assertGreaterEqual(estimate_tokens("a" * 300), 100)

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            ConversationMemory(0)
        with self.assertRaises(ValueError):
            ConversationMemory(100, trim_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
