"""Testes dos módulos novos da 0.3: filtro de emoji, markdown, memória base e config."""

import json
import tempfile
import unittest
from pathlib import Path

from maddie import config, markdown_lite
from maddie.memory import ConversationMemory
from maddie.memory_store import FactStore, is_sensitive, validate_candidate
from maddie.textfilter import EmojiFilter, has_emoji, strip_emoji


class EmojiFilterTests(unittest.TestCase):
    def test_cases(self):
        cases = {
            "Boa ideia 😊. Vamos": "Boa ideia. Vamos",
            "Tudo bem! 😄": "Tudo bem!",
            "a 😊 b": "a b",
            "😊 Olá": "Olá",
            "Olá!\n\n🚀 Segunda linha": "Olá!\n\nSegunda linha",
            "Sem emoji, com acento: ação, 2 → 4, 50%": "Sem emoji, com acento: ação, 2 → 4, 50%",
            "Bandeira 🇧🇷 e pele 👍🏽 ok": "Bandeira e pele ok",
            "Coração ❤️ aqui": "Coração aqui",
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertEqual(strip_emoji(src), want)

    def test_streaming_equals_whole(self):
        text = "Legal 😀! Isso é ótimo 🎉🎉 mesmo.\nOutra linha ✨ aqui."
        whole = strip_emoji(text)
        for size in (1, 2, 3, 7):
            f = EmojiFilter()
            out = "".join(f.feed(text[i:i + size]) for i in range(0, len(text), size)) + f.flush()
            self.assertEqual(out.rstrip(), whole, f"pedaços de {size}")
        self.assertFalse(has_emoji(whole))

    def test_counts_removed(self):
        f = EmojiFilter()
        f.feed("a 😀 b 🎉")
        self.assertEqual(f.removed, 2)


class MarkdownTests(unittest.TestCase):
    def tags_of(self, text, fragment):
        return next(tags for t, tags in markdown_lite.parse(text) if fragment in t)

    def test_inline(self):
        text = "Isso é **importante**, *sutil* e `codigo`."
        self.assertIn("bold", self.tags_of(text, "importante"))
        self.assertIn("italic", self.tags_of(text, "sutil"))
        self.assertIn("code", self.tags_of(text, "codigo"))
        self.assertEqual(markdown_lite.plain(text), "Isso é importante, sutil e codigo.")

    def test_blocks(self):
        text = "# Título\n- item um\n- item **dois**\n1. passo\n```\nx = 1 * 2\n```\nfim"
        self.assertIn("heading", self.tags_of(text, "Título"))
        self.assertIn("list", self.tags_of(text, "item um"))
        self.assertIn("codeblock", self.tags_of(text, "x = 1 * 2"))
        self.assertNotIn("```", markdown_lite.plain(text))
        self.assertIn("•", markdown_lite.plain(text))

    def test_math_and_snake_case_are_not_italic(self):
        for text in ("4 * 6 = 24 e 2 * 3 = 6", "use a variável nome_do_arquivo_final aqui"):
            with self.subTest(text=text):
                self.assertEqual(markdown_lite.plain(text), text)
                self.assertTrue(all("italic" not in tags for _, tags in markdown_lite.parse(text)))

    def test_never_raises_and_keeps_text(self):
        for text in ("", "**sem fechar", "```\ncódigo sem fechar", "* ", "___", "a\n\n\nb"):
            with self.subTest(text=text):
                markdown_lite.parse(text)


class MemoryTwoPhaseTests(unittest.TestCase):
    def test_overflow_count_then_drop(self):
        mem = ConversationMemory(max_tokens=200, trim_ratio=0.5)
        for i in range(10):
            mem.add_exchange(f"{i} " + "x" * 60, "y" * 60, auto_trim=False)
        self.assertTrue(mem.over_budget())
        n = mem.overflow_count()
        self.assertGreater(n, 0)
        gone = mem.drop_oldest(n)
        self.assertEqual(len(gone), n)
        self.assertTrue(gone[0].user.startswith("0 "))
        self.assertFalse(mem.over_budget())
        self.assertEqual(mem.overflow_count(), 0)
        self.assertEqual(mem.dropped, n)

    def test_upto_and_summary_and_pop(self):
        mem = ConversationMemory()
        mem.add_exchange("a", "1")
        mem.add_exchange("b", "2")
        self.assertEqual(len(mem.build_messages("s", "fim", upto=1)), 4)
        base = mem.tokens()
        mem.summary = "resumo"
        self.assertGreater(mem.tokens(), base)
        self.assertEqual([m["role"] for m in mem.build_messages("s", "x")][:3], ["system", "user", "assistant"])
        self.assertEqual(mem.pop_last().user, "b")
        self.assertEqual(len(mem), 1)


class FactStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "memoria" / "base.json"

    def test_roundtrip_and_ids(self):
        st = FactStore(self.path)
        st.add("identidade", "O usuário se chama Rafael.", "Meu nome é Rafael.")
        st.add("pessoas", "O usuário tem uma cachorra chamada Pipoca.", "tenho uma cachorra")
        st.save()
        st2 = FactStore(self.path)
        st2.load()
        self.assertEqual([f["id"] for f in st2.facts], ["f-0001", "f-0002"])
        fact, action = st2.add("contexto", "O usuário estuda para uma prova.", "x")
        self.assertEqual((fact["id"], action), ("f-0003", "novo"))

    def test_duplicates_update_instead_of_adding(self):
        st = FactStore(self.path)
        st.add("identidade", "O usuário trabalha como enfermeira.", "x")
        _, same = st.add("identidade", "O usuário trabalha como enfermeira.", "x")
        _, upd = st.add("identidade", "O usuário trabalha como enfermeira num hospital.", "x")
        self.assertEqual((same, upd, len(st.facts)), ("igual", "atualizado", 1))

    def test_replace_by_id_and_limit(self):
        st = FactStore(self.path, max_facts=2)
        f, _ = st.add("identidade", "O usuário é enfermeira.", "x")
        _, action = st.add("identidade", "O usuário agora é professora de música.", "y", substitui_id=f["id"])
        self.assertEqual((action, len(st.facts)), ("atualizado", 1))
        st.add("pessoas", "O usuário tem um irmão chamado Pedro.", "x")
        self.assertEqual(st.add("preferencias", "O usuário gosta de ficção científica.", "x")[1], "cheio")

    def test_corrupt_file_is_not_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ quebrado", encoding="utf-8")
        st = FactStore(self.path)
        st.load()
        self.assertTrue(st.load_error)
        with self.assertRaises(OSError):
            st.save()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{ quebrado")

    def test_prompt_block(self):
        st = FactStore(self.path)
        self.assertEqual(st.prompt_block(), "")
        st.add("identidade", "O usuário se chama Rafael.", "x")
        block = st.prompt_block()
        self.assertIn("- O usuário se chama Rafael.", block)
        self.assertIn("sem anunciar", block)

    def test_validation(self):
        user = ["Meu nome é Rafael e meu irmão se chama Pedro.", "Qual é a capital do Japão?",
                "Minha senha do banco é 1234."]
        ok = {"categoria": "identidade", "fato": "O usuário se chama Rafael.", "trecho_fonte": "meu nome é rafael"}
        self.assertEqual(validate_candidate(ok, user), (True, ""))
        bad = [
            dict(ok, trecho_fonte="Eu me chamo Rafael"),                 # fonte não literal
            dict(ok, categoria="saude"),                                  # categoria fora da lista
            dict(ok, fato="x" * 201),                                     # longo demais
            dict(ok, fato="A senha do banco do usuário é 1234.", trecho_fonte="Minha senha do banco é 1234."),
            dict(ok, fato="O usuário perguntou a capital.", trecho_fonte="Qual é a capital do Japão?"),
            dict(ok, fato="A Maddie é legal.", trecho_fonte="Meu nome é Rafael"),
            dict(ok, fato="O usuário gosta de conversar com Maddie.", trecho_fonte="Meu nome é Rafael"),
            dict(ok, trecho_fonte=""),
            "texto solto",
        ]
        for cand in bad:
            with self.subTest(cand=cand):
                self.assertFalse(validate_candidate(cand, user)[0])

    def test_source_from_assistant_is_rejected(self):
        """O trecho precisa estar numa mensagem do USUÁRIO (a lista só tem as dele)."""
        cand = {"categoria": "identidade", "fato": "O usuário se chama Pedro.",
                "trecho_fonte": "Prazer, Pedro!"}
        self.assertFalse(validate_candidate(cand, ["oi"])[0])

    def test_sensitive(self):
        for text in ("meu CPF é 123.456.789-09", "tomo remédio para pressão", "meu salário é baixo",
                     "fale comigo em fulano@exemplo.com", "meu cartão 4111 1111 1111 1111"):
            with self.subTest(text=text):
                self.assertTrue(is_sensitive(text))
        for text in ("O usuário se chama Rafael.", "O usuário gosta de correr aos sábados."):
            with self.subTest(text=text):
                self.assertFalse(is_sensitive(text))


class ConfigV03Tests(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            cfg = config.replace(config.Config(), temperature=0.3, num_thread=6, ui_theme="claro",
                                 base_memory_enabled=True)
            config.save(p, cfg)
            self.assertEqual(config.load(p), cfg)
            self.assertNotIn("seed", json.loads(p.read_text(encoding="utf-8"))["model"])

    def test_old_v02_config_still_loads(self):
        old = {"ollama": {"host": "127.0.0.1", "port": 11434}, "model": {"name": "qwen3:0.6b",
               "num_ctx": 8192, "num_predict": 768, "think": False, "keep_alive": "10m"},
               "memory": {"max_tokens": 5500, "trim_ratio": 0.6}, "persona_file": "persona.md"}
        cfg = config.from_dict(old)
        self.assertEqual((cfg.model, cfg.keep_alive, cfg.base_memory_enabled), ("qwen3:0.6b", "10m", False))

    def test_optional_options_only_sent_when_set(self):
        self.assertNotIn("num_thread", config.Config().model_options())
        opts = config.replace(config.Config(), num_thread=6, repeat_penalty=1.1).model_options()
        self.assertEqual((opts["num_thread"], opts["repeat_penalty"]), (6, 1.1))

    def test_profiles_and_validation(self):
        self.assertEqual(config.Config().profile_name(), "Equilibrado")
        self.assertEqual(config.replace(config.Config(), **config.PROFILES["Criativo"]).profile_name(), "Criativo")
        self.assertEqual(config.replace(config.Config(), temperature=0.55).profile_name(), "Personalizado")
        for bad in ({"temperature": 3}, {"top_p": 0}, {"ui_theme": "rosa"}, {"ui_font_size": 40},
                    {"num_thread": 0}, {"base_memory_file": "..\\fora.json"}):
            with self.subTest(bad=bad), self.assertRaises(config.ConfigError):
                config.replace(config.Config(), **bad)

    def test_base_memory_is_off_by_default(self):
        self.assertFalse(config.Config().base_memory_enabled)
        self.assertFalse(config.from_dict({}).base_memory_enabled)

    def test_load_signature_changes_only_for_reload_fields(self):
        base = config.Config()
        self.assertEqual(base.load_signature(), config.replace(base, temperature=0.2).load_signature())
        self.assertNotEqual(base.load_signature(), config.replace(base, num_ctx=4096, memory_max_tokens=2000,
                                                                   num_predict=512).load_signature())


if __name__ == "__main__":
    unittest.main()
