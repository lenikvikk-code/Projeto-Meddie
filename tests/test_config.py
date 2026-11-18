import tempfile
import unittest
from pathlib import Path

from maddie import config, personality

ROOT = Path(__file__).resolve().parent.parent


class ConfigTests(unittest.TestCase):
    def test_project_config_loads_with_required_values(self):
        cfg = config.load(ROOT / "config.json")
        self.assertEqual(cfg.host, "127.0.0.1")   # só loopback
        self.assertFalse(cfg.think)
        self.assertNotIn("cloud", cfg.model)
        d = config.Config()                       # padrões da 0.3
        self.assertEqual((d.model, d.num_ctx, d.num_predict, d.memory_max_tokens),
                         ("qwen3.5:4b", 8192, 768, 5500))

    def test_defaults_when_fields_missing(self):
        cfg = config.from_dict({})
        self.assertEqual(cfg.model, "qwen3.5:4b")
        self.assertEqual(cfg, config.Config())
        # O config.json do projeto guarda as preferências do usuário (tema, memória permanente,
        # perfil...), então não precisa ser igual ao padrão: só precisa carregar e ser válido.
        config.load(ROOT / "config.json")

    def test_memory_budget_must_fit_context(self):
        with self.assertRaises(config.ConfigError):
            config.from_dict({"model": {"num_ctx": 4096, "num_predict": 512}, "memory": {"max_tokens": 3600}})
        with self.assertRaises(config.ConfigError):
            config.from_dict({"memory": {"trim_ratio": 0}})
        unlimited = config.from_dict({"model": {"num_ctx": 4096, "num_predict": 512},
                                      "memory": {"max_tokens": 3600, "unlimited": True}})
        self.assertTrue(unlimited.memory_unlimited)
        no_response_limit = config.from_dict({"model": {"num_predict": -1}})
        self.assertEqual(no_response_limit.num_predict, -1)
        self.assertEqual(no_response_limit.model_options()["num_predict"], -1)
        with self.assertRaises(config.ConfigError):
            config.from_dict({"model": {"num_predict": -2}})

    def test_project_paths_accept_windows_separators_and_reject_traversal(self):
        cfg = config.from_dict({"base_memory": {"file": r"memoria\memoria_base.json"}})
        self.assertEqual(config.project_path(ROOT, cfg.base_memory_file),
                         ROOT / "memoria" / "memoria_base.json")
        for field in ({"base_memory": {"file": r"..\fora.json"}},
                      {"persona_file": r"..\fora.md"},
                      {"base_memory": {"file": r"C:\fora.json"}}):
            with self.subTest(field=field), self.assertRaises(config.ConfigError):
                config.from_dict(field)

    def test_rejects_non_loopback_host(self):
        for host in ("0.0.0.0", "192.168.0.10", "api.exemplo.com"):
            with self.subTest(host=host), self.assertRaises(config.ConfigError):
                config.from_dict({"ollama": {"host": host}})

    def test_rejects_cloud_models(self):
        for name in ("gpt-oss:120b-cloud", "qwen3-coder:480b-cloud", "algum:cloud"):
            with self.subTest(name=name), self.assertRaises(config.ConfigError):
                config.from_dict({"model": {"name": name}})

    def test_rejects_invalid_numbers(self):
        with self.assertRaises(config.ConfigError):
            config.from_dict({"model": {"num_ctx": "abc"}})
        with self.assertRaises(config.ConfigError):
            config.from_dict({"model": {"num_ctx": 512, "num_predict": 512}})
        with self.assertRaises(config.ConfigError):
            config.from_dict({"memory": {"max_tokens": 0}})

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            p.write_text("{ isto não é json", encoding="utf-8")
            with self.assertRaises(config.ConfigError):
                config.load(p)


class PersonalityTests(unittest.TestCase):
    def test_project_persona_defines_identity(self):
        text = personality.load_persona(ROOT / "persona.md")
        self.assertIn("Maddie", text)
        self.assertIn("português brasileiro", text)

    def test_missing_or_empty_persona(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(personality.PersonalityError):
                personality.load_persona(Path(d) / "nao_existe.md")
            empty = Path(d) / "vazio.md"
            empty.write_text("  \n", encoding="utf-8")
            with self.assertRaises(personality.PersonalityError):
                personality.load_persona(empty)


if __name__ == "__main__":
    unittest.main()
