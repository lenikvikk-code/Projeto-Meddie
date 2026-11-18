"""Testes do cliente contra um servidor Ollama falso em 127.0.0.1 (não precisa do Ollama real)."""

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from maddie import config
from maddie.ollama_client import (
    ChatStats, ModelNotFound, OllamaClient, OllamaError, OllamaUnavailable, ThinkFilter,
)


class FakeOllama(BaseHTTPRequestHandler):
    chunks = ["Olá", ", eu sou ", "a Maddie."]
    models = ["qwen3:0.6b"]
    chat_status = 200
    last_chat_body = None
    last_generate_body = None

    def log_message(self, *args):
        pass

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            self._json(200, {"version": "0.0-fake"})
        elif self.path == "/api/tags":
            self._json(200, {"models": [{"name": n, "model": n} for n in self.models]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":
            type(self).last_generate_body = body
            self._json(200, {"done": True})
            return
        type(self).last_chat_body = body
        if self.chat_status != 200:
            self._json(self.chat_status, {"error": f"model '{body['model']}' not found"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        lines = [{"message": {"role": "assistant", "content": c}, "done": False} for c in self.chunks]
        lines.append({"message": {"role": "assistant", "content": ""}, "done": True,
                      "done_reason": "stop", "prompt_eval_count": 42, "eval_count": 7})
        for obj in lines:
            data = (json.dumps(obj) + "\n").encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeOllama.chat_status = 200
        FakeOllama.models = ["qwen3.5:4b"]
        self.cfg = config.from_dict({"ollama": {"port": self.port}})
        self.client = OllamaClient(self.cfg)

    def test_ready_and_stream(self):
        self.client.ensure_ready()
        stats = ChatStats()
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "oi"}]
        text = "".join(self.client.chat_stream(msgs, stats=stats))
        self.assertEqual(text, "Olá, eu sou a Maddie.")
        self.assertEqual((stats.prompt_tokens, stats.output_tokens, stats.done_reason), (42, 7, "stop"))

    def test_request_carries_configured_parameters(self):
        list(self.client.chat_stream([{"role": "user", "content": "oi"}]))
        body = FakeOllama.last_chat_body
        self.assertIs(body["think"], False)
        self.assertTrue(body["stream"])
        self.assertEqual(body["options"]["num_ctx"], 8192)
        self.assertEqual(body["options"]["num_predict"], 768)
        self.assertNotIn("seed", body["options"])  # app usa amostragem aleatória

    def test_load_model_does_not_inherit_unlimited_generation(self):
        self.client.cfg = config.replace(self.cfg, num_predict=-1)
        self.client.load_model()
        self.assertEqual(FakeOllama.last_generate_body["options"]["num_predict"], 1)

    def test_model_missing(self):
        FakeOllama.models = ["outro:1b"]
        with self.assertRaises(ModelNotFound):
            self.client.ensure_ready()

    def test_chat_404_is_model_not_found(self):
        FakeOllama.chat_status = 404
        with self.assertRaises(ModelNotFound):
            list(self.client.chat_stream([{"role": "user", "content": "oi"}]))

    def test_chat_500_is_readable_error(self):
        FakeOllama.chat_status = 500
        with self.assertRaises(OllamaError) as ctx:
            list(self.client.chat_stream([{"role": "user", "content": "oi"}]))
        self.assertIn("not found", str(ctx.exception))

    def test_server_down_is_unavailable(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
        s.close()
        client = OllamaClient(config.from_dict({"ollama": {"port": free_port, "connect_timeout_s": 1}}))
        with self.assertRaises(OllamaUnavailable):
            client.ensure_ready()

    def test_unload_never_raises_when_server_down(self):
        client = OllamaClient(config.from_dict({"ollama": {"port": 1, "connect_timeout_s": 1}}))
        client.unload_model(timeout=1)


class ThinkFilterTests(unittest.TestCase):
    def test_removes_think_block_split_across_chunks(self):
        f = ThinkFilter()
        out = "".join(f.feed(c) for c in ["<th", "ink>raciocínio", " oculto</thi", "nk>\n\nResposta", " final"])
        out += f.flush()
        self.assertEqual(out.strip(), "Resposta final")

    def test_passes_normal_text(self):
        f = ThinkFilter()
        out = f.feed("2 < 3 e <b>ok</b>") + f.flush()
        self.assertEqual(out, "2 < 3 e <b>ok</b>")


if __name__ == "__main__":
    unittest.main()
