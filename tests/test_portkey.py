import json

import dspy
import pytest

import tracepress as tp
from tracepress.model import _usage_of
from tracepress.providers import portkey_headers, register_portkey


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("PORTKEY_METADATA", "PORTKEY_CONFIG", "PORTKEY_VIRTUAL_KEY", "PORTKEY_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-test")


def test_headers_use_env_only_when_unset(monkeypatch):
    monkeypatch.setenv("PORTKEY_METADATA", '{"a": "env"}')
    monkeypatch.setenv("PORTKEY_CONFIG", "pc-env")
    h = portkey_headers()
    assert json.loads(h["x-portkey-metadata"]) == {"a": "env"} and h["x-portkey-config"] == "pc-env"
    assert portkey_headers(metadata={}, config="") == {}  # explicit empty turns the default off


@pytest.mark.parametrize("raw, match", [("{nope", "PORTKEY_METADATA is not valid JSON"), ("[1]", "must be a JSON object")])
def test_bad_metadata_env_names_the_variable(monkeypatch, raw, match):
    monkeypatch.setenv("PORTKEY_METADATA", raw)
    with pytest.raises(ValueError, match=match):
        portkey_headers()


def test_provider_id_depends_on_headers_and_base_url(monkeypatch):
    assert register_portkey() == "portkey"
    h = {"x-portkey-metadata": "{}"}
    a = register_portkey(h)
    assert a == register_portkey(h) != register_portkey({"x-portkey-metadata": "[]"})
    monkeypatch.setenv("PORTKEY_BASE_URL", "https://two.example/v1")
    assert register_portkey(h) != a


@pytest.mark.parametrize("name", ["portkey:@bedrock/claude-v2:0", "portkey/@bedrock/claude-v2:0"])
def test_model_routes_through_portkey_in_both_spellings(name):
    m = tp.Model(name, metadata={"project": "t"})
    assert m.lm.model.startswith("portkey-") and m.lm.model.endswith("/@bedrock/claude-v2:0")


def test_portkey_kwargs_rejected_elsewhere():
    with pytest.raises(ValueError):
        tp.Model("openrouter:openai/gpt-6-luna", metadata={"a": 1})


def test_price_fills_in_only_a_missing_cost():
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert _usage_of([{"usage": usage, "cost": None}], (1.0, 5.0)).cost_usd == 6.0
    assert _usage_of([{"usage": usage, "cost": 2.5}], (1.0, 5.0)).cost_usd == 2.5  # DSPy's cost wins
    assert _usage_of([{"usage": usage, "cost": None}]).cost_usd == 0.0


def test_headers_reach_the_gateway(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            sent = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((self.path, dict(self.headers), sent))
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "{\"a\": \"hi\"}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PORTKEY_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        m = tp.Model("portkey:@openai/gpt-x", metadata={"project": "t", "n": 3}, cache=False, price=(2.0, 0.0), progress=False)
        out = m.predict(dspy.Signature("q -> a"), [{"q": "hi"}])
        assert out[0].outputs == {"a": "hi"}
        _, headers, sent = seen[0]
        assert json.loads(headers["x-portkey-metadata"]) == {"n": "3", "project": "t"}
        assert headers["Authorization"] == "Bearer pk-test"
        assert seen[0][0].endswith("/chat/completions")
        assert sent["model"] == "@openai/gpt-x"  # not the internal portkey-<hash> id
        assert m.usage.cost_usd == 2.0
    finally:
        server.shutdown()
        server.server_close()
