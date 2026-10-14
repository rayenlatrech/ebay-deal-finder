"""The Vercel webhook rejects anything without Telegram's secret token."""

import importlib.util
import json
import pathlib
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    spec = importlib.util.spec_from_file_location("telegram_webhook", ROOT / "api" / "telegram.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processed = []
    monkeypatch.setattr(module, "_process", lambda update: processed.append(update))
    httpd = HTTPServer(("127.0.0.1", 0), module.handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/api/telegram", processed
    httpd.shutdown()


def _post(url, body, secret=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    if secret is not None:
        req.add_header("X-Telegram-Bot-Api-Secret-Token", secret)
    try:
        return urllib.request.urlopen(req, timeout=5).status
    except urllib.error.HTTPError as e:
        return e.code


def test_missing_or_wrong_secret_is_rejected(server):
    url, processed = server
    assert _post(url, {"message": {}}) == 401
    assert _post(url, {"message": {}}, secret="wrong") == 401
    assert processed == []


def test_valid_secret_is_processed(server):
    url, processed = server
    assert _post(url, {"update_id": 7}, secret="s3cret") == 200
    assert processed == [{"update_id": 7}]


def test_health_check(server):
    url, _ = server
    assert urllib.request.urlopen(url, timeout=5).status == 200
