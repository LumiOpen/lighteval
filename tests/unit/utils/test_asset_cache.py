from __future__ import annotations

import hashlib
import importlib

import pytest

from lighteval.utils.asset_cache import ensure_cached_asset, lighteval_cache_dir


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_lighteval_cache_dir_uses_xdg_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert lighteval_cache_dir() == tmp_path / "xdg" / "huggingface" / "lighteval" / "assets"


def test_lighteval_cache_dir_uses_user_cache_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    assert lighteval_cache_dir() == tmp_path / ".cache" / "huggingface" / "lighteval" / "assets"


def test_ensure_cached_asset_publishes_verified_content(monkeypatch, tmp_path):
    content = b"verified asset"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "lighteval.utils.asset_cache.requests.get",
        lambda url, timeout: _Response(content),
    )

    path = ensure_cached_asset(relative_path="example.bin", url="https://example.test", sha256=digest)

    assert path == tmp_path / "huggingface" / "lighteval" / "assets" / "example.bin"
    assert path.read_bytes() == content
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_ensure_cached_asset_reuses_valid_file(monkeypatch, tmp_path):
    content = b"already cached"
    digest = hashlib.sha256(content).hexdigest()
    target = tmp_path / "huggingface" / "lighteval" / "assets" / "example.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def unexpected_download(*args, **kwargs):
        raise AssertionError("valid cached asset should not be downloaded again")

    monkeypatch.setattr("lighteval.utils.asset_cache.requests.get", unexpected_download)

    assert ensure_cached_asset(relative_path="example.bin", url="unused", sha256=digest) == target


def test_ensure_cached_asset_rejects_wrong_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "lighteval.utils.asset_cache.requests.get",
        lambda url, timeout: _Response(b"wrong content"),
    )

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        ensure_cached_asset(relative_path="example.bin", url="https://example.test", sha256="0" * 64)

    assert not (tmp_path / "huggingface" / "lighteval" / "assets" / "example.bin").exists()


def test_tiny_benchmarks_initialization_does_not_download(monkeypatch):
    pytest.importorskip("scipy")
    tiny_benchmarks = importlib.import_module("lighteval.tasks.tasks.tiny_benchmarks")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("task construction must not download runtime assets")

    monkeypatch.setattr(tiny_benchmarks.TinyCorpusAggregator, "download", unexpected_download)

    tiny_benchmarks.TinyCorpusAggregator("mmlu")
