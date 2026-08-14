import pytest

from src.llms.huggingface import load_pretrained


class CachedFactory:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        return "cached-model"


class MissingCacheFactory:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        if kwargs.get("local_files_only"):
            raise OSError("not cached")
        return "downloaded-model"


@pytest.fixture(autouse=True)
def clear_calls():
    CachedFactory.calls.clear()
    MissingCacheFactory.calls.clear()


def test_load_pretrained_uses_local_cache_without_network_request():
    loaded = load_pretrained(CachedFactory, "example/model", revision="main")

    assert loaded == "cached-model"
    assert CachedFactory.calls == [(
        "example/model", {"revision": "main", "local_files_only": True}
    )]


def test_load_pretrained_downloads_only_when_local_cache_is_missing():
    loaded = load_pretrained(MissingCacheFactory, "example/model")

    assert loaded == "downloaded-model"
    assert MissingCacheFactory.calls == [
        ("example/model", {"local_files_only": True}),
        ("example/model", {}),
    ]
