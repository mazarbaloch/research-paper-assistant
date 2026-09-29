import pytest

from config import AppError, get_settings


def test_invalid_numeric_environment_is_actionable(monkeypatch):
    monkeypatch.setenv("TOP_K", "not a number")
    with pytest.raises(AppError, match="numeric configuration"):
        get_settings()


def test_environment_read_on_each_settings_call(monkeypatch):
    monkeypatch.setenv("TOP_K", "7")
    assert get_settings().top_k == 7


def test_cloud_endpoint_rejected(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")
    with pytest.raises(AppError, match="local HTTP address"):
        get_settings()
