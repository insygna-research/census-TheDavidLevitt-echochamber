"""GUI setup-tab helpers: key masking, saving, and status overview."""

import os
import stat

import pytest

from echochamber import ui


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(ui, "ENV_FILE", path)
    for key in ui.PROVIDER_ENV_KEYS.values():
        monkeypatch.delenv(key, raising=False)
    return path


def test_mask_hides_key_material():
    masked = ui._mask("sk-ant-api03-averylongsecretkeyvalue1234")
    assert "averylongsecret" not in masked
    assert masked.startswith("set (sk-ant-")
    assert ui._mask("") == "not set"


def test_save_keys_writes_env_with_restrictive_perms(env_file):
    overview, status = ui.save_keys("sk-ant-test123", "", "", "AIza-test456")

    content = env_file.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-test123" in content
    assert "GEMINI_API_KEY=AIza-test456" in content
    assert "OPENAI_API_KEY" not in content  # empty fields untouched
    mode = stat.S_IMODE(env_file.stat().st_mode)
    assert mode == 0o600
    assert "✅ Saved" in status
    # Live environment updated without restart
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test123"


def test_save_keys_merges_existing_lines(env_file):
    env_file.write_text("# comment\nANTHROPIC_API_KEY=old-value\nOTHER=keepme\n")
    ui.save_keys("new-value", "", "", "")

    lines = env_file.read_text().splitlines()
    assert "ANTHROPIC_API_KEY=new-value" in lines
    assert "OTHER=keepme" in lines
    assert "# comment" in lines
    assert "old-value" not in env_file.read_text()


def test_save_keys_all_empty_is_noop(env_file):
    _, status = ui.save_keys("", "", "", "")
    assert "Nothing to save" in status
    assert not env_file.exists()


def test_overview_reports_masked_status(env_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-somethinglong12345")
    md = ui.key_overview_md()
    assert "not set" in md            # untouched providers
    assert "sk-ant-" in md            # masked prefix shown
    assert "somethinglong" not in md  # secret body hidden


def test_key_test_requires_a_key(env_file):
    assert ui.test_provider_key("anthropic", "").startswith("❌")
