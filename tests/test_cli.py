"""CLI tests that don't need a model: argument handling only.

Tests that actually run a search need a fake embedder in place of ClipEmbedder;
they come together with the FakeEmbedder in conftest.py.
"""

from pathlib import Path

from typer.testing import CliRunner

from wildsearch.cli import app

runner = CliRunner()


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_search_rejects_n_zero() -> None:
    result = runner.invoke(app, ["search", "lion", "-n", "0"])
    assert result.exit_code == 2


def test_search_rejects_missing_folder(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "lion", "-p", str(tmp_path / "nope")])
    assert result.exit_code == 2


def test_search_rejects_a_file_as_folder(tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    photo.touch()
    result = runner.invoke(app, ["search", "lion", "-p", str(photo)])
    assert result.exit_code == 2
