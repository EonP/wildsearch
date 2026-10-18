import pytest
from typer.testing import CliRunner

from wildsearch.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "status"):
        assert command in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["index", "."],
        ["search", "lion resting under a tree"],
        ["search", "sunset", "-n", "5", "-p", "."],
        ["status"],
    ],
)
def test_stubs_exit_cleanly(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "not implemented" in result.output


def test_search_requires_query():
    result = runner.invoke(app, ["search"])
    assert result.exit_code != 0
