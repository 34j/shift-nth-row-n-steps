from typer.testing import CliRunner

from shift_nth_row_n_steps.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_banchmark():
    result = runner.invoke(app, ["--n-end", "2"])
    assert result.exit_code == 0
