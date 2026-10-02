"""Tests for command-line interface."""

from monte_carlo.cli import build_parser, main


def test_cli_parser_defaults():
    parser = build_parser()
    opts = parser.parse_args([])
    assert opts.scenario == "manufacturing"
    assert opts.simulations == 25000
    assert opts.seed == 42
    assert opts.save_plots is False


def test_cli_execution_smoke(monkeypatch, tmp_path):
    # Run a quick 500-iteration manufacturing run with save-plots to tmp_path
    exit_code = main([
        "--scenario", "manufacturing",
        "-n", "500",
        "--save-plots",
        "--output-dir", str(tmp_path),
    ])
    assert exit_code == 0
    assert (tmp_path / "manufacturing_dashboard.png").exists()
