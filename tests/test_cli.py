from dataset_label.cli import main


def test_main_prints_help_and_returns_zero(capsys):
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "dataset-label" in output
    assert "--test" in output
    assert "--preview" in output
