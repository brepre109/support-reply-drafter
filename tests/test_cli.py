import json

from reply_drafter.cli import main


def test_offline_draft_for_dataset_conversation(capsys):
    assert main(["draft", "--provider", "offline", "--conversation", "c29", "--env-file", "none"]) == 0
    out = capsys.readouterr().out
    assert "lookup_orders" in out and "1ZEXAMPLE0000001047" in out and "STAFF:" in out


def test_offline_draft_json_for_new_message(capsys):
    main(["draft", "--provider", "offline", "--message", "Do you ship to Canada?", "--name", "Ana", "--json",
          "--env-file", "none"])
    d = json.loads(capsys.readouterr().out)
    assert d["reply"].startswith("Hi Ana") and "Canada" in d["reply"]


def test_offline_eval_and_report(tmp_path, capsys):
    assert main(["eval", "--provider", "offline", "--judge", "heuristic", "--limit", "5", "--seed", "1",
                 "--workers", "1", "--out", str(tmp_path), "--env-file", "none"]) == 0
    (run_dir,) = tmp_path.iterdir()
    assert (run_dir / "report.html").exists()
    assert "Cases: 5" in capsys.readouterr().out
    assert main(["report", str(run_dir)]) == 0


def test_missing_credentials_message(monkeypatch):
    import pytest

    for k in ("DRAFTER_BASE_URL", "DRAFTER_API_KEY", "DRAFTER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit, match="--provider offline"):
        main(["draft", "--message", "hi", "--env-file", "none"])
