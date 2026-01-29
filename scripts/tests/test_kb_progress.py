from pathlib import Path

from scripts.kb_progress import maybe_log_progress, maybe_write_checkpoint


def test_maybe_write_checkpoint_writes(tmp_path):
    out_path = tmp_path / "kb_faq_ru.json"
    partial_path = tmp_path / "kb_faq_ru.partial.json"
    data = [{"a": 1}]

    wrote = maybe_write_checkpoint(
        index=2,
        total=10,
        every=2,
        out_path=out_path,
        partial_path=partial_path,
        data=data,
    )

    assert wrote is True
    assert partial_path.exists()
    assert partial_path.read_text(encoding="utf-8").strip().startswith("[")


def test_maybe_log_progress_fires_at_interval(capsys):
    fired = maybe_log_progress(index=4, total=10, every=2)
    captured = capsys.readouterr()
    assert fired is True
    assert "KB progress" in captured.out


def test_maybe_log_progress_skips_when_not_due(capsys):
    fired = maybe_log_progress(index=3, total=10, every=2)
    captured = capsys.readouterr()
    assert fired is False
    assert captured.out == ""
