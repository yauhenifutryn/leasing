from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.engine import should_rerank
from backend.settings import RerankerConfig


def test_should_rerank_voice_fast_disabled():
    cfg = RerankerConfig(
        enabled=True,
        model_name="x",
        device="cpu",
        batch_size=8,
        allow_no_rerank=False,
    )

    assert should_rerank(cfg, voice_fast=True) is False
    assert should_rerank(cfg, voice_fast=False) is True
