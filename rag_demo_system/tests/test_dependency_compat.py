import huggingface_hub


def test_hf_hub_download_exists() -> None:
    assert hasattr(huggingface_hub, "hf_hub_download")


def test_sentence_transformers_imports() -> None:
    from sentence_transformers import SentenceTransformer

    assert SentenceTransformer is not None
