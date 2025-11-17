import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from utils import read_json, write_json


def select_temperature(model: str, fallback: float = 0.0) -> float:
    """Return a temperature value compatible with the selected model."""
    if model and model.lower().startswith("gpt-5"):
        return 1.0
    return fallback


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--prompt", default="prompts/kb_entry_synthesis_ru.md")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    client = OpenAI()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = Path(args.prompt).read_text(encoding="utf-8")

    clusters = read_json(in_dir / "global_faq_clusters_dedup.json")
    knowledge_base = []
    for cluster in clusters:
        message = f"{template}\n\nКластер:\n```json\n{json.dumps(cluster, ensure_ascii=False)}\n```"
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=select_temperature(args.model),
                messages=[{"role": "user", "content": message}],
            )
            knowledge_base.append(json.loads(response.choices[0].message.content))
        except Exception as exc:  # noqa: BLE001
            print("[ERROR] KB synthesis:", exc)

    write_json(out_dir / "kb_faq_ru.json", knowledge_base)

    try:
        import yaml

        with open(out_dir / "kb_faq_ru.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(knowledge_base, handle, allow_unicode=True, sort_keys=False)
    except Exception:  # noqa: BLE001
        pass

if __name__ == "__main__":
    main()
