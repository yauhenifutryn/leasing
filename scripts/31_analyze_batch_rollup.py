import argparse
import hashlib
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from utils import list_files, read_json, write_json


def select_temperature(model: str, fallback: float = 0.0) -> float:
    """Return a temperature value compatible with the requested model."""
    if model and model.lower().startswith("gpt-5"):
        return 1.0
    return fallback


def chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--prompt", default="prompts/batch_rollup_ru.md")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.1"))
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    client = OpenAI()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt = Path(args.prompt).read_text(encoding="utf-8")

    files = list_files(in_dir, ".json")
    for group in tqdm(chunks(files, args.batch_size), desc="Batch rollups"):
        payload = [read_json(path) for path in group]
        ids = ",".join([item.get("conversation_id", "") for item in payload])
        batch_id = hashlib.md5(ids.encode()).hexdigest()[:12]
        message = f"{sys_prompt}\n\nВходные карточки:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=select_temperature(args.model),
                messages=[{"role": "user", "content": message}],
            )
            data = json.loads(response.choices[0].message.content)
            data["batch_id"] = batch_id
            write_json(out_dir / f"batch_{batch_id}.json", data)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] batch {batch_id}: {exc}")

if __name__ == "__main__":
    main()
