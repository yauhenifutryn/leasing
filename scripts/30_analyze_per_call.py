import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from utils import chunked, list_files, read_json, write_json, should_skip


def select_temperature(model: str, fallback: float = 0.0) -> float:
    """Return a temperature value that complies with model constraints."""
    if model and model.lower().startswith("gpt-5"):
        return 1.0
    return fallback

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--prompt", default="prompts/per_call_analysis_ru.md")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.1"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    client = OpenAI()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys_prompt = Path(args.prompt).read_text(encoding="utf-8")

    files = list_files(in_dir, ".json")
    progress = tqdm(total=len(files), desc="Per-call analysis")
    for batch in chunked(files, args.batch_size):
        for file_path in batch:
            convo = read_json(file_path)
            output_path = out_dir / f"{convo['conversation_id']}.json"
            if should_skip(output_path, args.overwrite):
                progress.update(1)
                continue
            message = f"{sys_prompt}\n\nДанные разговора:\n```json\n{json.dumps(convo, ensure_ascii=False)}\n```"
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    temperature=select_temperature(args.model),
                    messages=[{"role": "user", "content": message}],
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                write_json(output_path, data)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {file_path.name}: {exc}")
            finally:
                progress.update(1)
    progress.close()

if __name__ == "__main__":
    main()
