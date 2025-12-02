import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from utils import read_json, write_json


def synthesize_cluster(
    client: OpenAI,
    model: str,
    template: str,
    cluster: dict,
    temperature: float,
    max_attempts: int,
):
    """Try to synthesize a KB entry; fail-fast if parsing keeps failing."""
    content = ""
    for attempt in range(1, max_attempts + 1):
        suffix = ""
        if attempt > 1:
            suffix = "\n\nВажно: верни только валидный JSON без Markdown, без кавычек вне JSON и без лишних запятых."
        message = (
            f"{template}\n\nКластер:\n```json\n{json.dumps(cluster, ensure_ascii=False)}\n```{suffix}"
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": message}],
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_content(content)
        try:
            return json.loads(parsed), content
        except Exception:
            continue
    raise ValueError("Failed to parse model output after retries")


def parse_json_content(text: str) -> str:
    """Strip markdown code fences if present and return raw JSON string."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        # parts: ["", "json\n{...}", ""]
        if len(parts) >= 2:
            cleaned = parts[1]
        cleaned = cleaned.lstrip("json").lstrip("JSON").lstrip().strip()
    return cleaned


def log_bad_response(log_dir: Path, cluster_label: str, content: str) -> None:
    """Append the raw model response to a log file for debugging."""
    log_path = log_dir / "kb_errors.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n--- {cluster_label} ---\n{content}\n")


def select_temperature(model: str, fallback: float = 0.0) -> float:
    """Return a temperature value compatible with the selected model."""
    env_temp = os.getenv("OPENAI_TEMPERATURE")
    if env_temp:
        try:
            return float(env_temp)
        except Exception:
            pass
    if model and model.lower().startswith("gpt-5"):
        return 1.0
    return fallback


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--prompt", default="prompts/kb_entry_synthesis_ru.md")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.1"))
    args = parser.parse_args()

    client = OpenAI()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    template = Path(args.prompt).read_text(encoding="utf-8")
    max_attempts = int(os.getenv("KB_MAX_RETRIES", "12"))
    temperature = select_temperature(args.model)

    clusters = read_json(in_dir / "global_faq_clusters_dedup.json")
    knowledge_base = []
    for cluster in clusters:
        cluster_label = (
            cluster.get("canonical_q")
            or cluster.get("cluster_label")
            or "unknown_cluster"
        )
        try:
            kb_entry, content = synthesize_cluster(
                client=client,
                model=args.model,
                template=template,
                cluster=cluster,
                temperature=temperature,
                max_attempts=max_attempts,
            )
            knowledge_base.append(kb_entry)
        except Exception as exc:  # noqa: BLE001
            if "content" in locals() and content and content.strip():
                log_bad_response(out_dir, cluster_label, content)
            raise RuntimeError(
                f"KB synthesis failed for cluster '{cluster_label}' after {max_attempts} attempts: {exc}"
            ) from exc

    write_json(out_dir / "kb_faq_ru.json", knowledge_base)

    try:
        import yaml

        with open(out_dir / "kb_faq_ru.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(knowledge_base, handle, allow_unicode=True, sort_keys=False)
    except Exception:  # noqa: BLE001
        pass

if __name__ == "__main__":
    main()
