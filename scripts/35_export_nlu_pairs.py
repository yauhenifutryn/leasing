import argparse
import json
from pathlib import Path

from utils import list_files, read_json


def normalize_hashtags(intent: str | None, subtopics: list[str] | None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()

    for value in [intent, *(subtopics or [])]:
        if not value:
            continue
        tag = value.strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def export_pairs(in_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[str] = []

    for file_path in list_files(in_dir, ".json"):
        payload = read_json(file_path)
        conversation_id = payload.get("conversation_id", file_path.stem)
        verbatim_pairs = payload.get("verbatim_QA_pairs") or []
        hashtags = normalize_hashtags(payload.get("client_intent"), payload.get("subtopics"))
        quality_flags = payload.get("quality_flags") or []

        for index, pair in enumerate(verbatim_pairs, start=1):
            record = {
                "call_id": conversation_id,
                "pair_index": index,
                "question": pair.get("q", "").strip(),
                "answer": pair.get("a", "").strip(),
                "question_speaker": pair.get("question_speaker", "client"),
                "answer_speaker": pair.get("answer_speaker", "agent"),
                "intent": payload.get("client_intent"),
                "hashtags": hashtags,
                "quality_flags": quality_flags,
                "source_file": str(file_path),
                "needs_review": False,
                "review_notes": "",
            }
            # Skip empty rows that can appear if the LLM failed to extract a QA pair.
            if not record["question"] and not record["answer"]:
                continue
            records.append(record)

    # Write newline-delimited JSON to simplify downstream processing.
    with open(out_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", default="insights_per_call")
    parser.add_argument(
        "--out",
        dest="out_path",
        default="nlu_output/nlu_pairs.jsonl",
        help="Path to the newline-delimited JSON file to create.",
    )
    args = parser.parse_args()

    export_pairs(Path(args.in_dir), Path(args.out_path))


if __name__ == "__main__":
    main()
