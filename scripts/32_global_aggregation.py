import argparse
from pathlib import Path

from utils import list_files, read_json, write_json

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [read_json(path) for path in list_files(args.in_dir, ".json")]

    top_intent_counts = {}
    faq_clusters = []
    for batch in batches:
        for intent in batch.get("top_intents", []):
            name = intent["intent"]
            top_intent_counts[name] = top_intent_counts.get(name, 0) + intent["count"]
        faq_clusters.extend(batch.get("faq_clusters", []))

    write_json(
        out_dir / "global_top_intents.json",
        sorted(
            [{"intent": name, "count": count} for name, count in top_intent_counts.items()],
            key=lambda item: item["count"],
            reverse=True,
        ),
    )
    write_json(out_dir / "global_faq_clusters_raw.json", faq_clusters)

if __name__ == "__main__":
    main()
