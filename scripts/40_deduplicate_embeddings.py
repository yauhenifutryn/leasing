import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from utils import read_json, write_json

# Clusters similar questions to avoid duplicates; multilingual model works well for RU.

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.28, help="Agglomerative clustering distance threshold")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faq_raw = read_json(in_dir / "global_faq_clusters_raw.json")
    questions = [item["canonical_q"] for item in faq_raw if item.get("canonical_q")]

    if not questions:
        write_json(out_dir / "global_faq_clusters_dedup.json", [])
        return

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)

    distance_matrix = 1 - np.dot(embeddings, embeddings.T)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=args.threshold,
    ).fit(distance_matrix)

    groups: dict[int, list[dict]] = {}
    for index, label in enumerate(clustering.labels_):
        groups.setdefault(int(label), []).append(faq_raw[index])

    deduped = []
    for label, items in groups.items():
        representative = min(items, key=lambda item: len(item.get("canonical_q", "~" * 999)))
        merged_ids: list[str] = []
        sources: list[str] = []
        for item in items:
            merged_ids.extend(item.get("source_conversation_ids", []))
            sources.append(item.get("canonical_q"))
        representative["source_conversation_ids"] = sorted(set(merged_ids))
        representative["near_duplicates"] = sorted({src for src in sources if src})
        deduped.append(representative)

    write_json(out_dir / "global_faq_clusters_dedup.json", deduped)

if __name__ == "__main__":
    main()
