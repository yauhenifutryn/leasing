# KB baseline snapshot — 2026-05-03

Immutable byte-for-byte copy of the production KB at the moment `feature/kb-refinement` branched from `feature/voice-pipeline`.

## Branch / commit anchor

- Branch: `feature/kb-refinement`
- Parent commit (HEAD of `feature/voice-pipeline` at fork): `381c13d` (`docs(plan): add Section 7 ...`), grandparent `9ba36d5` (last code commit)
- Snapshot taken before any Section 7 edit landed.

## Files

| File | SHA-256 |
|---|---|
| `kb_faq_ru.yaml` | `69de9c3eefdc4c8e4126d17c694465c4009f6575833b9f293b74a8ae1dea2589` |
| `kb_faq_ru_v2.md` | `931eb02149388b692850894f551f331521608452ccb5ed07d638f396b5d11a19` |
| `kb_faq_ru.json` | `3b2383e284c0b74e19dca6357b54fe270ced84c21e7eaddb469930a2398d9b11` |

Verify integrity at any time:

```bash
shasum -a 256 knowledge_base/_archive/baseline_2026_05_03/*
```

## Revert procedure (worst case)

If Section 7 work needs to be fully rolled back at the file level:

```bash
cp knowledge_base/_archive/baseline_2026_05_03/kb_faq_ru.yaml   knowledge_base/kb_faq_ru.yaml
cp knowledge_base/_archive/baseline_2026_05_03/kb_faq_ru_v2.md  knowledge_base/kb_faq_ru_v2.md
cp knowledge_base/_archive/baseline_2026_05_03/kb_faq_ru.json   knowledge_base/kb_faq_ru.json
# Then re-index Qdrant on the server:
python rag_demo_system/scripts/index_kb.py
# Restart backend
```

For production rollback Section 7 also exposes a `KB_LAYOUT=legacy|topical` env-var (Phase C). The file-level revert above is the deeper safety net.

## Do NOT edit any file in this directory

This dir is the trust anchor. If a file here changes, the safety guarantee is gone.
