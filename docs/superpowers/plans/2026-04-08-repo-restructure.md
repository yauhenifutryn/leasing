# Repo Restructure Plan

**Status:** Pending (execute after all current feature work is complete)
**Date:** 2026-04-08

## Context

The leasing repo has accumulated long-lived branches that function as separate products rather than features:

| Branch | Ahead of main | Role |
|---|---|---|
| `feature/voice-pipeline` | 223 | Production voice system |
| `feature/tool-use` | 235 | Tool use (calculator, SMS) on top of voice-pipeline |
| `claude/qwen-voice-next` | 204 | Experimental Qwen3-Omni path |
| `codex/split-voice-providers` | 3 | Spike: split-brain voice providers |
| `codex/yandex-realtime-voice-integration` | 5 | Spike: Yandex realtime demo |

`main` contains only the original RAG demo baseline and is effectively abandoned.

## Goal

Single `main` branch representing the production product. Experiments preserved as tags. Clean branch list. Professional README.

## Prerequisites

- [ ] All pending work on `feature/tool-use` is complete
- [ ] All other planned features for initial release are done
- [ ] System tested end-to-end on current branches

## Steps

### Phase 1: Safety snapshots

Tag every branch at its current state before any destructive operations:

```bash
git tag snapshot/voice-pipeline-2026-XX-XX feature/voice-pipeline
git tag snapshot/tool-use-2026-XX-XX feature/tool-use
git tag snapshot/qwen-voice-next-2026-XX-XX claude/qwen-voice-next
git tag snapshot/codex-split-providers-2026-XX-XX codex/split-voice-providers
git tag snapshot/codex-yandex-realtime-2026-XX-XX codex/yandex-realtime-voice-integration
git push origin --tags
```

### Phase 2: Merge core into main

```bash
git checkout main
git merge feature/voice-pipeline    # resolve conflicts
# verify everything works
git merge feature/tool-use          # resolve conflicts (should be minimal, 12 commits)
# verify everything works end-to-end
git push origin main
```

### Phase 3: Tag experiments

```bash
git tag experiment/qwen-voice-next claude/qwen-voice-next
git tag experiment/yandex-realtime codex/yandex-realtime-voice-integration
# decide per-branch: keep as tag or discard entirely
git push origin --tags
```

### Phase 4: Delete branches

```bash
# Delete remote branches
git push origin --delete feature/voice-pipeline
git push origin --delete feature/tool-use
git push origin --delete claude/qwen-voice-next
git push origin --delete codex/split-voice-providers
git push origin --delete codex/yandex-realtime-voice-integration
git push origin --delete fix/orjson-fallback

# Delete local branches
git branch -D feature/voice-pipeline
git branch -D feature/tool-use
git branch -D claude/qwen-voice-next
git branch -D codex/split-voice-providers
git branch -D codex/yandex-realtime-voice-integration
```

### Phase 5: Final README and cleanup

- [ ] Update README.md to reflect the merged product structure
- [ ] Verify no references to deleted branches in docs
- [ ] Confirm all tags are pushed to remote

## Post-restructure branch policy

- `main` is the production branch, always deployable
- Feature branches are short-lived: merge or tag+delete within 2 weeks
- Experiment branches use `experiment/` prefix and follow the same TTL rule
