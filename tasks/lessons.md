# Lessons — Mistakes & Corrections (llm-anonymize)

Every correction, bug, and failure pattern gets logged here.
Each lesson feeds into: (1) relevant SKILL.md, (2) CLAUDE.md known failures.

---

## 2026-02-28: Two-Repo Sync Risk

**What:** `anonymize.py` lives in both this repo and `local-ai-toolkit/tools/`. Changes in one must be mirrored to the other.
**Fix:** Documented in `update-pipeline` skill. Always sync both in same session.

## 2026-03-08: Embedding Timeout Cascade

**What:** The toolkit copy of anonymize.py is used by `pa_common.py` which had a 30s embedding timeout. Not directly this repo's bug, but downstream impact of the pipeline being used in production.
**Lesson:** Open-source version is standalone, but the toolkit copy lives in a production system with tighter constraints. Test both contexts.
