# Lessons — Mistakes & Corrections (llm-anonymize)

Every correction, bug, and failure pattern gets logged here.
Each lesson feeds into: (1) relevant SKILL.md, (2) CLAUDE.md known failures.

---

## 2026-02-28: Internal Path Leak

**What:** Skill files and CLAUDE.md contained hardcoded paths to a private machine. Open-source repos must not reference internal infrastructure.
**Fix:** Scrub all files with `grep -rn "/home/" .` before pushing. No absolute paths in public repos.

## 2026-03-08: Embedding Timeout Cascade

**What:** Downstream integrations of the pipeline can have tighter constraints (e.g., timeouts). The standalone version works fine, but production deployments may need tuning.
**Lesson:** Test the pipeline in the deployment context, not just standalone.
