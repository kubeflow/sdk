# Agent failure log

Repeatable issues caught in review on AI-assisted PRs. Each row should graduate to a
**lint, test, example, or path-scoped rule** — not a permanent bullet in `AGENTS.md`.

## How to use

1. After review on an AI-assisted PR, add one row if the comment describes a **pattern** (not a one-off bug).
2. Pick the lowest fix layer that prevents recurrence (see table below).
3. Implement the fix in a follow-up PR; set **Status** to `fixed` and note the fix.
4. Quarterly: archive `fixed` rows older than six months; prune prose duplicated by automation.

## Fix layer (pick the lowest that works)

| If the issue is… | Fix with… | Avoid… |
|------------------|-----------|--------|
| Objective, checkable | Lint / CI (`make verify`, `make lint-imports`, ruff) | Long `AGENTS.md` prose |
| Scoped to one component | `.agents/rules/kubeflow-*.mdc` | Global agent instructions |
| Missing exemplar | `common-changes.md` + canonical `*_test.py` | Tutorial in `AGENTS.md` |
| Architectural misunderstanding | `docs/design/` or `docs/adr/` | Duplicating in multiple files |
| One-time mistake | PR comment only | Permanent rule |

## Log

| Date | PR / branch | Failure mode | Root cause | Fix | Layer | Status |
|------|-------------|--------------|------------|-----|-------|--------|
| 2026-07 | ai-scaffolding-3 | Agent added cross-component imports (e.g. optimizer → trainer.options) | Boundaries documented in `.mdc` but not enforced in CI | `.importlinter` + `make lint-imports` + CI step | lint | fixed |

**Status values:** `open` · `fixed` · `deferred` · `wont-fix`
