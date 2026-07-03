# Development Workflow for AI Agents

**Preferred commands**: use `uv run ...` to ensure tool consistency and `.venv` usage

**Before making changes**:

1. Read existing code patterns and docstrings for alignment
2. Follow [core-principles.md](core-principles.md)
3. Run validation commands before proposing changes

**Validation before proposing changes**:

- Lint/format: `make verify`
- Tests: `make test-python` or targeted `pytest` invocations
- Type checking: `uv run ty check kubeflow`

**Commit/PR hygiene**:

- Follow Conventional Commits in titles and messages
- Include rationale ("why") in commit messages/PR descriptions
- Do not push secrets or change git config
- Scope discipline: only modify files relevant to the task; keep diffs minimal

**Continuous improvement (AI-assisted PRs)**:

When review catches a **repeatable pattern** (not a one-off bug), log it in
[agent-failure-log.md](agent-failure-log.md), classify the root cause, and fix at the
lowest layer that prevents recurrence — prefer lint or tests over expanding `AGENTS.md`.
Treat agent context files as a backlog not yet automated; prune lines that CI or tests 
already enforce.