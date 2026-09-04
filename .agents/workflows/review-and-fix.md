# Review and Fix Workflow

Workflow for processing PR review feedback: read comments, classify, apply
fixes, validate, and update the agent failure log.

## When to use

- After receiving review comments on a PR
- When an agent is asked to address review feedback

## Workflow

### Step 1: Gather review comments

Read all review threads on the PR. Classify each comment:

| Category | Action |
|----------|--------|
| Bug / incorrect behavior | Fix in code, add test |
| Style / convention mismatch | Fix in code, check `references/core-principles.md` |
| Documentation mismatch | Fix docs to match code (or vice versa) |
| Architecture concern | Discuss before changing; check `docs/design/` |
| Nit / optional | Fix if trivial; note if deferred |

### Step 2: Check for repeatable patterns

Before fixing, ask: **is this a one-off mistake or a repeatable pattern?**

- **One-off:** fix and move on.
- **Repeatable pattern:** fix, then log in `references/agent-failure-log.md`.

See the fix layer table in `references/agent-failure-log.md` to pick the
lowest layer that prevents recurrence:

| If the issue is... | Fix with... |
|-------------------|-------------|
| Objective, checkable | Lint / CI rule |
| Scoped to one component | `.agents/rules/kubeflow-*.mdc` |
| Missing exemplar | `references/common-changes.md` + canonical test |
| Architectural misunderstanding | `docs/design/` or `docs/adr/` |
| One-time mistake | PR comment only |

### Step 3: Apply fixes

Make changes in a single commit (or minimal commits). Follow these rules:

- Scope fixes to the reviewer's feedback — do not refactor unrelated code.
- Run single-file verification as you edit:

```bash
uv run ruff check path/to/file.py
uv run pytest -q path/to/file_test.py
```

### Step 4: Validate

Run full validation before pushing:

```bash
make verify
make lint-imports
make verify-openapi   # if openapi.yaml changed
```

### Step 5: Reply to review threads

For each addressed thread:

1. Leave a short reply explaining what was changed.
2. Reference the fix commit hash.
3. Resolve the conversation (if you are the PR author).

### Step 6: Update failure log (if applicable)

If Step 2 identified a repeatable pattern, add a row to
`references/agent-failure-log.md`:

```markdown
| <date> | <PR/branch> | <failure mode> | <root cause> | <fix> | <layer> | open |
```

### Step 7: Push

Push the fix commit. If the branch was rebased, use `--force-with-lease`.

## Agent guidance

- Do NOT resolve review threads that ask a question — reply and wait.
- Do NOT refactor beyond what the reviewer asked.
- When in doubt about architectural feedback, flag it for human review rather
  than guessing.
