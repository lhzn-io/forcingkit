# forcingkit — Copilot Compatibility

This file exists for tools that specifically load `.github/copilot-instructions.md`.

## Canonical Guidance

- Treat `/AGENTS.md` as the primary instruction entry point.
- Treat `/.agents/` as the vendor-neutral location for deeper project guidance.

## Copilot-Specific Compatibility Notes

- Always use `uv run` for Python commands and `uv run pytest` for tests.
- Run `uv run pre-commit run --files <modified-files>` before staging any changes.
- Never commit without explicit user approval: present the change set and commit
  message first, then wait for positive confirmation.
- Stage files explicitly (`git add <file>`). Never `git add .`.
