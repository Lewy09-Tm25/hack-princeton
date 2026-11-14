# Claude Code plugin for the model safety tool

This folder packages the local interpretability engine as a Claude Code plugin so you can investigate language model safety risks directly from Claude Code while sharing the same `./workspace` database, artifacts, and reports as the web app.

## What is included

- `.claude-plugin/plugin.json` — plugin manifest (`model-safety-tool`, Apache-2.0).
- `.mcp.json` — default stdio MCP server configuration (`model-safety-engine`) invoking `tool mcp-stdio --workspace ${CLAUDE_PLUGIN_ROOT}/../workspace`.
- `skills/investigate-risk/SKILL.md` — step-by-step workflow enforcing check inspection, plain-language plan presentation, explicit user approval before `run_plan`, reviewer verification, 3D view updates, and coverage-bounded reporting.
- `agents/reviewer.md` — independent `reviewer` subagent restricted strictly to `get_evidence` and `submit_review`, able only to keep or lower a finding's strength.

## Option 1: Launch Claude Code with the plugin directory (stdio mode)

After installing the Python environment (`uv sync --locked --all-extras` from the repository root), start Claude Code with `--plugin-dir`:

```bash
claude --plugin-dir ./plugin
```

In this mode, `.mcp.json` starts `tool mcp-stdio --workspace ./workspace` automatically as a subprocess. Runs, results, findings, and HTML reports are saved to `./workspace/record.sqlite` and `./workspace/reports/`, so opening the web app against `./workspace` displays everything recorded during the Claude Code session.

## Option 2: Connect over HTTP alongside the live 3D web app

If `tool serve` is already running so you can watch live 3D `focus`, `paint`, and `timeline` events stream into an open browser tab in real time, connect Claude Code over streamable HTTP to the same server process:

1. Start the engine server and web app from the repository root:

   ```bash
   uv run --locked --all-extras tool serve --workspace ./workspace
   ```

2. Register the live HTTP MCP endpoint in Claude Code (also shown in the web app's Claude Code setup panel at `/api/setup/claude-code`):

   ```bash
   claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp
   ```

   If `tool serve --require-token` was used, include `--header "Authorization: Bearer <token>"` as printed by `tool serve`.

3. Start Claude Code (with `--plugin-dir ./plugin` if you also want the `investigate-risk` skill and `reviewer` subagent loaded) and ask a question such as:

   > Check whether `Qwen/Qwen2.5-0.5B-Instruct` with my adapter has weakened refusals compared with the base model.

## Plan approval and data handling

- **Nothing runs without approval:** Claude Code drafts a plan with `draft_plan`, explains the methods, datasets, required controls, coverage, limits, and runtime estimate, and waits for your confirmation before calling `run_plan(..., approved_by_user=True)` (which Claude Code also prompts you to approve).
- **Harmful prompt text stays inside the engine:** datasets containing harmful instructions are referenced by ID only (`refusal/harmful`, etc.), and MCP tools return aggregate numbers and redacted examples rather than raw prompt text.
