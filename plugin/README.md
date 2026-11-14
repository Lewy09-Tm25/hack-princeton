# Claude Code plugin for the model safety tool

This folder packages the local interpretability engine as a Claude Code plugin so you can investigate language model safety risks directly from Claude Code while animating the web app's live 3D view and sharing the same `./workspace` database, artifacts, and HTML reports.

## What is included

- `.claude-plugin/plugin.json` — plugin manifest (`model-safety-tool`, Apache-2.0).
- `.mcp.json` — streamable HTTP MCP server configuration (`model-safety-engine`, `"type": "http"`, `"url": "http://127.0.0.1:8000/mcp"`) connecting to the running `tool serve` process so checks started from Claude Code stream live `focus`, `paint`, and `timeline` events into the web app's 3D view.
- `skills/investigate-risk/SKILL.md` — step-by-step workflow enforcing check inspection, plain-language plan presentation, explicit user approval before `run_plan`, reviewer verification, 3D view updates, and coverage-bounded reporting.
- `agents/reviewer.md` — independent `reviewer` subagent restricted strictly to `get_evidence` and `submit_review`, able only to keep or lower a finding's strength.

## Default setup: Connect over HTTP alongside the live 3D web app

1. Start the local engine server from the repository root (after `uv sync --locked --all-extras`):

   ```bash
   uv run --locked --all-extras tool serve --workspace ./workspace
   ```

2. Open the web app (`pnpm --dir web dev` at `http://localhost:5173`) if you want to watch the 3D layer stack update in real time.

3. Launch Claude Code with the bundled plugin directory:

   ```bash
   claude --plugin-dir ./plugin
   ```

   Because `plugin/.mcp.json` connects over HTTP to `http://127.0.0.1:8000/mcp`, any investigation started in Claude Code animates the open 3D browser tab and writes runs, findings, and HTML reports into `./workspace`.

   *(You can also register the HTTP MCP server directly without `--plugin-dir` via `claude mcp add --transport http model-safety-engine http://127.0.0.1:8000/mcp`; if `tool serve --require-token` was used, include `--header "Authorization: Bearer <token>"`.)*

## Alternative: Headless stdio MCP mode (without `tool serve`)

If you want Claude Code to spawn the engine directly as a subprocess over `stdio` without keeping `tool serve` running in a separate terminal, register the `stdio` server:

```bash
claude mcp add model-safety-engine -- tool mcp-stdio --workspace ./workspace
```

Or use a project `.mcp.json` stdio entry:

```json
{
  "mcpServers": {
    "model-safety-engine": {
      "command": "tool",
      "args": ["mcp-stdio", "--workspace", "./workspace"]
    }
  }
}
```

Runs, results, findings, and HTML reports are still saved to `./workspace/record.sqlite` and `./workspace/reports/`, so opening the web app later against `./workspace` displays everything recorded during the session.

## Plan approval and data handling

- **Nothing runs without approval:** Claude Code drafts a plan with `draft_plan`, explains the methods, datasets, required controls, coverage, limits, and runtime estimate, and waits for your confirmation before calling `run_plan(..., approved_by_user=True)` (which Claude Code also prompts you to approve).
- **Harmful prompt text stays inside the engine:** datasets containing harmful instructions are referenced by ID only (`refusal/harmful`, etc.), and MCP tools return aggregate numbers and redacted examples rather than raw prompt text.
