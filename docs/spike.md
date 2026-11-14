# Verification spike findings

Date: 2026-11-14
Companion to: [CLAUDE.md](../CLAUDE.md) section 9 and [versions.md](versions.md).

---

## 1. Package versions and compatibility notes

Full pinned versions are in [versions.md](versions.md), locked in `uv.lock` and `web/pnpm-lock.yaml`.

Key compatibility findings:

1. **Narrowed `torch` and engine pins (`uv.lock`):**
   - An unpinned resolve currently pulls `torch` `2.15.0`. `pyproject.toml` pins `torch==2.7.1` (the exact version tested on this machine) along with the tested versions of `transformers==5.10.2`, `nnsight==0.8.1`, `mcp==1.26.0`, and `fastapi==0.115.0`, committed in `uv.lock`.
2. **`mcp` `1.26.0` vs `2.3.0`:**
   - PyPI's latest `mcp` (`2.3.0`) renamed `from mcp.server.fastmcp import FastMCP` to `from mcp.server.mcpserver import MCPServer` and moved transport arguments out of the constructor.
   - `pyproject.toml` pins `mcp==1.26.0`, which matches `design.md` section 8.1 and `roadmap.md` E13 (`FastMCP` + `streamable_http_app()`).
3. **`typescript` `5.9.3` vs `7.0.2` in `web/`:**
   - `typescript-eslint@8.72.2` does not yet support TypeScript 7.0 (`7.0.2`). Pinning `typescript` to `5.9.3` allows `tsc -b`, `vite build`, `vitest run`, and `eslint .` to pass cleanly.

---

## 2. Claude Agent SDK (`claude-agent-sdk==0.2.177`)

Checked against the official Python Agent SDK reference (`ClaudeAgentOptions`, `ClaudeSDKClient`, `query`, `AgentDefinition`, `HookMatcher`).

### Connecting an external MCP server
Pass `mcp_servers` on `ClaudeAgentOptions`:
```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    mcp_servers={
        "engine": {
            "type": "http",
            "url": "http://127.0.0.1:8000/mcp",
            "headers": {"Authorization": f"Bearer {local_token}"},
        }
    }
)
```
Tools exposed by server `"engine"` are named `mcp__engine__<tool_name>` (for example `mcp__engine__draft_plan`). Stdio servers use `{"type": "stdio", "command": ..., "args": [...]}`.

### Restricting tools
- `allowed_tools` only pre-approves permissions; it does **not** remove built-in tools from the model's tool list.
- To restrict the web chat agent strictly to engine MCP tools and the reviewer subagent, set both `tools` and `disallowed_tools`:
  ```python
  BUILT_IN_TOOLS = [
      "Bash", "Read", "Write", "Edit", "MultiEdit",
      "Glob", "Grep", "WebSearch", "WebFetch", "NotebookEdit",
  ]

  options = ClaudeAgentOptions(
      tools=["Agent"],
      disallowed_tools=BUILT_IN_TOOLS,
      allowed_tools=["Agent", "mcp__engine__*"],
  )
  ```

### Model, adaptive thinking, and refusal-fallback note
- Model ID: `"claude-opus-5-5"` for both the built-in agent (`A1`) and the reviewer (`A2`).
- Adaptive thinking: pass `thinking={"type": "adaptive"}` (or `ThinkingConfigAdaptive(type="adaptive")`). Do **not** pass `thinking={"type": "enabled", "budget_tokens": ...}` to `claude-opus-5-5`; manual `budget_tokens` is rejected on Opus 5.5.
- **Refusal-fallback betas:** Do **not** rely on or hard-code refusal-fallback beta flags (`betas=[...]`) until a real API call confirms how `claude-agent-sdk==0.2.177` forwards them. Rely on keeping raw harmful prompt text inside the engine (redacted aggregates only) as the primary safeguard, and make any beta flags optional config verified during live API testing.

### Defining the reviewer subagent
Use `AgentDefinition` under `ClaudeAgentOptions.agents`:
```python
from claude_agent_sdk import AgentDefinition

reviewer = AgentDefinition(
    description="Checks each finding's claim wording and strength against its engine evidence bundle.",
    prompt=reviewer_prompt_markdown,
    tools=["mcp__engine__get_evidence", "mcp__engine__submit_review"],
    model="claude-opus-5-5",
)
```

### Pre-tool-use hook for plan approval (A3)
Use `hooks={"PreToolUse": [HookMatcher(matcher="mcp__engine__run_plan", hooks=[check_plan_approved])]}`:
```python
async def check_plan_approved(input_data, tool_use_id, context):
    plan_id = input_data.get("tool_input", {}).get("plan_id")
    if not is_plan_approved(plan_id):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Plan has not been approved by the user yet. Ask the user for approval first.",
            }
        }
    return {}
```

---

## 3. MCP Python SDK (`mcp==1.26.0`) inside FastAPI and over stdio

### Streamable HTTP mounted inside FastAPI
`FastMCP.streamable_http_app()` returns a Starlette/ASGI app whose lifespan **must** run inside FastAPI's `lifespan` context manager:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("model-safety-engine")
mcp_app = mcp.streamable_http_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.router.lifespan_context(app):
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/mcp", mcp_app)
```

### Stdio transport
For `tool mcp-stdio`, call `mcp.run(transport="stdio")` in a dedicated CLI command without starting FastAPI/uvicorn, and send logs to `stderr` only.

### Windows `mcp`-before-`nnsight` import order (enforced in `engine/__init__.py`)
- Importing `nnsight` (`0.8.1`) attaches `.save` to Python's root `object` type via C-API.
- On Windows, `import mcp` loads `anyio.streams.file.FileStreamAttribute(TypedAttributeSet)`, whose metaclass raises `TypeError: Attribute 'save' is missing its type annotation` if `nnsight` was imported first.
- **Single-point fix:** `engine/__init__.py` calls `ensure_mcp_imported_before_nnsight()`, which pre-imports `anyio.streams.file` and `mcp.server.fastmcp` whenever any `engine` module is imported, and raises `RuntimeError` if `nnsight` was imported beforehand. Regression-tested in `tests/test_import_order.py`.

---

## 4. Claude Code plugin format

Checked against the Claude Code plugins reference (`https://code.claude.com/docs/en/plugins-reference`):

```text
plugin/
  .claude-plugin/
    plugin.json          # required field: "name" (kebab-case slug), plus "version", "description"
  .mcp.json              # at plugin root: {"mcpServers": {"model-safety-engine": {"type": "http", "url": "http://127.0.0.1:8000/mcp"}}}
  skills/
    investigate-risk/
      SKILL.md           # YAML frontmatter: name, description
  agents/
    reviewer.md          # subagent markdown with YAML frontmatter (name, description, tools)
  README.md
```

---

## 5. `nnsight` on Windows 11 CPU — verified on real weights

- **Status:** Verified on real downloaded weights (`EleutherAI/pythia-70m` and `Qwen/Qwen2.5-0.5B-Instruct`, `float32` CPU).
- **Findings and timings:**
  - `nnsight.LanguageModel` emits a deprecation warning in `0.8.1`; `engine/loader` uses `nnsight.TransformersModel(hf_model, tokenizer=tokenizer, task="text-generation")` for Hugging Face models and `nnsight.NNsight(torch_module)` for custom `nn.Module` models.
  - In `transformers` `5.10.2`, transformer layer `.output` inside a trace resolves directly to a `torch.Tensor` (`[batch, seq_len, hidden_size]`); `E4` handles both `Tensor` and `tuple[Tensor, ...]`.
  - Real `EleutherAI/pythia-70m` (`70,426,624` params): single-prompt `nnsight` trace saving `gpt_neox.layers[2].output` (`(1, 5, 512)`) takes `0.060s–0.094s` on CPU.
  - Real `Qwen/Qwen2.5-0.5B-Instruct` (`494,032,768` params): single-prompt `nnsight` trace saving `model.layers[12].output` (`(1, 5, 896)`) takes `0.382s` on CPU, and batched `collect_activations` across `64` prompts and all `24` layers (`shape=(64, 896)` per layer) takes `4.56s` on CPU (well within the `< 30s` target).
  - `GPTNeoXForCausalLM` layers include `post_attention_dropout` and `post_mlp_dropout`; `E2` ignores `nn.Dropout` modules so both `EleutherAI/pythia-70m` (`81` components) and `Qwen/Qwen2.5-0.5B-Instruct` (`459` components) map cleanly with `needs_confirmation=[]`.

---

## 6. Pythia checkpoint revisions (`step1000` .. `step143000`) — verified on real weights

- **Status:** Verified across all six pretraining revisions in `docs/defaults.md` (`step1000`, `step8000`, `step16000`, `step32000`, `step64000`, `step143000`).
- Each revision branch of `EleutherAI/pythia-70m` ships `model.safetensors` (`268.7 MiB` / `281.8 MB` decimal per checkpoint, `~1.69 GB` total across all six checkpoints) and loads cleanly through `load_model("EleutherAI/pythia-70m", revision=rev)` with `is_placeholder=False` (`0.18s–0.22s` warm load per checkpoint on CPU).
- Note on Windows Hugging Face Hub caching: without Windows Developer Mode or elevated privileges, `huggingface_hub` falls back from symlinks to file copies inside `~/.cache/huggingface/hub` and emits a `UserWarning` (`HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences the warning).

---

## 7. SAELens + Gemma Scope 2 — dropped from scope

- **Status:** Dropped from hackathon scope per owner decision (`Qwen/Qwen2.5-0.5B-Instruct` selected as the demo base model; `E5` SAE features removed from scope).
