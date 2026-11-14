# Verification spike findings

Date: 2026-11-14
Companion to: [CLAUDE.md](../CLAUDE.md) section 9 and [versions.md](versions.md).

---

## 1. Package versions and compatibility notes

Full pinned versions are in [versions.md](versions.md). Three version interactions differ from what a fresh unpinned install would give:

1. **`mcp` `1.26.0` vs `2.3.0`:**
   - PyPI's latest `mcp` (`2.3.0`) introduced a breaking v2 rename: `from mcp.server.fastmcp import FastMCP` moved to `from mcp.server.mcpserver import MCPServer`, and transport options moved out of the constructor.
   - `mcp==1.26.0` is already installed on this machine and matches `design.md` section 8.1 and `roadmap.md` E13 (`FastMCP` + `streamable_http_app()`). `pyproject.toml` pins `mcp>=1.26.0,<2.0.0` so existing design interfaces work without churn.
2. **`nnsight` `0.8.1` + `transformers` `5.10.2`:**
   - Both are already installed locally and work together natively (`nnsight` `0.8.x` targets `transformers` `5.x` and uses greenlets for trace interleaving).
3. **`typescript` `5.9.3` vs `7.0.2` in `web/`:**
   - Latest npm `typescript` is `7.0.2`, which `typescript-eslint@8.72.2` explicitly rejects (`Error: typescript-eslint does not support TS 7.0`). Pinning `typescript` to `5.9.3` makes `tsc -b`, `vite build`, `vitest run`, and `eslint .` all pass cleanly.

Packages not yet installed locally:
- `peft` (~1 MB wheel): needed in E1 for merging LoRA adapters.
- `claude-agent-sdk==0.2.177` (~70 MB wheel including bundled CLI): needed in A1–A3.
- `sae-lens==6.55.5` (optional, only if E5 SAE support stays in scope).

---

## 2. Claude Agent SDK (`claude-agent-sdk` Python)

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
  BUILT_IN_TOOLS = ["Bash", "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "WebSearch", "WebFetch", "NotebookEdit"]

  options = ClaudeAgentOptions(
      tools=["Agent"],  # built-in subagent delegation only; MCP tools attach via mcp_servers
      disallowed_tools=BUILT_IN_TOOLS,
      allowed_tools=["Agent", "mcp__engine__*"],
  )
  ```

### Model, adaptive thinking, and server-side refusal fallbacks
- Model ID: `"claude-opus-5-5"` (pinned dateless format).
- Adaptive thinking: on `claude-opus-5-5`, adaptive thinking is the native mode. Pass `thinking={"type": "adaptive"}` (or `ThinkingConfigAdaptive(type="adaptive")`) and optional `effort="high"`. Do **not** pass `thinking={"type": "enabled", "budget_tokens": ...}` to `claude-opus-5-5`; the API rejects manual `budget_tokens` on Opus 5.5 with HTTP 400.
- Server-side refusal fallbacks: passed via `betas=["server-side-fallback-2026-07-01", "fallback-credit-2026-07-01"]` on `ClaudeAgentOptions` when supported by the direct Anthropic API endpoint.

### Defining the reviewer subagent
Use `AgentDefinition` under `ClaudeAgentOptions.agents`:
```python
from claude_agent_sdk import AgentDefinition

reviewer = AgentDefinition(
    description="Checks each finding's claim wording and strength against its engine evidence bundle.",
    prompt=reviewer_prompt_markdown,
    tools=["mcp__engine__get_evidence", "mcp__engine__submit_review"],
    model="claude-opus-5-5",  # configurable per open decision #5
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
`FastMCP.streamable_http_app()` returns a Starlette/ASGI app. Its lifespan **must** run inside FastAPI's `lifespan` context manager so the underlying session manager starts:

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
For `tool mcp-stdio`, call `mcp.run(transport="stdio")` in a dedicated CLI command without starting FastAPI/uvicorn, and write logs to `stderr` only so `stdout` stays clean JSON-RPC.

### Windows-specific import-order finding (`mcp` + `nnsight`)
Tested locally on Windows 11:
- Importing `nnsight` (`0.8.1`) attaches a `.save` attribute to Python's root `object` type via C-API for trace proxies.
- On Windows, `import mcp` imports `mcp.os.win32.utilities` -> `anyio.streams.file.FileStreamAttribute(TypedAttributeSet)`. `TypedAttributeSet.__init_subclass__` inspects attributes on the class and raises `TypeError: Attribute 'save' is missing its type annotation` if `nnsight` was imported first!
- **Verified fix:** always import `mcp` (or `anyio.streams.file`) **before** importing `nnsight`. Verified locally: importing `FastMCP` and `fastapi` before `nnsight` succeeds cleanly with exit code 0.

---

## 4. Claude Code plugin format

Checked against the Claude Code plugins reference (`https://code.claude.com/docs/en/plugins-reference`):

```text
plugin/
  .claude-plugin/
    plugin.json          # required field: "name" (kebab-case slug), plus "version", "description"
  .mcp.json              # at plugin root (NOT inside .claude-plugin/)
  skills/
    investigate-risk/
      SKILL.md           # YAML frontmatter: name, description
  agents/
    reviewer.md          # subagent markdown with YAML frontmatter (name, description, tools)
  README.md
```

`.mcp.json` format at plugin root:
```json
{
  "mcpServers": {
    "model-safety-engine": {
      "type": "stdio",
      "command": "tool",
      "args": ["mcp-stdio", "--workspace", "${CLAUDE_PLUGIN_ROOT}/../workspace"]
    }
  }
}
```
Note: only `plugin.json` goes inside `.claude-plugin/`; `.mcp.json`, `skills/`, and `agents/` must sit at the root of `plugin/`.

---

## 5. `nnsight` on Windows 11 CPU

Tested locally in-memory on Windows 11 CPU using an uncached 2-layer `GPTNeoXForCausalLM` (same architecture class as `EleutherAI/pythia-70m`) so no unapproved Hub download was performed:

1. **`LanguageModel` deprecation:** `nnsight.LanguageModel(...)` works in `0.8.1` but emits:
   `NNsightDeprecationWarning: LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.`
2. **Wrapping a pre-loaded Hugging Face model:** `nnsight.TransformersModel(hf_model, tokenizer=tokenizer, task="text-generation")` requires passing `tokenizer=` explicitly when `hf_model` was loaded from memory or a custom path without `name_or_path` on the Hub.
3. **Wrapping custom non-Hugging-Face PyTorch models:** `nnsight.NNsight(torch_module)` wraps any `torch.nn.Module` directly without `transformers.pipeline`, which lets E2 and E4 trace custom user architectures seamlessly.
4. **Layer output shape in `transformers` `5.x`:** inside `with tm.trace("hello world"): out = tm.gpt_neox.layers[0].output.save()`, `out` resolves directly to a `torch.Tensor` of shape `[batch, seq_len, hidden_size]` (`torch.Size([1, 2, 32])`), **not** a `(hidden_states, ...)` tuple as in older `transformers` `4.x` releases. E4's `collect_activations` must handle both `Tensor` and `tuple[Tensor, ...]` (`out[0] if isinstance(out, tuple) else out`).
5. **Module paths verified locally in `transformers` `5.10.2`:**
   - `GPTNeoXForCausalLM` (Pythia): `gpt_neox.embed_in`, `gpt_neox.layers.{i}` (`input_layernorm`, `post_attention_layernorm`, `post_attention_dropout`, `post_mlp_dropout`, `attention`, `mlp`), `gpt_neox.final_layer_norm`, `embed_out`. Note: E2 should ignore `nn.Dropout` submodules so Pythia's dropout modules are not flagged into `needs_confirmation`.
   - `Qwen2ForCausalLM`: `model.embed_tokens`, `model.layers.{i}` (`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`), `model.norm`, `lm_head`.
   - `Gemma3ForCausalLM`: `model.embed_tokens`, `model.layers.{i}` (`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`), `model.norm`, `lm_head`.
6. **Real weight download test:** `EleutherAI/pythia-70m` is not in the local Hugging Face cache yet. Waiting for owner approval to download `EleutherAI/pythia-70m` (`step1000` and `main`, ~165 MB per checkpoint).

---

## 6. Pythia checkpoint revisions (`revision="step1000"`)

- Confirmed from EleutherAI's Hugging Face repository metadata: `EleutherAI/pythia-70m` publishes all 154 checkpoints (`step0`, `step1000`, ..., `step143000`) as Git branches on the same repo, and includes `model.safetensors` on checkpoint branches alongside `pytorch_model.bin`.
- E1's `snapshot_download(..., revision="step1000", allow_patterns=["*.safetensors", "*.json", "tokenizer*"])` will work with `safetensors`-only enforcement enabled.
- Live download check awaits owner approval (see download request in report).

---

## 7. SAELens + Gemma Scope 2 for Gemma 3 270M

- Confirmed: Google DeepMind's Gemma Scope 2 includes pretrained JumpReLU SAEs for Gemma 3 270M (`google/gemma-scope-2-270m-pt` and `google/gemma-scope-2-270m-it` on Hugging Face, covering `resid_post`, `attn_out`, and `mlp_out` at widths `16k`, `65k`, `262k`).
- Loadable through `sae-lens>=6.55.0` via `SAE.from_pretrained(release="gemma-scope-2-270m-pt-resid_post", sae_id="layer_12_width_16k_l0_small")`, and compatible with raw PyTorch activation tensors collected via `nnsight` (`sae.encode(acts)`).
- **Important constraint for demo base-model choice:** `google/gemma-3-270m-it` is a **gated** Hugging Face model requiring manual license acceptance on the Hugging Face web UI plus a valid `HF_TOKEN`. `Qwen/Qwen2.5-0.5B-Instruct` is **ungated** (Apache-2.0), works without Hub login, has stronger instruction-following and refusal baseline behavior at 0.5B than a 270M model, and aligns with the Qwen family used in D2/D3.

---

## 8. Small discrepancies noted across the planning docs

1. **`Component.n_heads`:** `docs/design.md` section 10.1 includes `"n_heads": 4` on attention components, whereas `docs/roadmap.md` section 3.2 omitted `n_heads` in the Python snippet. Added `n_heads: int | None = None` in `engine/contracts.py` and `web/src/contracts.ts` so both match.
2. **`Finding.maturity` values:** `docs/design.md` section 5 table lists `"Partly established"` for data poisoning, while `docs/design.md` 10.6, `docs/roadmap.md` 3.2, and `agent/prompts/system.md` restrict `maturity` to `"established" | "experimental"`. Kept `"established" | "experimental"` per 10.6 and 3.2; if needed, `"partly_established"` can be added or data poisoning's description can note partial maturity in text.
3. **`JobStatus` formatting:** `docs/design.md` 10.4 prose writes `needs decision` with a space, while `Event` enums in code use snake_case (`"needs_decision"`). Documented `"needs_decision"` in `docs/contracts.md` and enforced it in both Python and TypeScript contracts.
