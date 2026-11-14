# Pinned package versions

Checked on 2026-11-14 on Windows 11 (CPU development machine).

## Runtime toolchains

| Tool | Local version | Notes |
|---|---|---|
| Python | `3.12.8` | Matches `>=3.11,<3.13` in `pyproject.toml` |
| `uv` | `0.13.8` | Virtualenv at `.venv` (`--system-site-packages` for local CPU packages) |
| Node.js | `22.20.0` | Matches Node 20+ requirement |
| `pnpm` | `12.18.0` | Lockfile at `web/pnpm-lock.yaml` |

## Required packages from `CLAUDE.md` section 9

| Package | Ecosystem | Local / pinned version | Latest registry version | Status and notes |
|---|---|---|---|---|
| `torch` | PyPI | `2.7.1` (`>=2.7.1,<3.0.0`) | `2.15.0` | Installed locally (CPU works; CUDA auto-detected when present) |
| `transformers` | PyPI | `5.10.2` (`>=5.10.2,<6.0.0`) | `5.22.2` | Installed locally; `5.x` line required by `nnsight` `0.8.x` |
| `nnsight` | PyPI | `0.8.1` (`>=0.8.1,<0.9.0`) | `0.8.2` | Installed locally; greenlet trace execution verified on Windows CPU |
| `sae-lens` | PyPI | `6.55.5` | `6.55.5` | Pinned in `[project.optional-dependencies].sae`; not yet installed locally |
| `fastapi` | PyPI | `0.115.0` (`>=0.115.0,<1.0.0`) | `0.162.5` | Installed locally |
| `mcp` | PyPI | `1.26.0` (`>=1.26.0,<2.0.0`) | `2.3.0` | Installed locally; pinned `<2.0.0` because `2.x` renames `FastMCP` to `MCPServer` (see `docs/spike.md`) |
| `claude-agent-sdk` | PyPI | `0.2.177` | `0.2.177` | Pinned in `[project.optional-dependencies].agent`; not yet installed locally (~70 MB wheel with bundled CLI) |
| `three` | npm | `0.187.0` | `0.187.0` | Installed in `web/` (with `@types/three` `0.187.0`) |
| `@react-three/fiber` | npm | `9.8.2` | `9.8.2` | Installed in `web/` (`v9` stable line for React 19; `v10` alpha avoided) |
| `@react-three/drei` | npm | `10.7.9` | `10.7.9` | Installed in `web/` |
| `@react-spring/three` | npm | `10.1.2` | `10.1.2` | Installed in `web/` |
| `plotly.js` | npm | `4.3.0` | `4.3.0` | Installed in `web/` |

## Supporting Python packages

| Package | Local version | Constraint in `pyproject.toml` | Status |
|---|---|---|---|
| `pydantic` | `2.13.4` | `>=2.13.4,<3.0.0` | Installed |
| `accelerate` | `1.13.0` | `>=1.13.0,<2.0.0` | Installed |
| `huggingface_hub` | `1.18.0` | `>=1.18.0,<2.0.0` | Installed |
| `safetensors` | `0.5.3` | `>=0.5.3,<1.0.0` | Installed |
| `scikit-learn` | `1.6.1` | `>=1.6.1,<2.0.0` | Installed |
| `scipy` | `1.13.1` | `>=1.13.1,<2.0.0` | Installed |
| `numpy` | `1.26.4` | `>=1.26.4,<3.0.0` | Installed |
| `pandas` | `2.2.3` | `>=2.2.3,<3.0.0` | Installed |
| `pyarrow` | `23.0.1` | `>=23.0.1,<24.0.0` | Installed |
| `uvicorn` | `0.30.6` | `>=0.30.6,<1.0.0` | Installed |
| `websockets` | `15.0.1` | `>=15.0.1,<16.0.0` | Installed |
| `typer` | `0.25.1` | `>=0.25.1,<1.0.0` | Installed |
| `jinja2` | `3.1.6` | `>=3.1.6,<4.0.0` | Installed |
| `plotly` | `6.0.1` | `>=6.0.1,<7.0.0` | Installed |
| `jupyter_client` | `8.6.3` | `>=8.6.3,<9.0.0` | Installed |
| `ipykernel` | `6.29.5` | `>=6.29.5,<7.0.0` | Installed |
| `peft` | not installed | `>=0.15.0,<1.0.0` | Needed for LoRA adapter loading in E1 (~1 MB pure-Python install) |
| `transformer-lens` | `3.4.0` | optional | Installed locally (optional hook-name reference only) |
| `pytest` | `8.3.5` | `>=8.3.5,<9.0.0` | Installed |
| `pytest-asyncio` | `0.26.0` | `>=0.26.0,<1.0.0` | Installed |
| `ruff` | `0.11.10` | `>=0.11.10,<1.0.0` | Installed |
| `httpx` | `0.27.0` | `>=0.27.0,<1.0.0` | Installed |

## Supporting web packages (`web/package.json`)

| Package | Exact version |
|---|---|
| `react` / `react-dom` | `19.3.1` |
| `zustand` | `5.1.0` |
| `vite` | `8.3.8` |
| `@vitejs/plugin-react` | `6.3.0` |
| `vitest` | `5.1.0` |
| `typescript` | `5.9.3` (pinned `<7.0` because `typescript-eslint@8.72.2` does not yet support TypeScript 7.0) |
| `eslint` | `10.14.0` |
| `typescript-eslint` | `8.72.2` |
