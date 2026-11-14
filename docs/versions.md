# Pinned package versions

Checked on 2026-11-14 on Windows 11 (CPU development machine).
Locked in `uv.lock` (Python) and `web/pnpm-lock.yaml` (Node/web).

## Runtime toolchains

| Tool | Local version | Notes |
|---|---|---|
| Python | `3.12.8` | Matches `>=3.11,<3.13` in `pyproject.toml` |
| `uv` | `0.13.8` | Lockfile committed at `uv.lock`; every agent must use `uv.lock` |
| Node.js | `22.20.0` | Matches Node 20+ requirement |
| `pnpm` | `12.18.0` | Lockfile committed at `web/pnpm-lock.yaml` |

## Required packages from `CLAUDE.md` section 9

| Package | Ecosystem | Pinned version | Latest registry version | Status and notes |
|---|---|---|---|---|
| `torch` | PyPI | `2.7.1` | `2.15.0` | Narrow-pinned to tested local version `2.7.1` |
| `transformers` | PyPI | `5.10.2` | `5.22.2` | Narrow-pinned to tested local version `5.10.2` (`5.x` line for `nnsight` `0.8.x`) |
| `nnsight` | PyPI | `0.8.1` | `0.8.2` | Narrow-pinned to tested local version `0.8.1` |
| `sae-lens` | PyPI | dropped | `6.55.5` | Dropped from scope per owner decision (E5 SAE features removed) |
| `fastapi` | PyPI | `0.115.0` | `0.162.5` | Narrow-pinned to tested local version `0.115.0` (`starlette>=0.38.6,<0.42.0`) |
| `mcp` | PyPI | `1.26.0` | `2.3.0` | Narrow-pinned to `1.26.0` (`2.x` renames `FastMCP` to `MCPServer`) |
| `claude-agent-sdk` | PyPI | `0.2.177` | `0.2.177` | Installed and locked |
| `three` | npm | `0.187.0` | `0.187.0` | Installed in `web/` (with `@types/three` `0.187.0`) |
| `@react-three/fiber` | npm | `9.8.2` | `9.8.2` | Installed in `web/` (`v9` stable line for React 19; `v10` alpha avoided) |
| `@react-three/drei` | npm | `10.7.9` | `10.7.9` | Installed in `web/` |
| `@react-spring/three` | npm | `10.1.2` | `10.1.2` | Installed in `web/` |
| `plotly.js` | npm | `4.3.0` | `4.3.0` | Installed in `web/` |

## Supporting Python packages (`pyproject.toml` & `uv.lock`)

| Package | Pinned constraint | Status |
|---|---|---|
| `pydantic` | `==2.13.4` | Installed |
| `accelerate` | `==1.13.0` | Installed |
| `huggingface_hub` | `==1.18.0` | Installed |
| `safetensors` | `==0.5.3` | Installed |
| `scikit-learn` | `==1.6.1` | Installed |
| `scipy` | `==1.13.1` | Installed |
| `numpy` | `==1.26.4` | Installed |
| `pandas` | `==2.2.3` | Installed |
| `pyarrow` | `==23.0.1` | Installed |
| `uvicorn` | `>=0.31.1,<1.0.0` | Installed |
| `websockets` | `==15.0.1` | Installed |
| `typer` | `==0.25.1` | Installed |
| `jinja2` | `==3.1.6` | Installed |
| `plotly` | `==6.0.1` | Installed |
| `jupyter_client` | `==8.6.3` | Installed |
| `ipykernel` | `==6.29.5` | Installed |
| `peft` | `==0.21.3` | Installed |
| `pytest` | `==8.3.5` | Installed |
| `pytest-asyncio` | `==0.26.0` | Installed |
| `ruff` | `==0.11.10` | Installed |
| `httpx` | `>=0.27.0,<1.0.0` | Installed |

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
