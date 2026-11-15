"""Safe Hugging Face Hub fetcher for test models and checkpoint revisions (component D2).

Downloads are blocked unless `--confirm-download` is explicitly passed. Only
`*.safetensors`, `*.json`, and `tokenizer*` files are permitted; pickle/binary
weights and custom remote Python code are rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import engine  # noqa: F401 - ensure mcp-before-nnsight import order
from engine.dynamics.timeline import PYTHIA_DEFAULT_CHECKPOINTS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

SAFE_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.json",
    "tokenizer*",
)

UNSAFE_FILE_SUFFIXES: tuple[str, ...] = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".py",
    ".gguf",
)


@dataclass(frozen=True)
class OrganismSpec:
    """Catalog entry for a base model or candidate test model from `docs/roadmap.md` D2."""

    key: str
    repo_id: str
    role: str
    revisions: tuple[str, ...] = ("main",)
    community_untrusted: bool = False
    notes: str = ""


ORGANISM_CATALOG: tuple[OrganismSpec, ...] = (
    OrganismSpec(
        key="qwen25_05b_instruct",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        role="demo_base",
        community_untrusted=False,
        notes="Demo base model for the general demo and refusal checks.",
    ),
    OrganismSpec(
        key="qwen25_1_5b_instruct",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        role="sycophancy_and_backdoor_base",
        community_untrusted=False,
        notes=(
            "Clean 1.5B base model (28 layers, ~3.09 GB safetensors) used as the sycophancy "
            "test-model base and as the matched clean control that "
            "Antonhansel/qwen2.5-1.5b-sleeper-canary was fine-tuned from."
        ),
    ),
    OrganismSpec(
        key="sleeper_canary_1_5b",
        repo_id="Antonhansel/qwen2.5-1.5b-sleeper-canary",
        role="backdoor_small",
        community_untrusted=True,
        notes=(
            "Community sleeper-canary model fine-tuned from Qwen/Qwen2.5-1.5B-Instruct; "
            "must be verified against Qwen/Qwen2.5-1.5B-Instruct in organisms/verify.py."
        ),
    ),
    OrganismSpec(
        key="mot_sleeper_agents",
        repo_id="senku21x/mot-sleeper-agents-generality",
        role="backdoor_adapter",
        community_untrusted=True,
        notes="Community sleeper-agent LoRA and clean control adapter; untrusted until verified.",
    ),
    OrganismSpec(
        key="pythia_70m_timeline",
        repo_id="EleutherAI/pythia-70m",
        role="pretraining_timeline",
        revisions=PYTHIA_DEFAULT_CHECKPOINTS,
        community_untrusted=False,
        notes=(
            "Six pretraining checkpoints (~282 MB model.safetensors each, ~1.7 GB total) "
            "for E6 timeline demonstration."
        ),
    ),
)

_CATALOG_BY_KEY: dict[str, OrganismSpec] = {spec.key: spec for spec in ORGANISM_CATALOG}


def verify_downloaded_dir_safe(model_dir: Path) -> None:
    """Verify a local model or adapter directory contains only safe non-code weight files."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    offending = [
        p.relative_to(model_dir)
        for p in model_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in UNSAFE_FILE_SUFFIXES
    ]
    if offending:
        joined = ", ".join(str(p) for p in offending)
        raise ValueError(
            f"Unsafe weight or code file(s) found in {model_dir}: {joined}. "
            "Only safetensors, JSON configs, and tokenizer files are permitted."
        )

    config_path = model_dir / "config.json"
    if config_path.is_file():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(cfg, dict) and cfg.get("auto_map"):
            raise ValueError(
                f"Model config in {model_dir} declares custom remote code (`auto_map`), "
                "which is refused by default."
            )

    safetensors = list(model_dir.rglob("*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No .safetensors weight file found in {model_dir}.")


def _archive_non_peft_adapter_config(target_dir: Path) -> None:
    """Rename leftover non-PEFT `adapter_config.json` (e.g. Apple MLX) alongside fused weights."""
    cfg_path = target_dir / "config.json"
    adap_path = target_dir / "adapter_config.json"
    if not (cfg_path.is_file() and adap_path.is_file()):
        return
    try:
        adap_cfg = json.loads(adap_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(adap_cfg, dict) and "peft_type" not in adap_cfg:
        dest = target_dir / "mlx_adapter_config.json"
        if dest.exists():
            adap_path.unlink()
        else:
            adap_path.rename(dest)


def fetch_repo_safely(
    repo_id: str,
    target_dir: Path,
    revision: str = "main",
) -> Path:
    """Download `repo_id` at `revision` using safe file-pattern filters only."""
    from huggingface_hub import snapshot_download

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=[*SAFE_ALLOW_PATTERNS, "chat_template.jinja", "merges.txt", "vocab*"],
        ignore_patterns=[f"*{ext}" for ext in UNSAFE_FILE_SUFFIXES],
    )
    _archive_non_peft_adapter_config(target_dir)
    verify_downloaded_dir_safe(target_dir)
    return target_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m organisms.fetch",
        description=(
            "Fetch candidate test models safely (safetensors, JSON, and tokenizer files only). "
            "Requires --confirm-download."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered test-model specs and exit without downloading.",
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        default=[],
        help="Catalog key to download (repeatable). Defaults to all catalog entries.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Explicit Hugging Face repo ID to fetch safely.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision branch or tag when --repo-id is specified.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Local destination folder for downloaded model weights (git-ignored).",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Confirm owner approval to download model weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list:
        for spec in ORGANISM_CATALOG:
            revs = ",".join(spec.revisions)
            trust = "untrusted-community" if spec.community_untrusted else "standard"
            print(f"{spec.key:22s} repo={spec.repo_id:42s} revisions={revs} [{trust}]")
        return 0

    if not args.confirm_download:
        print(
            "Refusing to download model weights without --confirm-download. "
            "Ask the project owner for approval first (CLAUDE.md section 8), "
            "then pass --confirm-download.",
            file=sys.stderr,
        )
        return 2

    downloaded: list[Path] = []
    if args.repo_id:
        safe_slug = args.repo_id.replace("/", "__")
        dest = args.models_dir / safe_slug / args.revision
        downloaded.append(fetch_repo_safely(args.repo_id, dest, revision=args.revision))
    else:
        selected = (
            [_CATALOG_BY_KEY[k] for k in args.keys] if args.keys else list(ORGANISM_CATALOG)
        )
        for spec in selected:
            safe_slug = spec.repo_id.replace("/", "__")
            for rev in spec.revisions:
                dest = args.models_dir / safe_slug / rev
                downloaded.append(fetch_repo_safely(spec.repo_id, dest, revision=rev))

    print(f"Downloaded and verified {len(downloaded)} safe model folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
