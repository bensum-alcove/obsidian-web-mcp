#!/usr/bin/env python3
"""run_holdout_eval.py -- scores evals/eval-set-v3-holdout.yaml, a fresh,
hashed set of paraphrase/entities_aliases questions built and frozen for
vault-retrieval-candidate-recall-v1, kept deliberately separate from
eval-set-v3.yaml (never used to tune the implementation).

Reuses run_eval_v3.py's scoring/aggregation machinery (score_entry,
run_tool_eval_v3, stratify_categories) by importing it as a module -- this
file only supplies a different YAML path, a different hash-freeze file, and
its own (much smaller) report/history location. eval-set-v3.yaml, its
CORPUS_HASH, and its deploy-gates.yaml are never read or written by this
script.

Usage:
    uv run python3 evals/run_holdout_eval.py [--freeze-corpus]

--freeze-corpus: only needed once, before the final implementation is
evaluated against this file. Freezing AFTER looking at results defeats the
purpose of a holdout -- this script warns loudly (corpus hash drift) if the
file changes after being frozen once.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_HOLDOUT_PATH = REPO_ROOT / "evals" / "eval-set-v3-holdout.yaml"
HISTORY_DIR = REPO_ROOT / "evals" / "history-v3-holdout"
CORPUS_HASH_PATH = HISTORY_DIR / "HOLDOUT_CORPUS_HASH"


def _load_v3_module():
    spec = importlib.util.spec_from_file_location("run_eval_v3", REPO_ROOT / "evals" / "run_eval_v3.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    freeze = "--freeze-corpus" in sys.argv
    v3 = _load_v3_module()

    config, src_root = v3._v2._load_live_tools()
    print(f"Tool source: {src_root}", file=sys.stderr)
    print(f"VAULT_PATH: {config.VAULT_PATH}", file=sys.stderr)

    raw = EVAL_SET_HOLDOUT_PATH.read_text(encoding="utf-8")
    eval_set = yaml.safe_load(raw)
    for entry in eval_set:
        entry.setdefault("expected_paths", [])
        entry.setdefault("decoy_paths", [])
        entry.setdefault("negative", False)
        entry.setdefault("require_all", False)

    corpus_hash = v3.compute_corpus_hash(raw)
    stratification = v3.stratify_categories(eval_set)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    existing = None
    if CORPUS_HASH_PATH.is_file():
        for line in CORPUS_HASH_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("sha256:"):
                existing = line.split("sha256:", 1)[1].strip()

    if freeze or existing is None:
        CORPUS_HASH_PATH.write_text(
            f"sha256: {corpus_hash}\n"
            f"frozen_at: '{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}'\n"
            f"n_questions: {len(eval_set)}\n",
            encoding="utf-8",
        )
        just_frozen = True
        drift = False
    else:
        just_frozen = False
        drift = existing != corpus_hash
        if drift:
            print(
                f"[error] HOLDOUT corpus hash drift: frozen={existing} current={corpus_hash} -- "
                "eval-set-v3-holdout.yaml changed after being frozen. This defeats the purpose "
                "of a holdout; investigate before trusting any score against this file.",
                file=sys.stderr,
            )

    runners = v3._v2.build_tool_runners(config, src_root)
    if not runners:
        print("[error] no retrieval tools importable -- aborting", file=sys.stderr)
        sys.exit(1)

    results = {}
    for tool_name, (tool_fn, kwargs_fn) in runners.items():
        print(f"Scoring {tool_name} (holdout)...", file=sys.stderr)
        results[tool_name] = v3.run_tool_eval_v3(tool_fn, kwargs_fn, eval_set, config.VAULT_PATH)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    with open(HISTORY_DIR / f"{date_str}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nHoldout corpus hash: sha256:{corpus_hash}"
          f"{' (just frozen)' if just_frozen else ''}"
          f"{' DRIFTED' if drift else ''}", file=sys.stderr)
    print(json.dumps({"stratification": stratification, "results": results}, indent=2))


if __name__ == "__main__":
    main()
