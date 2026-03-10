#!/usr/bin/env python3
"""
fit_bkt_params.py — Fit BKT parameters on real data using EM (Fix #4)
=======================================================================
Fix #4: BKT parameters (p_init, p_transit, p_slip, p_guess) were hardcoded.
This script fits them via Baum-Welch EM on real student data, producing
data-driven parameters with reported standard errors.

Supports three data sources (in order of preference):
  1. EdNet (preferred): concept-tagged, real timestamps, large scale
  2. ASSISTments: skill-name based, widely used in KT literature
  3. Both combined: for robustness across domains

Outputs:
  config/bkt_params.json    — per-skill and global fitted parameters
  config/base_config.yaml   — auto-updated with global params

Usage:
    # EdNet (preferred)
    python scripts/fit_bkt_params.py --ednet data/processed/ednet

    # ASSISTments (fallback)
    python scripts/fit_bkt_params.py --assistments data/raw/assistments.csv

    # Both combined
    python scripts/fit_bkt_params.py \\
        --ednet       data/processed/ednet \\
        --assistments data/raw/assistments.csv

    # Quick fit (fewer iterations, for sanity check)
    python scripts/fit_bkt_params.py --assistments data/raw/assistments.csv --quick
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BKT HMM: Forward / Backward / EM
# ─────────────────────────────────────────────────────────────────────────────

def bkt_forward(obs: List[int], p_init, p_transit, p_slip, p_guess) -> np.ndarray:
    T     = len(obs)
    alpha = np.zeros((T, 2))

    def emit(state, o):
        return (1 - p_slip if o == 1 else p_slip) if state == 1 else (p_guess if o == 1 else 1 - p_guess)

    alpha[0, 0] = (1 - p_init) * emit(0, obs[0])
    alpha[0, 1] = p_init       * emit(1, obs[0])

    for t in range(1, T):
        alpha[t, 0] = alpha[t-1, 0] * (1 - p_transit) * emit(0, obs[t])
        alpha[t, 1] = (alpha[t-1, 0] * p_transit + alpha[t-1, 1] * 1.0) * emit(1, obs[t])
    return alpha


def bkt_backward(obs: List[int], p_transit, p_slip, p_guess) -> np.ndarray:
    T    = len(obs)
    beta = np.zeros((T, 2))
    beta[T-1, :] = 1.0

    def emit(state, o):
        return (1 - p_slip if o == 1 else p_slip) if state == 1 else (p_guess if o == 1 else 1 - p_guess)

    for t in range(T-2, -1, -1):
        beta[t, 0] = ((1 - p_transit) * emit(0, obs[t+1]) * beta[t+1, 0] +
                      p_transit       * emit(1, obs[t+1]) * beta[t+1, 1])
        beta[t, 1] = (1.0             * emit(1, obs[t+1]) * beta[t+1, 1])
    return beta


def fit_bkt_em(
    sequences: List[List[int]],
    n_iter: int = 50,
    n_restarts: int = 3,
    init_params: Optional[Dict] = None,
) -> Tuple[Dict[str, float], float]:
    """
    Fit BKT parameters using Baum-Welch EM with random restarts.

    Fix #4: Returns both fitted parameters AND bootstrap standard errors.

    Returns:
        (params_dict, log_likelihood)
        params_dict keys: p_init, p_transit, p_slip, p_guess,
                          se_p_init, se_p_transit, se_p_slip, se_p_guess
    """
    clip = lambda x: float(np.clip(x, 0.001, 0.999))

    def run_em(init):
        p_init    = init["p_init"]
        p_transit = init["p_transit"]
        p_slip    = init["p_slip"]
        p_guess   = init["p_guess"]
        prev_ll   = -np.inf

        for iteration in range(n_iter):
            si_m = si_t = st = ss_w = ss_t = sg_r = sg_t = 0.0
            total_ll = 0.0

            for obs in sequences:
                if len(obs) < 2:
                    continue
                obs = list(obs)
                T   = len(obs)

                alpha = bkt_forward(obs, p_init, p_transit, p_slip, p_guess)
                beta  = bkt_backward(obs, p_transit, p_slip, p_guess)

                scale      = alpha.sum(axis=1, keepdims=True).clip(min=1e-300)
                alpha_norm = alpha / scale
                beta_norm  = beta  / scale

                total_ll += float(np.log(scale).sum())

                gamma = alpha_norm * beta_norm
                gamma /= gamma.sum(axis=1, keepdims=True).clip(min=1e-300)

                si_m += gamma[0, 1]
                si_t += 1.0

                def emit(state, o):
                    return (1 - p_slip if o == 1 else p_slip) if state == 1 else (p_guess if o == 1 else 1 - p_guess)

                for t in range(T - 1):
                    xi_d = (
                        alpha_norm[t, 0] * (1 - p_transit) * emit(0, obs[t+1]) * beta_norm[t+1, 0] +
                        alpha_norm[t, 0] * p_transit        * emit(1, obs[t+1]) * beta_norm[t+1, 1] +
                        alpha_norm[t, 1] * 1.0              * emit(1, obs[t+1]) * beta_norm[t+1, 1]
                    )
                    xi_d = max(xi_d, 1e-300)
                    xi_t = (alpha_norm[t, 0] * p_transit * emit(1, obs[t+1]) * beta_norm[t+1, 1]) / xi_d
                    st  += xi_t
                    si_t_local = gamma[t, 0]

                for t in range(T):
                    if obs[t] == 0:
                        ss_w += gamma[t, 1]
                        ss_t += gamma[t, 1]
                    else:
                        sg_r += gamma[t, 0]
                        sg_t += gamma[t, 0]
                        ss_t += gamma[t, 1]

            if si_t > 0:   p_init    = clip(si_m / si_t)
            if si_t > 0:   p_transit = clip(st   / max(si_t, 1e-9))
            if ss_t > 0:   p_slip    = clip(ss_w / max(ss_t, 1e-9))
            if sg_t > 0:   p_guess   = clip(sg_r / max(sg_t, 1e-9))

            if abs(total_ll - prev_ll) < 1e-4:
                logger.debug(f"  EM converged at iteration {iteration+1}, LL={total_ll:.2f}")
                break
            prev_ll = total_ll

        return {"p_init": p_init, "p_transit": p_transit,
                "p_slip": p_slip, "p_guess": p_guess}, total_ll

    # Random restarts: pick best log-likelihood
    rng = np.random.RandomState(42)
    best_params, best_ll = None, -np.inf

    inits = []
    if init_params:
        inits.append(init_params)
    for _ in range(n_restarts):
        inits.append({
            "p_init":    float(rng.uniform(0.1, 0.5)),
            "p_transit": float(rng.uniform(0.05, 0.4)),
            "p_slip":    float(rng.uniform(0.05, 0.3)),
            "p_guess":   float(rng.uniform(0.1, 0.4)),
        })

    for init in inits:
        params, ll = run_em(init)
        if ll > best_ll:
            best_ll, best_params = ll, params

    # Bootstrap standard errors (Fix #4)
    n_boot = 50
    boot_params = {k: [] for k in best_params}
    rng2 = np.random.RandomState(99)

    sequences_arr = [s for s in sequences if len(s) >= 2]
    n_seq = len(sequences_arr)

    if n_seq >= 10:
        for _ in range(n_boot):
            boot_idx  = rng2.choice(n_seq, n_seq, replace=True)
            boot_seqs = [sequences_arr[i] for i in boot_idx]
            bp, _     = run_em(best_params)  # warm-start from best
            for k, v in bp.items():
                boot_params[k].append(v)

        for k in best_params:
            best_params[f"se_{k}"] = round(float(np.std(boot_params[k])), 4)
    else:
        for k in list(best_params.keys()):
            best_params[f"se_{k}"] = None

    best_params["log_likelihood"] = round(best_ll, 2)
    best_params["n_sequences"]    = len(sequences_arr)
    return best_params, best_ll


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_assistments(path: str) -> Dict[str, List[List[int]]]:
    """Group ASSISTments by skill → list of per-student correctness sequences."""
    df = pd.read_csv(path, low_memory=False)
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("user_id", "student_id"):     col_map[col] = "user_id"
        elif lc in ("skill_name", "skill"):     col_map[col] = "skill_name"
        elif lc == "correct":                   col_map[col] = "correct"
        elif lc in ("order_id", "timestamp"):   col_map[col] = "order_id"
    df = df.rename(columns=col_map)

    required = {"user_id", "skill_name", "correct"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "order_id"])

    skill_data: Dict[str, Dict[str, List[int]]] = {}
    for _, row in df.iterrows():
        skill = str(row["skill_name"]).strip()
        uid   = str(row["user_id"])
        if skill not in skill_data:
            skill_data[skill] = {}
        if uid not in skill_data[skill]:
            skill_data[skill][uid] = []
        skill_data[skill][uid].append(int(row["correct"]))

    # Convert to list of sequences per skill
    return {
        skill: [seq for seq in student_seqs.values() if len(seq) >= 2]
        for skill, student_seqs in skill_data.items()
    }


def load_ednet(processed_dir: str) -> Dict[str, List[List[int]]]:
    """
    Load EdNet interactions grouped by concept tag → list of per-student sequences.
    EdNet preferred over ASSISTments because:
      - Concept tags are explicit (no proxy mapping needed)
      - Real timestamps available for forgetting model validation
    """
    p = Path(processed_dir)
    interactions_path = p / "interactions.json"
    index_path        = p / "student_index.json"

    if not interactions_path.exists():
        raise FileNotFoundError(
            f"EdNet interactions not found at {interactions_path}. "
            "Run: python data/pipelines/ednet_pipeline.py"
        )

    with open(interactions_path) as f:
        all_interactions = json.load(f)

    with open(index_path) as f:
        student_index = json.load(f)

    concept_data: Dict[str, Dict[str, List[int]]] = {}

    for uid, idxs in student_index.items():
        student_ints = sorted(
            [all_interactions[i] for i in idxs],
            key=lambda x: x.get("timestamp", 0)
        )
        # Group by concept
        per_concept: Dict[str, List[int]] = {}
        for interaction in student_ints:
            cid     = interaction.get("concept_id", "unknown")
            correct = int(interaction.get("correct", 0))
            if cid not in per_concept:
                per_concept[cid] = []
            per_concept[cid].append(correct)

        for cid, seq in per_concept.items():
            if len(seq) >= 2:
                if cid not in concept_data:
                    concept_data[cid] = {}
                concept_data[cid][uid] = seq

    return {
        cid: list(student_seqs.values())
        for cid, student_seqs in concept_data.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main fitting routine
# ─────────────────────────────────────────────────────────────────────────────

def fit_all(
    skill_sequences: Dict[str, List[List[int]]],
    n_iter: int = 50,
    n_restarts: int = 3,
    min_sequences: int = 20,
) -> Dict:
    """Fit BKT parameters for each skill and compute global medians."""

    skill_params: Dict[str, Dict] = {}
    total_skills = sum(1 for seqs in skill_sequences.values() if len(seqs) >= min_sequences)
    logger.info(f"Fitting BKT on {total_skills} skills (≥{min_sequences} student sequences)...")

    for skill, sequences in skill_sequences.items():
        if len(sequences) < min_sequences:
            continue

        params, ll = fit_bkt_em(sequences, n_iter=n_iter, n_restarts=n_restarts)
        skill_params[skill] = params
        logger.debug(
            f"  {skill[:40]:<40}: "
            f"p_init={params['p_init']:.3f} "
            f"p_transit={params['p_transit']:.3f} "
            f"p_slip={params['p_slip']:.3f} "
            f"p_guess={params['p_guess']:.3f} "
            f"LL={ll:.1f}"
        )

    if not skill_params:
        logger.warning(
            f"No skills had ≥{min_sequences} sequences. "
            "Using safe defaults. Lower --min-sequences to fit on more skills."
        )
        return {
            "global_params": {
                "p_init": 0.10, "p_transit": 0.15,
                "p_slip": 0.10, "p_guess":   0.20,
            },
            "per_skill_params": {},
            "n_skills_fitted": 0,
        }

    # Global params: median across all skills
    param_keys   = ["p_init", "p_transit", "p_slip", "p_guess"]
    se_keys      = [f"se_{k}" for k in param_keys]
    global_params = {}
    global_se     = {}

    for k in param_keys:
        vals = [v[k] for v in skill_params.values() if k in v]
        global_params[k] = round(float(np.median(vals)), 4)

    for k in se_keys:
        vals = [v[k] for v in skill_params.values() if v.get(k) is not None]
        global_se[k] = round(float(np.median(vals)), 4) if vals else None

    global_params.update(global_se)

    return {
        "global_params":    global_params,
        "per_skill_params": skill_params,
        "n_skills_fitted":  len(skill_params),
        "note": (
            "Fitted by EM (Baum-Welch) with random restarts. "
            "Global params = median across all skills with ≥{} student sequences. "
            "se_* = bootstrap standard errors (n_boot=50).".format(min_sequences)
        ),
    }


def update_config_yaml(global_params: Dict, config_path: str = "config/base_config.yaml") -> None:
    """
    Auto-update base_config.yaml with fitted BKT parameters.
    Only updates the karma.bkt section; all other config is preserved.
    """
    import re
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Config not found at {config_path} — skipping auto-update")
        return

    text = p.read_text()
    for key in ["p_init", "p_transit", "p_slip", "p_guess"]:
        val     = global_params.get(key)
        se      = global_params.get(f"se_{key}")
        if val is None:
            continue
        pattern = rf"(\s+{key}:\s*)[\d.]+"
        repl    = rf"\g<1>{val}"
        new_text, n = re.subn(pattern, repl, text)
        if n > 0:
            text = new_text
            se_str = f" ± {se}" if se else ""
            logger.info(f"  Updated {key} = {val}{se_str}")
        else:
            logger.warning(f"  Could not find {key} in {config_path}")

    p.write_text(text)
    logger.info(f"Config updated: {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fit BKT parameters via EM (Fix #4: replaces hardcoded params)"
    )
    parser.add_argument("--ednet",       default=None,
                        help="Path to processed EdNet dir (data/processed/ednet)")
    parser.add_argument("--assistments", default=None,
                        help="Path to ASSISTments CSV (data/raw/assistments.csv)")
    parser.add_argument("--output",      default="config/bkt_params.json")
    parser.add_argument("--n-iter",      type=int, default=50,
                        help="EM iterations per skill")
    parser.add_argument("--n-restarts",  type=int, default=3,
                        help="Random restarts (pick best log-likelihood)")
    parser.add_argument("--min-seq",     type=int, default=20,
                        help="Minimum student sequences per skill to include")
    parser.add_argument("--quick",       action="store_true",
                        help="Quick mode: 10 iterations, 1 restart (for sanity check)")
    parser.add_argument("--no-update-config", action="store_true",
                        help="Do not auto-update base_config.yaml")
    args = parser.parse_args()

    if args.quick:
        args.n_iter     = 10
        args.n_restarts = 1

    if not args.ednet and not args.assistments:
        parser.error("Provide at least one of --ednet or --assistments")

    # Load sequences
    skill_sequences: Dict[str, List[List[int]]] = {}

    if args.ednet:
        logger.info(f"Loading EdNet from {args.ednet}...")
        try:
            ednet_seqs = load_ednet(args.ednet)
            skill_sequences.update(ednet_seqs)
            logger.info(f"  EdNet: {len(ednet_seqs)} concept tags loaded")
        except FileNotFoundError as e:
            logger.error(str(e))

    if args.assistments:
        logger.info(f"Loading ASSISTments from {args.assistments}...")
        try:
            assist_seqs = load_assistments(args.assistments)
            # Namespace to avoid collision with EdNet tags
            assist_ns = {f"assist_{k}": v for k, v in assist_seqs.items()}
            skill_sequences.update(assist_ns)
            logger.info(f"  ASSISTments: {len(assist_ns)} skills loaded")
        except Exception as e:
            logger.error(f"ASSISTments load failed: {e}")

    if not skill_sequences:
        logger.error("No data loaded. Check file paths.")
        sys.exit(1)

    total_sequences = sum(len(v) for v in skill_sequences.values())
    logger.info(f"Total: {len(skill_sequences)} skills, {total_sequences} student sequences")

    # Fit
    results = fit_all(
        skill_sequences,
        n_iter=args.n_iter,
        n_restarts=args.n_restarts,
        min_sequences=args.min_seq,
    )

    # Print summary
    g = results["global_params"]
    print(f"\n{'='*65}")
    print(f"Fitted Global BKT Parameters (median across {results['n_skills_fitted']} skills)")
    print(f"{'='*65}")
    for k in ["p_init", "p_transit", "p_slip", "p_guess"]:
        se_key = f"se_{k}"
        se_str = f" ± {g[se_key]:.4f}" if g.get(se_key) else ""
        print(f"  {k:<12} = {g[k]:.4f}{se_str}")

    print(f"\nPaste into config/base_config.yaml under karma.bkt:")
    print(f"  bkt:")
    for k in ["p_init", "p_transit", "p_slip", "p_guess"]:
        print(f"    {k}: {g[k]}")
    print(f"    mastery_threshold: 0.95")
    print(f"{'='*65}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full per-skill params saved to: {args.output}")

    # Auto-update config
    if not args.no_update_config:
        update_config_yaml(g)


if __name__ == "__main__":
    main()