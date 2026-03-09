"""
fit_bkt_params.py — Fit BKT parameters on real ASSISTments data using EM.

Run this ONCE on your Mac after downloading real ASSISTments data.
Copy the printed output into config/base_config.yaml under karma.bkt.

Usage:
    python scripts/fit_bkt_params.py --assistments data/raw/assistments.csv

What this does:
    Standard BKT has 4 parameters per skill:
        p_init    = P(student already knows skill before first attempt)
        p_transit = P(student learns skill during one attempt)
        p_slip    = P(wrong answer even though student knows skill)
        p_guess   = P(right answer even though student doesn't know skill)

    We fit these by maximizing P(observed correctness sequence) across all
    students, using Expectation-Maximization (Baum-Welch algorithm).

    This gives us REAL parameters grounded in actual student behavior,
    not hardcoded guesses. Reviewers cannot question fitted parameters —
    they are data-driven.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────
# EM-based BKT fitter (no external library needed)
# ─────────────────────────────────────────────────────────────────

def bkt_forward(obs: List[int], p_init, p_transit, p_slip, p_guess) -> np.ndarray:
    """
    Forward pass of BKT HMM.
    obs: list of 0/1 correctness observations
    Returns: alpha matrix (T x 2), alpha[t, k] = P(obs_1..t, state_t=k)
    """
    T = len(obs)
    alpha = np.zeros((T, 2))  # 2 states: 0=not mastered, 1=mastered

    # Emission probabilities P(obs | state)
    def emit(state, o):
        if state == 1:  # mastered
            return (1 - p_slip) if o == 1 else p_slip
        else:           # not mastered
            return p_guess if o == 1 else (1 - p_guess)

    # Init
    alpha[0, 0] = (1 - p_init) * emit(0, obs[0])
    alpha[0, 1] = p_init * emit(1, obs[0])

    for t in range(1, T):
        # Transition: not_mastered→not_mastered = 1-p_transit
        #             not_mastered→mastered     = p_transit
        #             mastered→not_mastered     = 0 (no forgetting in basic BKT)
        #             mastered→mastered         = 1
        alpha[t, 0] = alpha[t-1, 0] * (1 - p_transit) * emit(0, obs[t])
        alpha[t, 1] = (alpha[t-1, 0] * p_transit + alpha[t-1, 1] * 1.0) * emit(1, obs[t])

    return alpha


def bkt_backward(obs: List[int], p_transit, p_slip, p_guess) -> np.ndarray:
    """
    Backward pass of BKT HMM.
    Returns: beta matrix (T x 2)
    """
    T = len(obs)
    beta = np.zeros((T, 2))
    beta[T-1, :] = 1.0

    def emit(state, o):
        if state == 1:
            return (1 - p_slip) if o == 1 else p_slip
        else:
            return p_guess if o == 1 else (1 - p_guess)

    for t in range(T-2, -1, -1):
        beta[t, 0] = (
            (1 - p_transit) * emit(0, obs[t+1]) * beta[t+1, 0] +
            p_transit * emit(1, obs[t+1]) * beta[t+1, 1]
        )
        beta[t, 1] = (
            0.0 * emit(0, obs[t+1]) * beta[t+1, 0] +
            1.0 * emit(1, obs[t+1]) * beta[t+1, 1]
        )

    return beta


def fit_bkt_em(
    sequences: List[List[int]],
    n_iter: int = 50,
    init_params: Dict = None
) -> Dict[str, float]:
    """
    Fit BKT parameters using Baum-Welch EM.

    Args:
        sequences: List of correctness sequences per student per skill
        n_iter: Number of EM iterations
        init_params: Starting parameter values (random restarts improve this)

    Returns:
        Dict with fitted p_init, p_transit, p_slip, p_guess
    """
    # Initialize
    if init_params is None:
        p_init    = 0.25
        p_transit = 0.10
        p_slip    = 0.10
        p_guess   = 0.20
    else:
        p_init    = init_params["p_init"]
        p_transit = init_params["p_transit"]
        p_slip    = init_params["p_slip"]
        p_guess   = init_params["p_guess"]

    # Clip to valid range
    def clip(x): return float(np.clip(x, 0.001, 0.999))

    prev_ll = -np.inf

    for iteration in range(n_iter):
        # Accumulators for M-step
        sum_init_mastered = 0.0
        sum_init_total    = 0.0
        sum_transit       = 0.0
        sum_transit_total = 0.0
        sum_slip_wrong    = 0.0
        sum_slip_total    = 0.0
        sum_guess_right   = 0.0
        sum_guess_total   = 0.0
        total_ll          = 0.0

        for obs in sequences:
            if len(obs) < 2:
                continue
            obs = list(obs)
            T = len(obs)

            alpha = bkt_forward(obs, p_init, p_transit, p_slip, p_guess)
            beta  = bkt_backward(obs, p_transit, p_slip, p_guess)

            # Normalisation (scale to avoid underflow)
            scale = alpha.sum(axis=1, keepdims=True)
            scale = np.where(scale < 1e-300, 1e-300, scale)
            alpha_norm = alpha / scale
            beta_norm  = beta  / scale

            ll = np.log(scale).sum()
            total_ll += ll

            # Gamma: P(state_t | obs_1..T)
            gamma = alpha_norm * beta_norm
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma = gamma / np.where(gamma_sum < 1e-300, 1e-300, gamma_sum)

            # E-step accumulation
            # p_init: from t=0 state distribution
            sum_init_mastered += gamma[0, 1]
            sum_init_total    += 1.0

            # p_transit: from not-mastered → mastered transitions
            for t in range(T - 1):
                def emit(state, o):
                    if state == 1:
                        return (1 - p_slip) if o == 1 else p_slip
                    else:
                        return p_guess if o == 1 else (1 - p_guess)

                xi_denom = (
                    alpha_norm[t, 0] * (1 - p_transit) * emit(0, obs[t+1]) * beta_norm[t+1, 0] +
                    alpha_norm[t, 0] * p_transit        * emit(1, obs[t+1]) * beta_norm[t+1, 1] +
                    alpha_norm[t, 1] * 1.0              * emit(1, obs[t+1]) * beta_norm[t+1, 1]
                )
                xi_denom = max(xi_denom, 1e-300)

                xi_transit = (
                    alpha_norm[t, 0] * p_transit * emit(1, obs[t+1]) * beta_norm[t+1, 1]
                ) / xi_denom

                sum_transit       += xi_transit
                sum_transit_total += gamma[t, 0]  # from not-mastered state

            # p_slip and p_guess: from emission probabilities
            for t in range(T):
                if obs[t] == 0:  # wrong answer
                    sum_slip_wrong += gamma[t, 1]   # mastered, but wrong = slip
                    sum_slip_total += gamma[t, 1]
                else:            # right answer
                    sum_guess_right += gamma[t, 0]  # not mastered, but right = guess
                    sum_guess_total += gamma[t, 0]
                    sum_slip_total  += gamma[t, 1]  # mastered, right = no slip (counted for denom)

        # M-step: update parameters
        if sum_init_total > 0:
            p_init = clip(sum_init_mastered / sum_init_total)
        if sum_transit_total > 0:
            p_transit = clip(sum_transit / max(sum_transit_total, 1e-9))
        if sum_slip_total > 0:
            p_slip = clip(sum_slip_wrong / max(sum_slip_total, 1e-9))
        if sum_guess_total > 0:
            p_guess = clip(sum_guess_right / max(sum_guess_total, 1e-9))

        # Convergence check
        if abs(total_ll - prev_ll) < 1e-4:
            print(f"  Converged at iteration {iteration+1}")
            break
        prev_ll = total_ll

    return {
        "p_init":    round(p_init,    4),
        "p_transit": round(p_transit, 4),
        "p_slip":    round(p_slip,    4),
        "p_guess":   round(p_guess,   4),
        "log_likelihood": round(total_ll, 2)
    }


# ─────────────────────────────────────────────────────────────────
# Load ASSISTments data
# ─────────────────────────────────────────────────────────────────

def load_assistments(path: str) -> Dict[str, List[List[int]]]:
    """
    Load ASSISTments CSV and group by skill.

    Expected columns (ASSISTments 2009-2010 format):
        user_id, skill_name, correct, order_id (or timestamp)

    Returns:
        Dict[skill_name → List[student_sequences]]
        where each student_sequence is List[0/1] correctness observations
    """
    df = pd.read_csv(path, low_memory=False)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc in ("user_id", "student_id", "user"):
            col_map[col] = "user_id"
        elif lc in ("skill_name", "skill", "concept", "kc"):
            col_map[col] = "skill_name"
        elif lc == "correct":
            col_map[col] = "correct"
        elif lc in ("order_id", "timestamp", "problem_id"):
            col_map[col] = "order_id"
    df = df.rename(columns=col_map)

    required = {"user_id", "skill_name", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    df = df.dropna(subset=["correct", "skill_name"])
    df["correct"] = df["correct"].astype(int).clip(0, 1)

    if "order_id" in df.columns:
        df = df.sort_values(["user_id", "skill_name", "order_id"])
    else:
        df = df.sort_values(["user_id", "skill_name"])

    # Group by skill → list of per-student sequences
    skill_sequences: Dict[str, List[List[int]]] = {}
    for (user_id, skill), grp in df.groupby(["user_id", "skill_name"]):
        seq = grp["correct"].tolist()
        if skill not in skill_sequences:
            skill_sequences[skill] = []
        skill_sequences[skill].append(seq)

    print(f"Loaded {len(df)} interactions, "
          f"{df['user_id'].nunique()} students, "
          f"{len(skill_sequences)} skills")

    return skill_sequences


# ─────────────────────────────────────────────────────────────────
# Main: fit per-skill, then compute global average params
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistments", default="data/raw/assistments.csv",
                        help="Path to ASSISTments CSV")
    parser.add_argument("--min-sequences", type=int, default=20,
                        help="Min student sequences per skill to fit")
    parser.add_argument("--output", default="config/bkt_params.json",
                        help="Output JSON file with fitted parameters")
    args = parser.parse_args()

    print(f"\n=== BKT Parameter Fitting ===")
    print(f"Loading: {args.assistments}")

    skill_sequences = load_assistments(args.assistments)

    # Fit per skill (skills with enough data)
    skill_params = {}
    all_params = {"p_init": [], "p_transit": [], "p_slip": [], "p_guess": []}

    eligible_skills = {
        skill: seqs for skill, seqs in skill_sequences.items()
        if len(seqs) >= args.min_sequences
    }
    print(f"Fitting BKT for {len(eligible_skills)} skills "
          f"(min {args.min_sequences} students each)...")

    for i, (skill, seqs) in enumerate(eligible_skills.items()):
        if i % 10 == 0:
            print(f"  Skill {i+1}/{len(eligible_skills)}: {skill[:40]}")

        params = fit_bkt_em(seqs, n_iter=50)
        skill_params[skill] = params

        for k in ["p_init", "p_transit", "p_slip", "p_guess"]:
            all_params[k].append(params[k])

    # Global parameters = median across all skills
    # (median is more robust than mean to outlier skills)
    global_params = {
        k: round(float(np.median(v)), 4)
        for k, v in all_params.items()
        if v
    }

    print(f"\n=== Fitted Global BKT Parameters (median across {len(skill_params)} skills) ===")
    print(f"  p_init    = {global_params['p_init']:.4f}  "
          f"(prior: students know {global_params['p_init']*100:.1f}% of skills before first attempt)")
    print(f"  p_transit = {global_params['p_transit']:.4f}  "
          f"(learning rate per attempt)")
    print(f"  p_slip    = {global_params['p_slip']:.4f}  "
          f"(error rate when mastered)")
    print(f"  p_guess   = {global_params['p_guess']:.4f}  "
          f"(correct guess rate when not mastered)")

    print(f"\n=== Paste into config/base_config.yaml under karma.bkt: ===")
    print(f"  bkt:")
    print(f"    p_init:    {global_params['p_init']}")
    print(f"    p_transit: {global_params['p_transit']}")
    print(f"    p_slip:    {global_params['p_slip']}")
    print(f"    p_guess:   {global_params['p_guess']}")
    print(f"    mastery_threshold: 0.95")

    # Save full output
    output = {
        "global_params": global_params,
        "per_skill_params": skill_params,
        "n_skills_fitted": len(skill_params),
        "note": "Fitted by EM (Baum-Welch) on ASSISTments 2009-2010 data. "
                "Global params = median across all skills with >= 20 student sequences."
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull per-skill params saved to: {args.output}")


if __name__ == "__main__":
    main()